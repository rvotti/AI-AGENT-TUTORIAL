import os
import streamlit as st

from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

# LangChain & Google Gemini Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent

# Import custom tools from your tools.py file
from tools import search_tool, wiki_tool, save_tool

# 1. Load your GOOGLE_API_KEY from the .env file
load_dotenv()


def get_google_api_key() -> str | None:
    """Return the configured Gemini API key, or None if it is missing."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        return api_key.strip()
    return None


def is_leaked_key_error(error: Exception) -> bool:
    message = str(error).lower()
    return "reported as leaked" in message or (
        "permission_denied" in message and "api key" in message
    )

# 2. Define the Research Schema (Matches the video's logic)
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: List[str]
    tools_used: List[str]

# Create the parser to force Gemini into a specific output format
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Setup Streamlit Page
st.set_page_config(page_title="AI Research Agent", page_icon="🔍")
st.title("🔍 AI Research Agent")
st.markdown(
    """
    <style>
    div[data-testid="stChatInput"] {
        padding: 0.75rem 1rem 1rem;
        background: rgba(255, 255, 255, 0.96);
        border-top: 1px solid #d9e2ec;
    }

    div[data-testid="stChatInput"] textarea {
        min-height: 3.25rem !important;
        padding: 0.85rem 3rem 0.85rem 1rem !important;
        border: 2px solid #7c93ad !important;
        border-radius: 8px !important;
        background: #ffffff !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08) !important;
        color: #111827 !important;
        font-size: 1rem !important;
    }

    div[data-testid="stChatInput"] textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.18) !important;
    }

    div[data-testid="stChatInput"] button {
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Cache the agent initialization so it doesn't reload on every UI interaction
@st.cache_resource
def get_agent(api_key: str):
    # 3. Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key,
    )
    
    # 4. Define the System Prompt
    system_prompt = (
        "You are a professional research assistant. "
        "Answer the user query using the tools provided. "
        f"You MUST output your final answer in the following format: \n{parser.get_format_instructions()}"
    )
    
    # 5. Combine Tools
    tools = [search_tool, wiki_tool, save_tool]
    
    # 6. Build and Execute the Agent
    return create_agent(model=llm, tools=tools, system_prompt=system_prompt)

google_api_key = get_google_api_key()
if not google_api_key:
    st.error("Missing GOOGLE_API_KEY. Add a valid Gemini API key to your .env file, then restart Streamlit.")
    st.stop()

agent = get_agent(google_api_key)

# Initialize session states to store conversation histories
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

# Render the chat history on the web page
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Accept user input from the web page
if user_query := st.chat_input("What would you like me to research today?"):
    # 1. Display user query and save it to history
    st.session_state.display_messages.append({"role": "user", "content": user_query})
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # 2. Run the agent and display the response
    with st.chat_message("assistant"):
        with st.spinner("Researching and structuring data..."):
            try:
                # Run the agent
                response = agent.invoke({"messages": st.session_state.chat_history})
                
                # Update internal history
                st.session_state.chat_history = response["messages"]
                final_output = st.session_state.chat_history[-1].content
                
                if isinstance(final_output, list):
                    final_output = "".join(block.get("text", "") for block in final_output if isinstance(block, dict))
                    
                structured_data = parser.parse(final_output)
                
                # Format the final output nicely for the web page
                formatted_response = (
                    f"**TOPIC:** {structured_data.topic}\n\n"
                    f"**SUMMARY:** {structured_data.summary}\n\n"
                    f"**SOURCES:** {', '.join(structured_data.sources)}\n\n"
                    f"**TOOLS USED:** {', '.join(structured_data.tools_used)}"
                )
                
                st.markdown(formatted_response)
                st.session_state.display_messages.append({"role": "assistant", "content": formatted_response})
                
            except Exception as e:
                if is_leaked_key_error(e):
                    st.error(
                        "Your Gemini API key has been blocked because Google detected it was leaked. "
                        "Create a new key, replace GOOGLE_API_KEY in your .env file, then restart Streamlit."
                    )
                else:
                    st.error(f"An error occurred: {e}")
