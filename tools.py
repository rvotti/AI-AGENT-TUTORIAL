from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_core.tools import Tool
from datetime import datetime

# --- 1. Web Search Tool (DuckDuckGo) ---
search_tool = DuckDuckGoSearchRun(
    name="search_web",
    description="Useful for searching the web for current information and broad topics."
)

# --- 2. Wikipedia Tool ---
# We wrap the API to limit the results for the demo
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# --- 3. Custom Save to File Tool ---
def save_to_txt(data: str, filename: str = "research_output.txt"):
    """Saves the provided research data string to a text file with a timestamp."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"RESEARCH OUTPUT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 50 + "\n")
        f.write(data)
    return f"Successfully saved to {filename}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Use this tool to save your final research summary and structured data to a text file."
)