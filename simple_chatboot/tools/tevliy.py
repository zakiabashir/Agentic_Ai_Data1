import requests
import os
import json
from agents import FunctionTool
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
api_key = os.getenv("TAVILY_API_KEY")

def tavily_search_func(query: str):
    url = "https://api.tavily.com/search"
    print("web_search tool loaded successfully.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "max_results": 3
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        results = data.get("results", [])
        if not results:
            return "No results found."
        output = "Top Web Results:\n"
        for i, result in enumerate(results, 1):
            output += f"{i}. {result.get('title', 'No Title')}\n{result.get('url', '')}\n{result.get('snippet', '')}\n\n"
        return output
    else:
        return f"❌ Error: {response.text}"

async def on_search_invoke(tool_context, params):
    if isinstance(params, str):
        params = json.loads(params)
    result = tavily_search_func(params["query"])
    
    return result

web_search = FunctionTool(
    name="web_search",
    description="Search the web using Tavily API and return top results.",
    params_json_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    },
    on_invoke_tool=on_search_invoke
)