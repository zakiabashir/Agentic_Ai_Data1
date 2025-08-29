from myconfig.gemini_config import model
from agents import Agent
from tools.tevliy import web_search

web_search_agent= Agent(
    name="MyAgent",
    instructions="You are a helpful assistant. ",
    model=model,
    tools=[web_search],
    
)