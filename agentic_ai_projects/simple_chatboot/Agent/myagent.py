from myconfig.gemini_config import model
from agents import Agent
from tools.weather import get_weather
from tools.tevliy import web_search

agent= Agent(
    name="MyAgent",
    instructions="You are a helpful assistant. ",
    model=model,
    tools=[get_weather,web_search],
    
)