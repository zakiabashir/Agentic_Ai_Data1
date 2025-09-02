from myconfig.gemini_config import Model
from agents import Agent
from tools.weather import get_weather
from tools.tevliy import web_search

agent= Agent(
    name="MyAgent",
    instructions="You are a helpful assistant. ",
    model=Model,
    tools=[get_weather,web_search],
    
)