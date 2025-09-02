from agents import Agent 
from myconfig.gemini_config import Model
general_agent = Agent(
    "GeneralAgent",
    instructions="You are a helpful assistant for general queries.",
    model=Model,
)