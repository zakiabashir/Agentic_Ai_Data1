from agents import Agent
from myconfig.gemini_config import model
agent = Agent(
    name="math tutor",
    instructions="You provide help with math problems, explain your reasoning at each step and include examples.",
    model=model
)
