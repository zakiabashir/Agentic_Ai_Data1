from agents import Agent 
from myconfig.gemini_config import Model
math_agent = Agent(
    "MathAgent",
    instructions="You are a math solving assistant. Only answer math questions.",
    model=Model,
)
