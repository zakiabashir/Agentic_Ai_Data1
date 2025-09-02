from dotenv import find_dotenv, load_dotenv
import os
from agents import  AsyncOpenAI , OpenAIChatCompletionsModel

load_dotenv(find_dotenv(), override=True)

api_key = os.getenv("GEMINI_API_KEY")
base_url = os.getenv("GEMINI_BASE_URL")
Model = os.getenv("GEMINI_MODEL_NAME")


client = AsyncOpenAI(api_key=api_key, base_url=base_url)
model = OpenAIChatCompletionsModel(openai_client=client, model=Model)