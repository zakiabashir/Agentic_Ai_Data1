from decouple import config
from agents import  AsyncOpenAI , OpenAIChatCompletionsModel

key=config("GEMINI_API_KEY")
base_url=config("GEMINI_BASE_URL")
model_name=config("GEMINI_MODEL_NAME")

#this is a workaround to set client with base url
client= AsyncOpenAI(api_key=key, base_url=base_url) # Initialize the client with the base URL and api key
# Initialize the model with the client
model=OpenAIChatCompletionsModel(model=model_name, openai_client=client)
