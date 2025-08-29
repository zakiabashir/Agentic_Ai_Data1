from agents import Agent
from myconfig.gemini_config import MODEL
from data_schema.myDataSchema import MyDataType

guardrial_agent = Agent(
    name="Guradrial Agent for Hotel Sannataquerries",
    instructions = """
You are a guardrail system. 
Check if the input is about a hotel or hotel services.
Return is_query_about_hotel=True if the query is about hotel information (rooms, booking, services).
Otherwise, return is_query_about_hotel=False.
""",
    model=MODEL,
    output_type=MyDataType

)