from agents import Agent
from myconfig.gemini_config import MODEL
from guardrial_function.guardrial_input_function import guardrial_input_function

# Store hotel details in a dictionary
HOTELS = {
    "Hotel Sannata": {
        "total_rooms": 200,
        "special_rooms": 20,
        "owner": "Mr. Ratan Lal",
        "city": "Karachi"
    },
    "Hotel Bliss": {
        "total_rooms": 150,
        "special_rooms": 10,
        "owner": "Ms. Anita Sharma",
        "city": "Lahore"
    },
    "Hotel Paradise": {
        "total_rooms": 100,
        "special_rooms": 5,
        "owner": "Mr. Ali Khan",
        "city": "Islamabad"
    }
}

def get_hotel_instructions(context, agent):
    query_text = getattr(context, "query_text", None)
    hotel_name = getattr(context, "hotel_name", None)

    # ✅ Special case: user asked for hotel list
    if query_text and ("list" in query_text.lower() or "names" in query_text.lower()):
        return "Available hotels are: " + ", ".join(HOTELS.keys())

    hotel = HOTELS.get(hotel_name)
    if hotel:
        return (
            f"You are a helpful hotel customer care assistant for {hotel_name}.\n"
            f"- Hotel name is {hotel_name}.\n"
            f"- Hotel Owner name is {hotel['owner']}.\n"
            f"- Hotel total room {hotel['total_rooms']}.\n"
            f"- {hotel['special_rooms']} rooms not available for public, It's for special guests."
        )
    else:
        return "Hotel information not found. Please specify a valid hotel name."


hotel_assistant = Agent(
    name="Hotel Customer Care",
    instructions=get_hotel_instructions,  # Pass function for dynamic instructions
    model=MODEL,
    input_guardrails=[guardrial_input_function],
    output_guardrails=[]
)
