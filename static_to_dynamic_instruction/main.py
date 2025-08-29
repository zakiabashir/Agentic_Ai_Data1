from agents import Runner, set_tracing_disabled, InputGuardrailTripwireTriggered
from my_agent.hotel_assistant import hotel_assistant

set_tracing_disabled(True)

try:
    while True:
        user_query = input("Enter your query (type 'exit' to quit): ")
        if user_query.strip().lower() == "exit":
            print("Exiting...")
            break

        res = Runner.run_sync(
            starting_agent=hotel_assistant, 
            input=user_query,
        )

        print(res.final_output)
except InputGuardrailTripwireTriggered as e:
    print(e)