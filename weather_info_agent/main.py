
from agents import  Runner, set_tracing_disabled
from Agent.myagent import weather_agent
set_tracing_disabled(True) # Disable tracing for better performance

while True:
    user_input = input("write querry here: ")
    if user_input.lower() == "exit":
        break
    response = Runner.run_sync(weather_agent, user_input)
    print("\n********\n" , response.final_output,"\n********\n" , "(type 'exit' to quit): ")