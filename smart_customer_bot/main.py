from agents.bot_agent import BotAgent
from agents.human_agent import HumanAgent
from guardrails.sentiment_guardrail import sentiment_guardrail
from log_utils.logger import Logger

def main():
    # Initialize logging
    logger = Logger()

    # Initialize agents
    bot_agent = BotAgent(logger)
    human_agent = HumanAgent(logger)

    print("Welcome to the Smart Customer Support Bot!")
    
    while True:
        user_input = input("You: ")

        # Check for exit condition
        if user_input.lower() in ["exit", "quit"]:
            print("Thank you for using the Smart Customer Support Bot. Goodbye!")
            break

        # Apply sentiment guardrail
        guardrail_result = sentiment_guardrail(user_input)
        if guardrail_result != user_input:
            print(f"Bot: {guardrail_result}")
            continue

        # Process query with bot_agent
        response = bot_agent.handle_query(user_input)
        if response is None:
            print(human_agent.handle_escalation(user_input))
        else:
            print(f"Bot: {response}")

if __name__ == "__main__":
    main()