def guardrail(func):
    def wrapper(user_input: str):
        return func(user_input)
    return wrapper

@guardrail
def sentiment_guardrail(user_input: str) -> str:
    negative_keywords = ["bad", "terrible", "hate", "worst", "offensive"]
    if any(keyword in user_input.lower() for keyword in negative_keywords):
        return "I'm sorry to hear that you're feeling this way. How about we rephrase that?"
    return user_input