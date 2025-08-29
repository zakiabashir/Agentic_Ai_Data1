class BotAgent:
    def __init__(self, logger):
        self.logger = logger
        self.faqs = {
            "What is your return policy?": "You can return any item within 30 days of purchase.",
            "How can I track my order?": "You can track your order using the order status tool.",
            "What payment methods do you accept?": "We accept credit cards, PayPal, and bank transfers."
        }
        self.greetings = ["hi", "hello", "hey", "good morning", "good evening"]

    def handle_query(self, query):
        if self.is_greeting(query):
            return "Hello! How can I assist you today?"
        if self.is_faq(query):
            return self.answer_faq(query)
        elif "order" in query:
            return self.get_order_status(query)
        else:
            return "I'm sorry, I can't assist with that. Let me connect you to a human agent."

    def is_greeting(self, query):
        return query.lower().strip() in self.greetings

    def is_faq(self, query):
        return query in self.faqs

    def answer_faq(self, query):
        return self.faqs[query]

    def get_order_status(self, query):
        # Extract order_id from the query (this is a placeholder for actual extraction logic)
        order_id = self.extract_order_id(query)
        if order_id:
            # Call the order status function tool here
            return f"Fetching status for order ID: {order_id}"
        else:
            return "Please provide a valid order ID."

    def extract_order_id(self, query):
        # Placeholder for order ID extraction logic
        return "12345"  # Simulated order ID for demonstration purposes

    def escalate_to_human(self):
        return "Escalating your request to a human agent."