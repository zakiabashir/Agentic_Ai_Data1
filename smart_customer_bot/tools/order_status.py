def get_order_status(order_id):
    # Simulated order status data
    order_status_data = {
        "123": "Your order is being processed.",
        "456": "Your order has been shipped.",
        "789": "Your order has been delivered."
    }
    
    if order_id in order_status_data:
        return order_status_data[order_id]
    else:
        raise ValueError("Order ID not found.")

@function_tool(is_enabled=lambda query: "order" in query.lower(), error_function=lambda e: "Sorry, I couldn't find that order. Please check the order ID and try again.")
def fetch_order_status(order_id):
    try:
        return get_order_status(order_id)
    except ValueError as e:
        return error_function(e)