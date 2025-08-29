# Smart Customer Support Bot

This project implements a Smart Customer Support Bot using OpenAI's Agent SDK. The bot is designed to assist customers with common queries, fetch order statuses, and escalate to a human agent when necessary. It incorporates guardrails to ensure positive interactions and showcases advanced features of the SDK.

## Features

- **FAQ Handling**: The bot can answer basic product FAQs.
- **Order Status Lookup**: Fetch order statuses using a simulated API.
- **Human Agent Escalation**: Automatically escalate complex queries or those with negative sentiment to a human agent.
- **Guardrails**: Implemented to block or rephrase negative or offensive user input.
- **Advanced Functionality**: Utilizes model settings and advanced function tools for enhanced performance.

## Project Structure

```
smart-support-bot
├── src
│   ├── agents
│   │   ├── bot_agent.py       # Defines the BotAgent class for handling FAQs and order lookups.
│   │   └── human_agent.py     # Defines the HumanAgent class for handling escalated queries.
│   ├── tools
│   │   └── order_status.py     # Implements the get_order_status function with function tools.
│   ├── guardrails
│   │   └── sentiment_guardrail.py # Implements sentiment guardrails for user input.
│   ├── logging
│   │   └── logger.py           # Contains logging functionality for tool invocations and handoffs.
│   ├── main.py                 # Entry point for the application.
│   └── config.py               # Configuration settings for the application.
├── requirements.txt             # Lists dependencies required for the project.
├── README.md                    # Documentation for the project.
└── .env                         # Stores environment variables needed for the application.
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd smart-support-bot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables in the `.env` file. Ensure you include any necessary API keys and configuration settings.

4. Run the application:
   ```
   python src/main.py
   ```

## Usage Examples

- Ask the bot a product-related question to receive an FAQ response.
- Provide an order ID to check the status of your order.
- If the bot cannot handle your query or detects negative sentiment, it will escalate the issue to a human agent.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.