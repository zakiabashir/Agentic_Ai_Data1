# Configuration settings for the Smart Customer Support Bot

class Config:
    # OpenAI API settings
    api_key = "GEMINI_API_KEY"
    base_url ="GEMINI_BASE_URL"
    model_name = "GEMINI_MODEL_NAME"
    
    # Tool settings
    TOOL_CHOICE = "auto"  # Options: "auto", "required"
    
    # Logging settings
    LOGGING_LEVEL = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    
    # Guardrail settings
    GUARDRAIL_ENABLED = True  # Enable or disable guardrails
    
    # Other configurations
    MAX_TOKENS = 150  # Maximum tokens for responses
    TEMPERATURE = 0.7  # Controls randomness in responses

    @staticmethod
    def get_env_variable(var_name):
        import os
        return os.getenv(var_name) or None