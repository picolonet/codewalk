import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

def get_api_key(variable_name: str) -> str:
    """
    Retrieves an API key from environment variables.

    Args:
        variable_name: The name of the environment variable.

    Returns:
        The API key string.

    Raises:
        ValueError: If the environment variable is not set.
    """
    api_key = os.getenv(variable_name)
    if not api_key:
        raise ValueError(f"Error: Environment variable '{variable_name}' is not set. Please set it in your environment or a .env file.")
    return api_key

# Retrieve API keys
GROQ_API_KEY = get_api_key("GROQ_API_KEY")
OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")
CLAUDE_API_KEY = get_api_key("CLAUDE_API_KEY")

