import os
import openai
from dotenv import load_dotenv

# Load environment variables from.env file
load_dotenv()

def get_client():
    return openai.OpenAI()

# configure for OpenAi API usage
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
COMPLETION_MODEL = os.getenv("OPENAI_COMPLETION_MODEL")

# Configure for Neon database API
DATABASE_URL = os.getenv("NEON_DATABASE_URL")

