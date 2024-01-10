import os
import weaviate

GOOGLE_MANUAL_URL = 'https://docs.google.com/spreadsheets/d/1FLLdDB9zvTgtwBSy1qW4wksI3Z-7_7t9mt9W0jAlA8U/export?format=csv'

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')

# Weaviate Configuration
WEAVIATE_URL = "https://manuals-roeiv8hv.weaviate.network"
# WEAVIATE_CLASS = your manual name here

# OpenAI Model Definitions
OPENAI_EMBEDDING_MODEL = 'text-embedding-ada-002'
OPENAI_COMPLETION_MODEL = 'gpt-4-1106-preview'#'gpt-3.5-turbo'

class WeaviateClient:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            auth_config = weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY)
            cls._instance = weaviate.Client(url=WEAVIATE_URL, 
                                            auth_client_secret=auth_config)
        return cls._instance

def get_client():
    return WeaviateClient.get_instance()
