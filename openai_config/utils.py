"""
OpenAI utility functions for configuration and authentication
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env.local or .env
env_path = Path(__file__).parent.parent / ".env.local"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()  # Try .env

from openai_config.config import (
    OPENAI_API_TYPE,
    OPENAI_API_KEY as CONFIG_API_KEY,
    GPT_MODEL,
    EMBEDDING_MODEL,
)

# Use environment variable if set, otherwise use config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", CONFIG_API_KEY)


def config_env():
    """
    Configure environment variables for OpenAI
    """
    os.environ["OPENAI_API_TYPE"] = OPENAI_API_TYPE

    # For Azure OpenAI
    if OPENAI_API_TYPE in ["azure", "azure_ad"]:
        from openai_config.config import AZURE_OPENAI_ENDPOINT, OPENAI_API_VERSION

        os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
        os.environ["OPENAI_API_VERSION"] = OPENAI_API_VERSION


def set_token():
    """
    Set OpenAI API token

    For standard OpenAI: Uses API key from config
    For Azure OpenAI: Uses Azure AD authentication
    """
    if OPENAI_API_TYPE == "openai":
        # Standard OpenAI - use API key directly
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

        # Validate key
        if OPENAI_API_KEY.startswith("sk-proj-REPLACE") or len(OPENAI_API_KEY) < 20:
            raise ValueError(
                "\n❌ ERROR: OpenAI API key not configured!\n"
                "\n📝 Please edit: openai_config/config.py"
                "\n   Replace OPENAI_API_KEY with your actual API key"
                "\n"
                "\n🔑 Get your key from: https://platform.openai.com/api-keys"
            )

    elif OPENAI_API_TYPE in ["azure", "azure_ad"]:
        # Azure OpenAI - use Azure AD authentication
        try:
            import azure.identity

            azure_credential = azure.identity.DefaultAzureCredential(
                exclude_managed_identity_credential=True
            )

            # Get token for Azure Cognitive Services
            token = azure_credential.get_token(
                "https://cognitiveservices.azure.com/.default"
            ).token
            os.environ["OPENAI_API_KEY"] = token

        except ImportError:
            raise ImportError(
                "Azure Identity not installed. Install with: pip install azure-identity"
            )

    else:
        raise ValueError(f"Unknown OPENAI_API_TYPE: {OPENAI_API_TYPE}")


def get_model_names():
    """Return configured model names"""
    return {
        "gpt_model": GPT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    }
