"""
OpenAI Configuration
====================

🔧 USER CONFIGURATION - EDIT THIS FILE
---------------------------------------

Replace the placeholder values below with your own credentials.

For OpenAI (default):
    - Set your OPENAI_API_KEY
    - Keep OPENAI_API_TYPE = "openai"

For Azure OpenAI (alternative):
    - Set OPENAI_API_TYPE = "azure_ad" or "azure"
    - Set AZURE_OPENAI_ENDPOINT to your Azure endpoint
    - Set OPENAI_API_VERSION
    - Configure Azure credentials
"""

# ============================================================================
# CONFIGURATION SECTION - EDIT THESE VALUES
# ============================================================================

# Standard OpenAI Configuration
# ------------------------------
OPENAI_API_TYPE = "openai"  # Changed from "azure_ad" to "openai"
OPENAI_API_KEY = ""  # Will be loaded from environment variable OPENAI_API_KEY

# Model names (OpenAI public API)
GPT_MODEL = "gpt-4o-mini"  # or "gpt-4o", "gpt-4-turbo"
EMBEDDING_MODEL = "text-embedding-3-large"  # or "text-embedding-3-small"

# Azure OpenAI Configuration (alternative - currently disabled)
# -------------------------------------------------------------
# OPENAI_API_TYPE = "azure_ad"
# AZURE_OPENAI_ENDPOINT = "https://YOUR-AZURE-ENDPOINT.openai.azure.com/"
# OPENAI_API_VERSION = "2024-10-21"
# AZURE_GPT_DEPLOYMENT = "gpt-4o"
# AZURE_EMBEDDING_DEPLOYMENT = "text-embedding-3-large"

# ============================================================================
# END OF USER CONFIGURATION
# ============================================================================
