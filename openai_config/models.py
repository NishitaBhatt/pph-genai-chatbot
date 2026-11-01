"""
OpenAI model getters for LangChain integration
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAIEmbeddings
from openai_config.config import OPENAI_API_TYPE, GPT_MODEL, EMBEDDING_MODEL


def get_embedding_model(model_name: str = None):
    """
    Get embedding model (OpenAI or Azure)

    Args:
        model_name: Model name/deployment (optional, uses config default)

    Returns:
        LangChain embedding model
    """
    model_name = model_name or EMBEDDING_MODEL

    if OPENAI_API_TYPE == "openai":
        # Standard OpenAI
        embedding_model = OpenAIEmbeddings(
            model=model_name,
        )

    elif OPENAI_API_TYPE in ["azure", "azure_ad"]:
        # Azure OpenAI
        import azure.identity

        embedding_model = AzureOpenAIEmbeddings(
            deployment=model_name,
            azure_ad_token_provider=azure.identity.get_bearer_token_provider(
                azure.identity.DefaultAzureCredential(
                    exclude_managed_identity_credential=True
                ),
                "https://cognitiveservices.azure.com/.default",
            ),
        )

    else:
        raise ValueError(f"Unknown OPENAI_API_TYPE: {OPENAI_API_TYPE}")

    return embedding_model


def get_gpt_model(model_name: str = None, temperature: float = 0.1):
    """
    Get GPT model (OpenAI or Azure)

    Args:
        model_name: Model name/deployment (optional, uses config default)
        temperature: Sampling temperature (0-1)

    Returns:
        LangChain chat model
    """
    model_name = model_name or GPT_MODEL

    if OPENAI_API_TYPE == "openai":
        # Standard OpenAI
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
        )

    elif OPENAI_API_TYPE in ["azure", "azure_ad"]:
        # Azure OpenAI
        llm = AzureChatOpenAI(
            azure_deployment=model_name,
            temperature=temperature,
        )

    else:
        raise ValueError(f"Unknown OPENAI_API_TYPE: {OPENAI_API_TYPE}")

    return llm
