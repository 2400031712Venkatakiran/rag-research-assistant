from langchain_core.embeddings import Embeddings

from app.config import Settings


def build_embeddings(settings: Settings) -> Embeddings:
    provider = settings.embedding_provider
    model = settings.embedding_model

    # Gemini is built directly: LangChain's init_embeddings only recognises
    # "google_genai" in some versions, so we don't depend on that.
    if provider == "google_genai":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=model)

    from langchain.embeddings import init_embeddings

    return init_embeddings(f"{provider}:{model}")