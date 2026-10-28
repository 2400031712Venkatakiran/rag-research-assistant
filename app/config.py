from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Export .env values (e.g. GOOGLE_API_KEY) into os.environ so provider
# packages can read them. Real environment variables take priority.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Retrieval
    retrieval_method: str = "dense"
    top_k: int = Field(default=4, ge=1, le=50)

    # Storage / ingestion
    chroma_dir: Path = Path("./data/chroma")
    collection_name: str = "documents"
    documents_dir: Path = Path("./documents")
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)

    # Models
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
