"""Maps RETRIEVAL_METHOD to a Retriever. New methods register here and nowhere else."""
from typing import Callable

from langchain_core.vectorstores import VectorStore

from app.config import Settings
from app.retrieval.base import Retriever
from app.retrieval.dense import DenseRetriever
Builder = Callable[[Settings, VectorStore], Retriever]
"""
It says the registry is a dictionary where:
- Each key is a str, such as "dense".
- Each value is a Builder function.
type defining at callable return type retirver and parameters 

lambda is a small way of creating func without using def*--*--*--*--*--*--*
Instead of if else we are using dict to get the particukar retriver in get_retriver method*--*--*--*--*--*--*
"""
_REGISTRY: dict[str, Builder] = {
    "dense": lambda settings, store: DenseRetriever(store),
}


def available_methods() -> list[str]:
    return sorted(_REGISTRY)


def get_retriever(settings: Settings, store: VectorStore) -> Retriever:
    try:
        builder = _REGISTRY[settings.retrieval_method]
    except KeyError:
        raise ValueError(
            f"Unknown RETRIEVAL_METHOD '{settings.retrieval_method}'. "
            f"Available: {available_methods()}"
        ) from None
    return builder(settings, store)