import chromadb
from tali_embedding import TaliAPIEmbeddings
from chromadb.utils import embedding_functions

embedding = embedding_functions.create_langchain_embedding(TaliAPIEmbeddings())

client = chromadb.HttpClient(
    host="192.168.11.199",
    port=1283,
    settings=chromadb.Settings(
        chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
        chroma_client_auth_credentials="HpFyDnmmt8pRoLiekiPc3mBzm7bJJ6shCrqSh00V9zc",
    ),
)
# html_collection = client.get_or_create_collection("html_help_docs", embedding_function=embedding)


# def get_langchain_chroma(collection_name: Collection) -> Chroma:
#     return Chroma(
#         collection_name=collection_name.name,
#         client=client,
#         embedding_function=embedding,
#         relevance_score_fn=lambda distance: 1.0 - distance / 2,
#     )
