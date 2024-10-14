from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr


class TaliAPIEmbeddings(OpenAIEmbeddings):
    openai_api_base: str = "http://192.168.11.199:1284/v1"
    openai_api_key: SecretStr = SecretStr("123456")
    check_embedding_ctx_length: bool = False
    rerank_api_base: str = "http://192.168.11.199:1286/rerank"
    

class ZhipuEmbeddings(OpenAIEmbeddings):
    openai_api_base: str = "https://open.bigmodel.cn/api/paas/v4"
    openai_api_key: SecretStr = SecretStr("2b9d3bec1f45cf54cd02c2b5e2797b08.EjXke42OHxheEyJ4")
    model: str = "embedding-3"
    check_embedding_ctx_length: bool = False