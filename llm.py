from pydantic import SecretStr
from langchain_openai import ChatOpenAI



class TaliLLM(ChatOpenAI):
    openai_api_base: str = "http://192.168.11.199:1282/v1"
    openai_api_key: SecretStr = SecretStr("123456")
    model_name: str = "gpt-4"
    temperature: float = 0.0



class ZhiPuLLM(ChatOpenAI):
    openai_api_base: str = "https://open.bigmodel.cn/api/paas/v4/"
    openai_api_key: SecretStr = SecretStr("16dc4fb1a9ffc719a4d4ec3d1a130678.bGNWa8mbq4bW0fgN")
    model_name: str = "GLM-4-Flash"
    temperature: float = 0.0

