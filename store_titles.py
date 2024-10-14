import os
import asyncio
import chromadb
import time
import json
import uuid
import urllib3
from loguru import logger
from tqdm.asyncio import tqdm
from bs4 import BeautifulSoup
from typing import Dict, List, Any
from aiofiles import open as aio_open
from langchain_core.documents import Document
from elasticsearch import Elasticsearch, helpers
from langchain_elasticsearch import ElasticsearchStore
from tali_llm import ZhiPuLLM
from tali_embedding import ZhipuEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from chromadb.utils import embedding_functions

# 禁用 InsecureRequestWarning 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
model = ZhiPuLLM()

logger.add(
    "store_en_titles.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="DEBUG",
)

translate_prompt = ChatPromptTemplate.from_template("""
请将下面的标题翻译为中文，确保翻译准确无误，语言简练。只返回翻译后的中文标题，无需包含其他任何内容。
标题：{title}
""")


trans_chain = translate_prompt | model

embedding = embedding_functions.create_langchain_embedding(ZhipuEmbeddings())

client = chromadb.HttpClient(
    host="192.168.11.199",
    port=1283,
    settings=chromadb.Settings(
        chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
        chroma_client_auth_credentials="HpFyDnmmt8pRoLiekiPc3mBzm7bJJ6shCrqSh00V9zc",
    ),
)


def init_client():
    try:
        if client.get_collection("help_docs_toc"):
            client.delete_collection("help_docs_toc")
        print("Chroma init success")
    except Exception as e:
        print(str(e))


async def get_title(file_path: str) -> List[Dict[str, Any]]:
    xml_files = []
    result: Dict[str, str] = {}
    for root, dirs, files in os.walk(file_path):
        if "act_xml" in root:
            continue
        for file in files:
            if file.endswith(".toc"):
                file_path = os.path.join(root, file)
                xml_files.append(file_path)

    async def process_file(xml_file: str, result:dict, pbar) -> Dict[str, Any]:
        url = os.path.dirname(xml_file)
        base_url = url.replace("commonfiles", "http://192.168.11.199:9600")

        try:
            async with aio_open(xml_file, "r", encoding="utf-8") as file:
                xml_content = await file.read()
        except Exception as e:
            logger.error(f"Error reading file {xml_file}: {e}")
            return {}

        soup = BeautifulSoup(xml_content, "lxml-xml")
        # 提取根标题

        if soup.title is not None:
            root_title = soup.title
            root_href = root_title["href"]
            root_name = root_title.text
            url = os.path.join(base_url, root_href)

            result[url] = root_name

        # 提取章节信息
        def extract_chapters(parent_name, dl_element):
            for dt in dl_element.find_all("dt"):
                chapter = dt.find("a")
                if chapter is not None:
                    chapter_name = chapter.text
                    chapter_href = chapter["href"]
                    chapter_url = os.path.join(base_url, chapter_href)

                    # 拼接父标题和子标题
                    full_name = f"{parent_name} {chapter_name}"

                    # 添加到结果字典
                    result[chapter_url] = full_name

                    # 检查是否有子章节
                    dd = dt.find_next_sibling("dd")
                    if dd and dd.dl:
                        extract_chapters(full_name, dd.dl)

        if soup.toc is not None and soup.toc.dl is not None:
            extract_chapters(root_name, soup.toc.dl)

        pbar.update(1)

        return result

    logger.warning("开始抽取信息")
    with tqdm(
        total=len(xml_files),
        desc="extra info",
        colour="white",
        smoothing=0.1,
    ) as pbar:
        tasks = [process_file(xml_file, result, pbar) for xml_file in xml_files]
        xml_infos = await asyncio.gather(*tasks)

    with open("zh_title_list_new.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(xml_infos, ensure_ascii=False))

    return xml_infos



# 存入chroma
async def save_to_chroma(title_infos: List[Dict[str, str]]) -> None:
    logger.warning("开始向量化")

    collection = client.get_or_create_collection(
        name="help_docs_toc", embedding_function=embedding
    )

    # 使用生成器表达式来避免一次性加载所有数据到内存
    def generate_data(zh_title_info):
        for title, url in zh_title_info.items():
            if len(title) > 500:
                yield str(uuid.uuid4()), title[:500], {"source": url}
            else:
                yield str(uuid.uuid4()), title, {"source": url}

    # 批量处理数据
    batch_size = 5000
    batch_ids, batch_documents, batch_metadata = [], [], []
    total_titles = sum(len(title_info) for title_info in title_infos)

    with tqdm(
        total=total_titles, desc="adding to Chroma", unit="doc", colour="green"
    ) as pbar:
        for zh_title_info in title_infos:
            for id_, document, metadata in generate_data(zh_title_info):
                batch_ids.append(id_)
                batch_documents.append(document)
                batch_metadata.append(metadata)

                pbar.update(1)

                if len(batch_ids) >= batch_size:
                    collection.add(
                        ids=batch_ids,
                        documents=batch_documents,
                        metadatas=batch_metadata,
                    )
                    batch_ids, batch_documents, batch_metadata = [], [], []

        # 处理剩余的数据
        if batch_ids:
            collection.add(
                ids=batch_ids, documents=batch_documents, metadatas=batch_metadata
            )

    # 记录向量化的文档数量
    total_documents = collection.count()
    logger.debug(
        f"document last item: {batch_documents[-1]}, metadata: {batch_metadata[-1]}"
    )
    logger.info(f"Total documents added to Chroma: {total_documents}")
    logger.success("HTML文档向量化完成！")



def split_string_generator(s: str, target_count: int):
    start = 0
    while start < len(s):
        end = min(start + target_count, len(s))
        yield s[start:end]
        start += target_count

def save_to_es(zh_titles_list: List[Dict[str, str]]):
    es_connection = Elasticsearch(
        "https://121.36.41.167:19200",
        basic_auth=("elastic", "Tali#2024"),
        request_timeout=30,
        verify_certs=False,
        max_retries=10,
        retry_on_timeout=True
    )

    index_name = "help_docs_toc"

    es_store = ElasticsearchStore(
        es_connection=es_connection,
        index_name=index_name,
        embedding=ZhipuEmbeddings(),
    )

    toc_docs = []
    
    
    for info in zh_titles_list:
        for title, url in info.items():
            surl = url.replace("http://192.168.11.199:9600/", "")
            if len(title) < 510:
                toc_docs.append(
                    Document(page_content=title, metadata={"source": surl})
                )
            else:
                for string in split_string_generator(title, 510):
                    toc_docs.append(
                        Document(page_content=string, metadata={"source": surl})
                    )
    # 分批
    batch_size = 5000
    total_batches = (len(toc_docs) + batch_size - 1) // batch_size
    with tqdm(total=total_batches, desc="Insterting docs to es") as pbar:
        for i in range(0, len(toc_docs), batch_size):
            batch_docs = toc_docs[i : i + batch_size]
            # 执行批量操作
            try:
                uuids = [str(uuid.uuid4()) for _ in range(len(batch_docs))]
                success_doc = es_store.add_documents(documents=batch_docs, ids=uuids, embedding=embedding, index_name=index_name,
                                es_connection=es_connection)
                # success, failed = helpers.bulk(es_store, actions)
                print(f"Batch {i // batch_size}: 成功插入{len(success_doc)}条记录"
                )
            except helpers.BulkIndexError as e:
                print(f"Batch {i // batch_size}: Bulk indexing error: {e}")
            except Exception as e:
                print(f"Batch {i // batch_size}: An unexpected error occurred: {e}")

            # 更新进度条
            pbar.update(1)
            
            # 防止内存过载
            time.sleep(0.5)  

    es_connection.close()
    print("所有文档已成功插入到Elasticsearch中。")

async def main(file_path: str) -> None:
    stime = time.time()
    with open("zh_title_list.json", "r", encoding="utf-8") as f:
        zh_titles_list = json.loads(f.read())
    # zh_titles_list = await get_title(file_path)
    # await save_to_chroma(zh_titles_list)

    save_to_es(zh_titles_list)
    elapsed_time = time.time() - stime
    print(f"任务结束，耗时：{elapsed_time:.2f} 秒")


if __name__ == "__main__":
    # init_client()
    asyncio.run(main("commonfiles/help/zh-cn/"))
