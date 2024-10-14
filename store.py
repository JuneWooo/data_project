import asyncio
import json
import os
import re
import time
import uuid
import urllib3
from typing import Any, Dict, List, Union

import chromadb
from bs4 import BeautifulSoup
from chromadb.utils import embedding_functions
from elasticsearch import Elasticsearch, helpers
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from tqdm.asyncio import tqdm

from tali_embedding import ZhipuEmbeddings
from tali_llm import ZhiPuLLM

# 禁用 InsecureRequestWarning 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

model = ZhiPuLLM()

logger.add(
    "store_generate.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="DEBUG",
)

translate_prompt = ChatPromptTemplate.from_template(
    """请将问题：{quesion}翻译成中文，最后只返回翻译后的信息，不需要添加其他说明信息。"""
)

translate_chain = translate_prompt | model

generate_question_prompt = ChatPromptTemplate.from_template("""使用文本：{context} 生成至多3个关于此上下文的可能问题。确保这些问题可以直接从上下文中得到答案，并且不包括任何答案或标题。用换行符将问题分隔开。
""")

question_chain = generate_question_prompt | model

summary_prompt = ChatPromptTemplate.from_template("""你的工作是编写摘要，使用中文对以下文本：{text} 编写简洁摘要（50字以内），最后只返回摘要，不需要添加其他说明信息。""")

summary_chain = summary_prompt | model

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
        if client.get_collection("help_docs_title_summary_question"):
            client.delete_collection("help_docs_title_summary_question")
        print("Chroma init success")
    except Exception as e:
        print(str(e))


# 获取4万个html页面的标题和内容
async def get_title(file_path: str) -> List[Dict[str, Any]]:
    html_files = []
    html_info_list = []

    for root, dirs, files in os.walk(file_path):
        if "act_xml" in root:
            continue
        for file in files:
            if file.endswith(".html"):
                file_path = os.path.join(root, file)
                html_files.append(file_path)

    async def process_file(html_file: str, pbar) -> Dict[str, Any]:
        url = html_file.replace("commonfiles", "http://192.168.11.199:9600")
        titles: List[str] = []
        content = ""
        try:
            with open(html_file, "r", encoding="utf-8") as file:
                html_content = file.read()
        except Exception as e:
            print(f"Error reading file {html_file}: {e}")
            logger.debug(f"读取文件：{url}，发生异常")

            return {
                "url": url,
                "title": titles,
                "content": content,
            }

        soup = BeautifulSoup(html_content, "html.parser")
        for title in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            title_text = title.get_text(strip=True).replace("\n", "").replace(" ", "")
            titles.append(title_text)

        divs = soup.find_all("div", class_="small-12 cell")
        for div in divs:
            paragraphs = div.find_all(["p", "span"], recursive=True)
            for paragraph in paragraphs:
                content += (
                    paragraph.get_text(strip=True).replace("\n", "").replace(" ", "")
                )
        # 返回空对象有错误, 记录
        if len(titles) == 0:
            logger.debug(f"url：{url}, page_content: {content}")
        if len(content) == 0:
            logger.debug(f"url：{url}, page_content: {content}")

        pbar.update(1)

        return {
            "url": url,
            "title": titles,
            "content": content,
        }

    logger.warning("开始抽取信息")

    # 创建一个异步进度条
    with tqdm(
        total=len(html_files),
        desc="抽取信息进度",
        colour="white",
        smoothing=0.1,
    ) as pbar:
        tasks = [process_file(html_file, pbar) for html_file in html_files]
        results = await asyncio.gather(*tasks)

    # 过滤掉所有info字典中，title列表中有重复汉字的元素
    for info in results:
        if len(info["title"]) > 0 and info["content"]:
            html_info_list.append(info)
            logger.debug(f"url：{info['url']}, page_content: {info['content']}")

    with open("html_info_list.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(html_info_list, ensure_ascii=False))

    logger.success("抽取信息完成！")

    return html_info_list


# 根据页面内容生成3个问题，并且组装为新的字典：
async def generate_question(
    html_info_list: List[Dict[str, Any]],
) -> None:
    # html_questions: Dict[str, list] = {}
    # title_quesion_list = []

    url = [info["url"] for info in html_info_list]

    titles = [info["title"] for info in html_info_list]
    
    # 获取所有内容
    page_content = [info["content"] for info in html_info_list]

    # 限制并发任务数量
    semaphore = asyncio.Semaphore(10)

    async def generate(content, pbar):
        async with semaphore:
            try:
                result = await question_chain.ainvoke(content)
                questions = result.content
                question_str = re.sub(r"\d+\.*\s", "", questions.strip())

                if len(content) >= 20:
                    summary_result = await summary_chain.ainvoke({"text": content})
                    summary_str = summary_result.content
                else:
                    summary_str = content

            except Exception as e:
                logger.error(
                    f"Error generating question for content: {content}. Error: {e}"
                )
                return f"Error: 调用模型生成问题失败：{e}"
            else:
                pbar.update(1)
                return question_str, summary_str

    logger.warning("开始生成问题")
    # # 创建一个异步进度条
    with tqdm(
        total=len(page_content),
        desc="summary&generation progress",
        unit="qa",
        colour="green",
        smoothing=0.1,
    ) as pbar:
        tasks = [generate(content, pbar) for content in page_content]
        question_list = await asyncio.gather(*tasks, return_exceptions=True)

    # 并发执行所有任务
    if len(question_list) != len(page_content):
        logger.error("问题数量与文本数量不一致，请检查!")

    html_titles = []
    html_questions = []
    html_summaries = []
    # 将内容和问题对应起来
    for html_url, title_str, question_summary in zip(url, titles, question_list):
        if isinstance(question_summary, Exception):
            logger.error(f"失败文件：{html_url} 问题生成失败，请检查！")
            continue
        if isinstance(question_summary, tuple):
            if "Error:" in question_summary[0]:
                logger.error(f"失败文件：{html_url} 问题生成失败，请检查！")
                continue
            questions = [q.strip() for q in question_summary[0].split("\n") if q]
            summaries = question_summary[1].strip()

            html_titles.append({"url": html_url, "title": title_str})
            html_questions.append({"url": html_url, "questions": questions})
            html_summaries.append({"url": html_url, "summaries": [summaries]})

    logger.success(f"标题生成完成！{html_titles}")
    logger.success(f"问题生成完成！{html_questions}")
    logger.success(f"总结生成完成！{html_summaries}")
    # if not os.path.exists("title_quesion_list.json"):
    #     logger.warning("缓存文件不存在，开始将生成的问题写入文件中")
    #     with open("title_quesion_list.json", "w") as f:
    #         f.write(json.dumps(title_quesion_list, ensure_ascii=False, indent=2))

    with open("help_docs_html_titles.json", "w") as f:
        f.write(json.dumps(html_titles, ensure_ascii=False, indent=2))
    
    with open("help_docs_html_questions.json", "w") as f:
        f.write(json.dumps(html_questions, ensure_ascii=False, indent=2))

    with open("help_docs_html_summaries.json", "w") as f:
        f.write(json.dumps(html_summaries, ensure_ascii=False, indent=2))

    # return title_quesion_list


# 存入chroma
async def save_to_chroma(html_questions: List[Dict[str, Any]]) -> None:
    logger.warning("开始向量化")

    collection = client.get_or_create_collection(
        name="help_docs_title_summary_question", embedding_function=embedding
    )

    # 刷所有，标题+问题+内容总结
    def generate_data():
        for item in html_questions:
            for context in filter(None, item["context"]):
                if len(context) > 500:
                    yield str(uuid.uuid4()), context[:500], {"source": item["url"]}
                else:
                    yield str(uuid.uuid4()), context, {"source": item["url"]}

    # 批量处理数据
    batch_size = 5000
    batch_ids, batch_documents, batch_metadata = [], [], []
    totals = sum([len(item["context"]) for item in html_questions])
    # 使用 tqdm 创建进度条
    with tqdm(total=totals, desc="vector process", unit="doc", colour="yellow") as pbar:
        for id_, document, meta in generate_data():
            batch_ids.append(id_)
            batch_documents.append(document)
            batch_metadata.append(meta)

            if len(batch_ids) >= batch_size:
                pbar.update(5000)
                # await asyncio.sleep(0)
                collection.add(
                    ids=batch_ids,
                    documents=batch_documents,
                    metadatas=batch_metadata,
                )
                batch_ids, batch_documents, batch_metadata = [], [], []

    # 处理剩余的数据
    if batch_ids:
        # await asyncio.sleep(0)
        collection.add(
            ids=batch_ids, documents=batch_documents, metadatas=batch_metadata
        )
    total_documents = collection.count()
    logger.success(f"HTML 问题向量化完成！一共添加了 {total_documents} 条数据！")


def save_to_es(title_question_list: List[Dict[str, Union[str, List]]], index_name:str):
    # es_connection = Elasticsearch("https://121.36.41.167:19200",basic_auth=("elastic", "Tali#2024"), verify_certs=False) 
    es_connection = Elasticsearch(
        "https://121.36.41.167:19200",
        basic_auth=("elastic", "Tali#2024"),
        request_timeout=30,
        verify_certs=False,
        max_retries=10,
        retry_on_timeout=True
    )
    

    es_store = ElasticsearchStore(
        es_connection=es_connection,
        index_name=index_name,
        embedding=ZhipuEmbeddings(),
    )

    title_question_docs = []
    for info in title_question_list:
        url = info["url"].replace("http://192.168.11.199:9600/", "")
        for pc in filter(None, info["questions"]):
            if len(pc) < 510:
                title_question_docs.append(
                    Document(page_content=pc, metadata={"source": url})
                )
            else:
                for string in split_string_generator(pc, 510):
                    title_question_docs.append(
                        Document(page_content=string, metadata={"source": url})
                    )
    # 分批
    batch_size = 1000
    total_batches = (len(title_question_docs) + batch_size - 1) // batch_size
    with tqdm(total=total_batches, desc="Insterting docs to es") as pbar:
        for i in range(0, len(title_question_docs), batch_size):
            batch_docs = title_question_docs[i : i + batch_size]
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


def split_string_generator(s: str, target_count: int):
    start = 0
    while start < len(s):
        end = min(start + target_count, len(s))
        yield s[start:end]
        start += target_count


async def main(file_path: str) -> None:
    # init_client()
    stime = time.time()
    # with open("html_info_list.json", "r", encoding="utf-8") as f:
    #     html_info_list = json.loads(f.read())
    # html_info_list = await get_title(file_path)

    # with open("title_quesion_list.json", "r", encoding="utf-8") as f:
    #     html_questions = json.loads(f.read())

    # html_questions = await generate_question(html_info_list)
    # await save_to_chroma(html_questions)

    with open("help_docs_html_questions.json", "r", encoding="utf-8") as f:
        help_docs_html_questions = json.loads(f.read())

    save_to_es(help_docs_html_questions, index_name="help_docs_html_questions")

    elapsed_time = time.time() - stime
    print(f"任务结束，耗时：{elapsed_time:.2f} 秒")


if __name__ == "__main__":
    asyncio.run(main("commonfiles/"))
