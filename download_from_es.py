
import json
from elasticsearch import Elasticsearch, helpers
from langchain_elasticsearch import ElasticsearchStore
from tali_embedding import TaliAPIEmbeddings

es_connection = Elasticsearch(
        "http://192.168.11.199:9200",
        basic_auth=("elastic", "Tali2023"),
        verify_certs=False,
        request_timeout=60,
        max_retries=10,
        retry_on_timeout=True,
    )

index_name = "help_docs_html_summaries"

es_store = ElasticsearchStore(
	es_connection=es_connection,
	index_name=index_name,
	embedding=TaliAPIEmbeddings(),
)



# 定义一个函数来查询Elasticsearch并返回结果
def search_and_transform(es, index_name, query=None, size=100):
    # 如果没有提供query，则默认搜索所有文档
    if not query:
        query = {
            "size": size,
            "query": {
                "match_all": {}
            }
        }
    
    # 执行搜索请求
    response = es.search(index=index_name, body=query, scroll='1m')
    scroll_id = response['_scroll_id']
    hits = response['hits']['hits']
    
    transformed_data = []
    
    while len(hits) > 0:
        for hit in hits:
            source = hit['_source']
            transformed_item = {
                "url": source["metadata"]["source"],
                "summary": [source["text"]]
            }
            transformed_data.append(transformed_item)
        
        # 继续滚动获取更多数据
        response = es.scroll(scroll_id=scroll_id, scroll='1m')
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']
    
    return transformed_data

# 使用search_and_transform函数获取并转换数据
transformed_data = search_and_transform(es_connection, index_name)

# 将数据转换为JSON字符串
json_data = json.dumps(transformed_data, ensure_ascii=False, indent=4)

# 将JSON字符串写入文件
with open('html_summary.json', 'w', encoding='utf-8') as f:
    f.write(json_data)

print("数据已保存到'data.json'文件")