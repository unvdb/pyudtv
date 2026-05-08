# 九有数据库pymilvus使用
## 简介
pyudtv 基于 unvdb vector 向量检索开发， client 接口向下兼容pymilvus接口，确保现有代码几乎零修改。

## install
```sh
# 如果原先有安装则先卸载pymilvus， 如果没有安装则跳过
python -m pip uninstall pymilvus

# 通过打包好的wheel直接安装九有的安装包
python -m pip install pyudtv-26.0.0-py3-none-any.whl

# 确保安装TV或者ALL版本的九有数据库， 再安装 vector 数据库扩展插件
# 25 版本
CREATE EXTENSION IF NOT EXISTS vector;
# 24 版本
CREATE EXTENSION IF NOT EXISTS ud_vector;
```

## pymilvus与pyudtv差异对比
1、 创建客户端连接方式差异。 这个差异也是代理不同后端的依据。
```
# 九有Client连接方式
from pymilvus import MilvusClient
client = MilvusClient(f"unvdb://{user_name}:{password}@{host}:{port}/{db_name}")

```
2、目前UDBClient包不支持connections方式创建连接

3、 索引类型对比

| 特性                | UDBClient | Milvus |
|-------------------|--------------|--------|
| 暴力搜索              | - | FLAT |
| 倒排文件索引            | IVF_FLAT | IVF_FLAT |
| 量化索引              | - | IVF_SQ8 |
| 乘积量化索引            | - | IVF_PQ |
| 分层可导航小世界图索引       | HNSW | HNSW |
| 磁盘索引（超大规模索引）      | - | DISKANN |

4、 距离计算类型对比

| 特性 | UDBClient | Milvus |
|------|--------------|--------|
| 欧氏距离 | L2 | L2 |
| 内积 | IP | IP |
| 余弦相似度 | COSINE | COSINE |
| 汉明距离（二进制向量） | - | HAMMING |
| 杰卡德距离（集合） | - | JACCARD |
| 谷本距离（集合） | - | TANIMOTO |
| 子结构距离（分子指纹） | - | SUBSTRUCTURE |
| 超结构距离（分子指纹） | - | SUPERSTRUCTURE |

5、 filter 查询
语法参考： https://milvus.io/docs/zh/basic-operators.md
向下兼容基本操作符， 不支持json数据过滤操作。 支持 unvdb 函数调用， 支持加减乘除计算。

### PyUDTV Filter 与 PyMilvus Filter 差异对比

| 特性维度             | 具体功能                                     | PyMilvus Filter | PyUDTV Filter     | 备注                                            |
|:-----------------|:-----------------------------------------|:----------------|:------------------|:----------------------------------------------|
| **基础比较**         | `==`, `!=`, `>`, `<`, `>=`, `<=`         | ✅ 支持            | ✅ 支持              | 两者完全兼容                                        |
| **逻辑运算**         | `and`, `or`, `not`                       | ✅ 支持            | ✅ 支持              | 逻辑组合语法一致                                      |
| **数学运算**         | `+`, `-`, `*`, `/`                       | ⚠️ 有限支持          | ✅ 支持              |                               |
| **高级运算**         | `**` (乘方), `%` (取模)                      | ⚠️ 有限支持          | ✅ 支持              | PyUDTV 表达式能力更强                                |
| **字符串匹配**        | `like`, `not like`                       | ✅ 支持            | ✅ 支持              | 通配符 `%` 用法一致                                  |
| **集合操作**         | `in`, `not in`                           | ✅ 支持            | ✅ 支持              | 列表筛选                                          |
| **JSON/ARRAY过滤** | filter = 'product["price"] > 1000'       | ⚠️ 有限支持         | ❌ 不支持             | **PyUDTV 暂时不支持 JSON/ARRAY  字段解析，后续优化**        |
| **函数调用**         | `sqrt()`, `abs()`, `round()` 等           | ❌ 不支持           | ✅ **支持 unvdb 函数** | PyUDTV 核心优势，可调用数据库函数。支持函数嵌套（version >= 3.1.2）。 |



## 快速入门
```python
from pymilvus import MilvusClient # UDBClient的异名类
import numpy as np
VECTOR_FIELD = 'vector'
user_name = "unvdb"
password = "unvdb"
host = "localhost"
port = 5678
db_name = "test"
# unvdb://是实例化UNVDB后端标志
client = MilvusClient(f"unvdb://{user_name}:{password}@{host}:{port}/{db_name}")
# client._wrapped_class 查看原始的类 MilvusClient或者UDBClientBase
print(f"{client._wrapped_class} 连接成功！")

# 创建表, 极简模式，默认创建vector字段为向量字段。 创建成功之后，可以在数据库找到test这张表
COLLECTION_NAME = 'test'
DIM = 6
client.create_collection(
            collection_name=COLLECTION_NAME, # 表名
            dimension=DIM
        )
print("集合创建成功！")

# 生成测试数据
num_entities = 100
rng = np.random.default_rng(seed=58)
entities = []
for i in range(num_entities):
    ent_ = {
        "id": i + 1,
        VECTOR_FIELD: rng.random((1, DIM))[0].tolist(),
        "tag": f"tag_{i % 10}"
    }
    entities.append(ent_)
# 插入数据
insert_result = client.insert(
        collection_name=COLLECTION_NAME,
        data=entities,
        progress_bar=True
    )

print("插入数据成功")

# 加载表， UNVDB 后端可选
client.load_collection(COLLECTION_NAME)

# 查找数据
query_vector = rng.random((1, DIM))[0].tolist()
search_results = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    limit=5,
    output_fields=["id", "tag", "vector"],
    anns_field=VECTOR_FIELD,
    search_params={"nprobe": 10} # 查找聚类中心数
)
print(type(search_results), type(search_results[0]))
print("搜索结果:")
for i, hits in enumerate(search_results):
    print(f"查询 {i+1} 的结果:")
    for hit in hits:
        print(f"  ID: {hit['id']}, 距离: {hit['distance']:.4f}, 标签: {hit.get('tag')}, 值: {hit.get('vector')}")

# 删除测试表
client.drop_collection(COLLECTION_NAME)
print(f"删除测试表：{COLLECTION_NAME}")

```

## 字符串处理
```txt
支持单引号与双引号字符串： 'abc', "abc", "print('hello world')"
单引号内的单引号需要两个单引号转义： 'It''s'
双引号内的双引号需要使用反斜杠转义： "print(\"hello world\")"
```
## 字段名称与关键字冲突处理
```txt
方法1：如果表字段使用了  desc、order、where 等数据库保留字段时， 需要使用 `` 反引号来查询， 例如： `desc` == "一个蓝色的水杯"

方法2： 在schema表中定义字段
schema.add_field(field_name="desc", datatype=DataType.VARCHAR, max_length=200, nullable=True)
此时 desc == "一个蓝色的水杯"  是可以查询的

```

## 接口列表

DBClient类实现了以下接口：

| 序号 | 方法名 | 说明                        |
|------|--------|---------------------------|
| 1 | `create_schema` | 创建 CollectionSchema 对象    |
| 2 | `create_struct_field_schema` | 创建结构字段 schema             |
| 3 | `create_field_schema` | 创建字段 schema               |
| 4 | `create_collection` | 创建集合（表）                   |
| 5 | `insert` | 插入数据                      |
| 6 | `upsert` | 插入或更新数据                   |
| 7 | `search` | 向量搜索                      |
| 8 | `query` | 标量字段查询                    |
| 9 | `delete` | 删除数据                      |
| 10 | `update` | 更新数据(UDBClient特有)         |
| 11 | `has_collection` | 检查集合是否存在                  |
| 12 | `list_collections` | 列出所有集合                    |
| 13 | `drop_index` | 删除索引                      |
| 14 | `hybrid_search` | 混合搜索（多向量融合）               |
| 15 | `drop_collection` | 删除集合                      |
| 16 | `describe_collection` | 获取集合信息                    |
| 17 | `close` | 关闭数据库连接                   |
| 18 | `list_indexes` | 列出集合的所有索引                 |
| 19 | `create_index` | 创建索引                      |
| 20 | `load_collection` | 加载集合到内存（UDBClient仅加载字段信息） |
| 21 | `release_collection` | 释放集合内存（UDBClient可选）       |
| 22 | `describe_index` | 获取索引详细信息                  |

## 接口详细说明

### 1. create_schema

```python
def create_schema(cls, **kwargs) -> CollectionSchema
```

创建 CollectionSchema 对象，用于定义集合的结构。

**参数：**
- `**kwargs`: 可选参数，包括 auto_id, description, enable_dynamic_field 等

**返回：**
- CollectionSchema 对象

---

### 2. create_collection

```python
def create_collection(
    self,
    collection_name: str,
    dimension: Optional[int] = None,
    primary_field_name: str = "id",
    id_type: str = "int",
    vector_field_name: str = "vector",
    metric_type: str = "COSINE",
    auto_id: bool = True,
    timeout: Optional[float] = None,
    schema: Optional[CollectionSchema] = None,
    **kwargs,
)
```

创建集合（UNVDB 表）。

**参数：**
- `collection_name`: 集合名称
- `dimension`: 向量维度（极简模式使用）
- `primary_field_name`: 主键字段名
- `id_type`: 主键类型（int, str）
- `vector_field_name`: 向量字段名
- `metric_type`: 度量类型（L2, IP, COSINE）
- `auto_id`: 是否自动生成主键
- `schema`: CollectionSchema 对象（高级模式使用）

---

### 3. insert

```python
def insert(
    self,
    collection_name: str,
    data: Union[Dict, List[Dict]],
    timeout: Optional[float] = None,
    partition_name: Optional[str] = "",
    **kwargs,
) -> Dict
```

插入数据到集合。

**参数：**
- `collection_name`: 集合名称
- `data`: 要插入的数据（单个字典或字典列表）

**返回：**
- `Dict`: 包含 insert_count 和 ids

**特性：**
- 支持自动生成主键（当 auto_id=True 时）
- 支持向量字段自动存储到独立的列

---

### 4. upsert

```python
def upsert(
    self,
    collection_name: str,
    data: Union[Dict, List[Dict]],
    timeout: Optional[float] = None,
    partition_name: Optional[str] = "",
    **kwargs,
) -> Dict
```

插入或更新数据（存在则更新，不存在则插入）。

---

### 5. search

```python
def search(
    self,
    collection_name: str,
    data: Union[List[List[float]], List[float]],
    limit: int = 10,
    output_fields: Optional[List[str]] = None,
    filter: Optional[str] = "",
    timeout: Optional[float] = None,
    anns_field: str = "",
    search_params: Optional[dict] = None,
    **kwargs,
) -> SearchResult
```

向量相似度搜索。

**参数：**
- `collection_name`: 集合名称
- `data`: 查询向量
- `limit`: 返回结果数量
- `output_fields`: 返回的字段列表
- `filter`: 过滤条件（SQL WHERE 子句）
- `anns_field`: 向量字段名
- `search_params`: 搜索参数（如 nprobe, ef 等）

**返回：**
- SearchResult 对象（兼容 Milvus 格式）

**特性：**
- 支持 IVF_FLAT 索引（nprobe 参数）
- 支持 HNSW 索引（ef 参数）
- 支持多向量字段返回

---

### 6. query

```python
def query(
    self,
    collection_name: str,
    filter: Optional[str] = "",
    output_fields: Optional[List[str]] = None,
    timeout: Optional[float] = None,
    limit: Optional[int] = None,
    **kwargs,
) -> HybridExtraList
```

标量字段查询。

**参数：**
- `collection_name`: 集合名称
- `filter`: 过滤条件（SQL WHERE 子句）
- `output_fields`: 返回的字段列表

**返回：**
- HybridExtraList 对象（兼容 Milvus 格式）

---

### 7. delete

```python
def delete(
    self,
    collection_name: str,
    ids: Optional[List[int]] = None,
    timeout: Optional[float] = None,
    filter: Optional[str] = "",
    **kwargs,
) -> Dict
```

删除数据。

**参数：**
- `collection_name`: 集合名称
- `ids`: 要删除的记录 ID 列表
- `filter`: 过滤条件（SQL WHERE 子句）

---

### 8. update

```python
def update(
    self,
    collection_name: str,
    data: Dict,
    ids: List[int] = None,
    timeout: Optional[float] = None,
    filter: Optional[str] = "",
    **kwargs,
) -> Dict
```

更新数据。

---

### 9. hybrid_search

```python
def hybrid_search(
    self,
    collection_name: str,
    reqs: List[AnnSearchRequest],
    ranker: Union[WeightedRanker, RRFRanker],
    limit: int = 10,
    output_fields: Optional[List[str]] = None,
    timeout: Optional[float] = None,
    **kwargs,
) -> HybridHits
```

混合搜索（多向量融合搜索）。

**参数：**
- `collection_name`: 集合名称
- `reqs`: AnnSearchRequest 列表，每个请求对应一个向量字段
- `ranker`: 排名器（WeightedRanker 或 RRFRanker）
- `limit`: 返回结果数量
- `output_fields`: 返回的字段列表

**特性：**
- 支持 WeightedRanker 加权排名
- 支持 RRFRanker reciprocal rank fusion 排名
- 支持分数归一化（arctan 方法）

---

### 10. create_index

```python
def create_index(
    self,
    collection_name: str,
    index_params: Optional[dict] = None,
    timeout: Optional[float] = None,
    field_name: Optional[str] = "",
    **kwargs,
)
```

创建索引。

**参数：**
- `collection_name`: 集合名称
- `index_params`: 索引参数（index_type, metric_type, params）
- `field_name`: 字段名

**支持的索引类型：**
- IVF_FLAT：倒排索引
- HNSW：分层可导航小世界图索引

---

### 11. describe_index

```python
def describe_index(
    self,
    collection_name: str,
    index_name: Optional[str] = "",
    timeout: Optional[float] = None,
    **kwargs,
) -> dict
```

获取索引详细信息。

---

### 12. drop_index

```python
def drop_index(
    self,
    collection_name: str,
    index_name: Optional[str] = "",
    timeout: Optional[float] = None,
    field_name: Optional[str] = "",
    **kwargs,
)
```

删除索引。

---

### 13. list_indexes

```python
def list_indexes(
    self,
    collection_name: str,
    field_name: Optional[str] = "",
    **kwargs,
) -> List[str]
```

列出集合的所有索引。

---

### 14. has_collection

```python
def has_collection(
    self,
    collection_name: str,
    timeout: Optional[float] = None,
    **kwargs,
) -> bool
```

检查集合是否存在。

---

### 15. list_collections

```python
def list_collections(self, **kwargs) -> List[str]
```

列出所有集合。

---

### 16. describe_collection

```python
def describe_collection(
    self,
    collection_name: str,
    timeout: Optional[float] = None,
    **kwargs,
) -> dict
```

获取集合信息。

---

### 17. drop_collection

```python
def drop_collection(
    self,
    collection_name: str,
    timeout: Optional[float] = None,
    **kwargs,
)
```

删除集合。

---

### 18. load_collection

```python
def load_collection(
    self,
    collection_name: str,
    *args,
    **kwargs,
)
```

加载集合到内存。
**UDBClient特性：**
- 刷新 client 对象的 _schemas 属性

---

### 19. release_collection

```python
def release_collection(self, *args, **kwargs)
```

释放集合内存。
UDBClient非必须release_collection 只是保持接口兼容
---

### 20. close

```python
def close(self)
```

关闭数据库连接。

---