import logging
import time
import json
from json import JSONEncoder
from typing import Dict, List, Optional, Union, Any
from functools import wraps

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from pymilvus import AnnSearchRequest
from pymilvus.client.abstract import BaseRanker
from pymilvus.milvus_client import MilvusClient
from pymilvus.milvus_client.base import BaseMilvusClient
from pymilvus.client.types import HybridExtraList, OmitZeroDict
from pymilvus.orm.collection import CollectionSchema, Function
from pymilvus.orm.types import DataType
from pymilvus.milvus_client.index import IndexParams
from pymilvus.orm.schema import FieldSchema
from pymilvus.orm.schema import StructFieldSchema
from pymilvus.client.abstract import WeightedRanker, RRFRanker
import re
import numpy as np
import math
import ast
from .udb_utils.udb_utils import MilvusFilterToSQL

logger = logging.getLogger(__name__)

class NumpyEncoder(JSONEncoder):
    """将 NumPy 类型转换为 Python 原生类型的 JSON 编码器"""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # 将数组转换为列表
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.str_):
            return str(obj)
        elif isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
        elif isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        return super().default(obj)

def atomic(func):
    """函数装饰器，用于实现数据库操作的原子性

    被装饰的函数会在一个事务中执行，确保所有数据库操作要么全部成功，要么全部失败。

    Args:
        func: 要装饰的函数

    Returns:
        装饰后的函数
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # 开始事务
        self._cursor.execute("BEGIN TRANSACTION")

        try:
            # 执行函数
            result = func(self, *args, **kwargs)
            # 提交事务
            self._cursor.execute("COMMIT")
            return result
        except Exception as e:
            # 回滚事务
            self._cursor.execute("ROLLBACK")
            logger.error(f"Atomic operation failed: {e}")
            raise

    return wrapper

class UDBClient(MilvusClient):
    """UDB Client that wraps PostgreSQL as the backend storage"""
    
    def __init__(
        self,
        uri: str = "unvdb://unvdb:unvdb@localhost:5432/milvus",
        user: str = "",
        password: str = "",
        db_name: str = "milvus",
        token: str = "",
        timeout: Optional[float] = None,
        print_sql=False, # unvdb特有字段
        **kwargs,
    ) -> None:
        """Initialize the UDB Client

        Args:
            uri (str): PostgreSQL connection string or host:port
            user (str): PostgreSQL username
            password (str): PostgreSQL password
            db_name (str): PostgreSQL database name
            token (str): Token for authentication (not used for PostgreSQL, but included for compatibility with MilvusClient)
            timeout (Optional[float]): Timeout for operations (not used for PostgreSQL, but included for compatibility with MilvusClient)
        """
        # Store token and timeout for compatibility with MilvusClient
        self.token = token
        self.timeout = timeout
        self.print_sql = print_sql
        # Parse connection string or use provided credentials
        if uri.startswith("unvdb://"):
            self._conn_string = uri.replace("unvdb://", "postgresql://", 1)
        else:
            # Build connection string from credentials
            self._conn_string = f"postgresql://{user}:{password}@{uri}/{db_name}"

        # Establish connection to PostgreSQL
        try:
            self._conn = psycopg2.connect(self._conn_string)
            self._conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            self._cursor = self._conn.cursor()
        except Exception as e:
            logger.error(f"Failed to connect to unvdb: {e}")
            raise

        # Initialize schema cache
        self._schemas = {}

        # Create necessary tables if they don't exist
        self._init_tables()

    def get_server_version(self, timeout=0, detail=False, **kwargs) -> Union[dict, str, None]:
        """
        """
        self._cursor.execute("SELECT VERSION()")

        version = self._cursor.fetchone()
        version = version[0] if version else ""
        if detail:
            return {"version": version}
        return version

    @classmethod
    def create_schema(cls, **kwargs):
        """Create a collection schema.

        Args:
            **kwargs: Additional keyword arguments for schema creation.

        Returns:
            CollectionSchema: The created collection schema.
        """
        kwargs["check_fields"] = False  # do not check fields for now
        return CollectionSchema([], **kwargs)

    @classmethod
    def create_struct_field_schema(cls) -> StructFieldSchema:
        """Create a struct field schema.

        Returns:
            StructFieldSchema: The created struct field schema.
        """
        return StructFieldSchema()

    @classmethod
    def create_field_schema(
        cls, name: str, data_type: DataType, desc: str = "", **kwargs
    ) -> FieldSchema:
        """Create a field schema. Wrapping orm.FieldSchema.

        Args:
            name (str): The name of the field.
            data_type (DataType): The data type of the field.
            desc (str): The description of the field.
            **kwargs: Additional keyword arguments.

        Returns:
            FieldSchema: the FieldSchema created.
        """
        return FieldSchema(name, data_type, desc, **kwargs)

    def _init_tables(self):
        """Create necessary tables in PostgreSQL"""
        # Enable pgvector extension if not already enabled
        # try:
        #     self._cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        #     logger.info("pgvector extension enabled successfully")
        # except Exception as e:
        #     logger.error(f"Failed to enable pgvector extension: {e}")
        #     raise
        
        # Create collections table to store collection schema information
        self._cursor.execute("""
        CREATE TABLE IF NOT EXISTS udb_collections (
            name VARCHAR(255) PRIMARY KEY,
            schema JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Create indexes table to track created indexes
        self._cursor.execute("""
        CREATE TABLE IF NOT EXISTS udb_indexes (
            id SERIAL PRIMARY KEY,
            collection_name VARCHAR(255) NOT NULL,
            field_name VARCHAR(255) NOT NULL,
            index_name VARCHAR(255) NOT NULL,
            index_type VARCHAR(255) NOT NULL,
            params JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (collection_name) REFERENCES udb_collections(name) ON DELETE CASCADE,
            UNIQUE (collection_name, field_name),
            UNIQUE (collection_name, index_name)
        )
        """)
        
        logger.info("Tables created successfully")

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
    ):
        """Create a collection in PostgreSQL
        
        Args:
            collection_name (str): The name of the collection to create
            dimension (Optional[int]): The dimension of the vector field (required if schema is not provided)
            primary_field_name (str): The name of the primary key field (default: "id")
            id_type (str): The data type of the primary key field (default: "int")
            vector_field_name (str): The name of the vector field (default: "vector")
            metric_type (str): The metric type for vector similarity (default: "COSINE")
                Options: "L2", "IP", "COSINE"
            auto_id (bool): Whether to automatically generate IDs (default: False)
            timeout (Optional[float]): Timeout for the operation in seconds (not used for PostgreSQL, included for compatibility)
            schema (Optional[CollectionSchema]): The schema of the collection (if provided, other parameters are ignored)
            **kwargs: Additional keyword arguments for compatibility with MilvusClient
        
        Returns:
            None: This method does not return a value
        """
        # Check if collection exists
        if self.has_collection(collection_name):
            logger.warning(f"Collection {collection_name} already exists")
            return

        # Create collection record
        if schema:
            # Use provided schema
            schema_dict = {
                "fields": []
            }
            for field in schema.fields:
                field_dict = {
                    "name": field.name,
                    "type": field.dtype.name.lower(),
                    "is_primary": field.is_primary,
                    "auto_id": field.auto_id
                }
                if field.dtype.name.lower() == "binary_vector":
                    raise Exception(f"Binary vector field ({field.name}) is not supported")

                if field.dtype.name.lower() in ["vector", "float_vector", "float16_vector"]:
                    field_dict["dimension"] = field.params.get("dim", dimension)
                    if not field.params.get("dim", dimension):
                        raise  Exception(f"dim is required for vector field. field name: {field.name}")
                    field_dict["metric_type"] = metric_type
                schema_dict["fields"].append(field_dict)
        else:
            # Default minimal schema with id and vector fields (compatible with Milvus)
            schema_dict = {
                "fields": [
                    {
                        "name": primary_field_name,
                        "type": id_type,
                        "is_primary": True,
                        "auto_id": auto_id
                    },
                    {
                        "name": vector_field_name,
                        "type": "vector",
                        "dimension": dimension,
                        "metric_type": metric_type
                    }
                ]
            }

        # Start transaction
        self._cursor.execute("BEGIN TRANSACTION")
        
        try:
            # Insert collection into udb_collections table
            self._cursor.execute(
                "INSERT INTO udb_collections (name, schema) VALUES (%s, %s)",
                (collection_name, json.dumps(schema_dict))
            )

            # Create a dedicated table for this collection with proper vector dimension
            # vector_fields = [f for f in schema_dict.get("fields", []) if f.get("type") in ["vector", "float_vector", "float16_vector", "binary_vector", "sparse_float_vector"]]
            vector_fields = [f for f in schema_dict.get("fields", []) if f.get("type") in ["vector", "float_vector", "float16_vector", "sparse_float_vector"]]
            if vector_fields:
                # Drop the table if it exists
                # self._cursor.execute(f"DROP TABLE IF EXISTS {collection_name}")

                # Build table creation SQL
                table_sql = f"CREATE TABLE {collection_name} ("
                
                # Determine primary key type based on schema
                primary_field = next((f for f in schema_dict.get("fields", []) if f.get("is_primary")), None)
                if primary_field:
                    field_type = primary_field.get("type")
                    auto_id = primary_field.get("auto_id", False)
                    
                    if field_type in ["int64", "long"]:
                        if auto_id:
                            table_sql += "id BIGSERIAL PRIMARY KEY,"
                        else:
                            table_sql += "id BIGINT NOT NULL PRIMARY KEY,"
                    elif field_type in ["int32", "int"]:
                        if auto_id:
                            table_sql += "id SERIAL PRIMARY KEY,"
                        else:
                            table_sql += "id INTEGER NOT NULL PRIMARY KEY,"
                    elif field_type in ["varchar", "string"]:
                        # For string types, auto_id is not supported in PostgreSQL
                        table_sql += "id VARCHAR(255) NOT NULL PRIMARY KEY,"
                    else:
                        # Default to VARCHAR if type is not recognized
                        table_sql += "id VARCHAR(255) NOT NULL PRIMARY KEY,"
                else:
                    # Default to VARCHAR if no primary field found
                    table_sql += "id VARCHAR(255) NOT NULL PRIMARY KEY,"
                
                table_sql += "data JSONB NOT NULL,"
                
                # Add vector columns for each vector field
                for field in vector_fields:
                    dimension = field.get("dimension", 128)
                    field_name = field.get("name")
                    field_type = field.get("type")
                    
                    # Map Milvus vector types to PostgreSQL types
                    if field_type in ["vector", "float_vector", "float16_vector"]:
                        # pgvector extension supports vector type
                        table_sql += f"{field_name} vector({dimension}),"
                    elif field_type == "binary_vector":
                        # pgvector extension supports binary vector as bit varying
                        table_sql += f"{field_name} bit varying({dimension}),"
                    elif field_type == "sparse_float_vector":
                        # pgvector extension does not support sparse vectors directly
                        # Store as JSONB for now
                        table_sql += f"{field_name} jsonb,"
                    else:
                        # Unknown vector type, use vector as default
                        table_sql += f"{field_name} vector({dimension}),"
                
                # Remove trailing comma
                table_sql = table_sql.rstrip(",")
                table_sql += ")"
                
                # Create the table
                self._cursor.execute(table_sql)
                logger.info(f"Created collection table {collection_name} with vector fields: {[f.get('name') for f in vector_fields]}")
            
            # Commit transaction
            self._cursor.execute("COMMIT")
        except Exception as e:
            # Rollback transaction if any error occurs
            self._cursor.execute("ROLLBACK")
            logger.error(f"Failed to create collection {collection_name}: {e}")
            raise

        # Cache the schema
        self._schemas[collection_name] = schema_dict
        
        # In minimal mode, automatically create vector index for vector field
        if not schema:
            # Get the vector field name
            vector_field_name = vector_field_name
            # Create index for vector field
            try:
                index_params = {
                    "index_type": "IVF_FLAT",
                    "metric_type": metric_type,
                    "params": {"nlist": dimension or 128}
                }
                self.create_index(
                    collection_name=collection_name,
                    index_params=index_params,
                    field_name=vector_field_name
                )
                logger.info(f"Automatically created index for vector field {vector_field_name}")
            except Exception as e:
                logger.error(f"Failed to create index automatically: {e}")

    def insert(
        self,
        collection_name: str,
        data: Union[Dict, List[Dict]],
        timeout: Optional[float] = None,
        partition_name: Optional[str] = "",
        **kwargs,
    ) -> Dict:
        """Insert data into the collection
        
        Args:
            collection_name (str): The name of the collection to insert data into
            data (Union[Dict, List[Dict]]): The data to insert. Can be a single dict or a list of dicts.
                Each dict should contain the primary key field and any other fields.
                Vector fields should be stored separately in the table.
            timeout (Optional[float]): Timeout for the operation in seconds (not used for PostgreSQL, included for compatibility)
            partition_name (Optional[str]): The name of the partition (not used for PostgreSQL, included for compatibility)
            **kwargs: Additional keyword arguments for compatibility with MilvusClient
        
        Returns:
            Dict: A dictionary containing:
                - insert_count (int): The number of records inserted
                - ids (List): The list of IDs of the inserted records
        """
        if isinstance(data, dict):
            data = [data]

        if not data:
            return {"insert_count": 0, "ids": []}

        # Get collection schema
        if collection_name not in self._schemas:
            self._load_schema(collection_name)

        schema = self._schemas[collection_name]
        primary_field = next(f for f in schema["fields"] if f["is_primary"])
        vector_fields = [f for f in schema["fields"] if f["type"] in ["vector", "float_vector", "float16_vector", "binary_vector", "sparse_float_vector"]]

        # Insert data into collection table
        inserted_ids = []
        for item in data:
            # Extract primary key
            pk = item.get(primary_field["name"])
            
            # Check if primary key is auto-generated
            auto_id = primary_field.get("auto_id", False)
            
            # If primary key is None and auto_id is True, use database auto-generation
            if pk is None and auto_id:
                # For auto_id, we'll let PostgreSQL generate the ID
                # We'll need to modify the SQL to exclude the id column
                use_auto_id = True
            else:
                # If primary key is None and auto_id is False, skip
                if pk is None:
                    continue
                use_auto_id = False

            # Create a copy of the item and remove vector fields
            item_copy = item.copy()
            vector_data = {}
            for field in vector_fields:
                field_name = field.get("name")
                if field_name in item_copy:
                    vector_data[field_name] = item_copy.pop(field_name)

            # Build insert SQL
            table_name = collection_name
            if use_auto_id:
                # For auto_id, exclude id column
                columns = ["data"]
                values = [json.dumps(item_copy, cls=NumpyEncoder)]
                placeholders = ["%s"]
                update_set = ["data = EXCLUDED.data"]
            else:
                # Include id column
                columns = ["id", "data"]
                
                # Determine primary key type
                pk_type = primary_field.get("type")
                if pk_type in ["int64", "long", "int32", "int"]:
                    # Keep numeric types as is
                    values = [pk, json.dumps(item_copy, cls=NumpyEncoder)]
                else:
                    # Convert to string for other types
                    values = [str(pk), json.dumps(item_copy, cls=NumpyEncoder)]
                    
                placeholders = ["%s", "%s"]
                update_set = ["data = EXCLUDED.data"]
            
            # Add vector fields
            for field in vector_fields:
                field_name = field.get("name")
                field_type = field.get("type")
                
                if field_name in vector_data:
                    vector = vector_data[field_name]
                    # Check if vector is not None and not empty
                    if vector is not None and len(vector) > 0:
                        # Convert numpy array to list if needed
                        if isinstance(vector, np.ndarray):
                            vector = vector.tolist()
                        
                        columns.append(field_name)
                        values.append(vector)
                        
                        # Handle different vector types
                        
                        if field_type in ["vector", "float_vector", "float16_vector"]:
                            # Use vector type for dense vectors
                            placeholders.append("%s::vector")
                            update_set.append(f"{field_name} = EXCLUDED.{field_name}::vector")
                        elif field_type == "binary_vector":
                            # Use bit varying for binary vectors
                            # Convert binary vector to bit string
                            binary_str = ''.join(str(bit) for bit in vector)
                            values[-1] = binary_str  # Replace the vector with the bit string
                            placeholders.append("%s")
                            update_set.append(f"{field_name} = EXCLUDED.{field_name}")
                        elif field_type == "sparse_float_vector":
                            # Use jsonb for sparse vectors
                            # Convert sparse vector to JSON string
                            values[-1] = json.dumps(vector)  # Replace the vector with the JSON string
                            placeholders.append("%s")
                            update_set.append(f"{field_name} = EXCLUDED.{field_name}")
                        else:
                            # Default to vector type
                            placeholders.append("%s::vector")
                            update_set.append(f"{field_name} = EXCLUDED.{field_name}::vector")
            
            # Build SQL statement
            columns_str = ", ".join(columns)
            placeholders_str = ", ".join(placeholders)
            
            if use_auto_id:
                # For auto_id, use RETURNING to get the generated ID
                sql = f"""
                INSERT INTO {table_name} ({columns_str})
                VALUES ({placeholders_str})
                RETURNING id
                """
                
                # Execute insert and get the generated ID
                self._cursor.execute(sql, values)
                generated_id = self._cursor.fetchone()[0]
                inserted_ids.append(generated_id)
            else:
                # For manual ID, use regular insert
                sql = f"""
                INSERT INTO {table_name} ({columns_str})
                VALUES ({placeholders_str})
                """
                
                # Execute insert
                self._cursor.execute(sql, values)
                inserted_ids.append(pk)

        return OmitZeroDict({
            "insert_count": len(inserted_ids),
            "ids": inserted_ids
        })

    def upsert(
        self,
        collection_name: str,
        data: Union[Dict, List[Dict]],
        timeout: Optional[float] = None,
        partition_name: Optional[str] = "",
        **kwargs,
    ) -> Dict:
        """Upsert data into the collection (insert or update if exists)
        
        Args:
            collection_name (str): The name of the collection to upsert data into
            data (Union[Dict, List[Dict]]): The data to upsert. Can be a single dict or a list of dicts.
                Each dict should contain the primary key field and any other fields.
                Vector fields should be stored separately in the table.
            timeout (Optional[float]): Timeout for the operation in seconds (not used for PostgreSQL, included for compatibility)
            partition_name (Optional[str]): The name of the partition (not used for PostgreSQL, included for compatibility)
            **kwargs: Additional keyword arguments for compatibility with MilvusClient
        
        Returns:
            Dict: A dictionary containing:
                - upsert_count (int): The number of records upserted
                - ids (List): The list of IDs of the upserted records
        """
        if isinstance(data, dict):
            data = [data]

        if not data:
            return {"upsert_count": 0, "ids": []}

        # Get collection schema
        if collection_name not in self._schemas:
            self._load_schema(collection_name)

        schema = self._schemas[collection_name]
        primary_field = next(f for f in schema["fields"] if f["is_primary"])
        vector_fields = [f for f in schema["fields"] if f["type"] in ["vector", "float_vector", "float16_vector", "binary_vector", "sparse_float_vector"]]

        # Upsert data into collection table
        upserted_ids = []
        for item in data:
            # Extract primary key
            pk = item.get(primary_field["name"])
            if pk is None:
                continue

            # Create a copy of the item and remove vector fields
            item_copy = item.copy()
            vector_data = {}
            for field in vector_fields:
                field_name = field.get("name")
                if field_name in item_copy:
                    vector_data[field_name] = item_copy.pop(field_name)

            # Build upsert SQL
            table_name = collection_name
            columns = ["id", "data"]
            
            # Determine primary key type
            pk_type = primary_field.get("type")
            if pk_type in ["int64", "long", "int32", "int"]:
                # Keep numeric types as is
                values = [pk, json.dumps(item_copy)]
            else:
                # Convert to string for other types
                values = [str(pk), json.dumps(item_copy)]
                
            placeholders = ["%s", "%s"]
            update_set = ["data = EXCLUDED.data"]
            
            # Add vector fields
            for field in vector_fields:
                field_name = field.get("name")
                field_type = field.get("type")
                
                if field_name in vector_data:
                    vector = vector_data[field_name]
                    # Check if vector is not None and not empty
                    if vector is not None and len(vector) > 0:
                        # Convert numpy array to list if needed
                        if isinstance(vector, np.ndarray):
                            vector = vector.tolist()
                        
                        columns.append(field_name)
                        values.append(vector)
                        
                        # Handle different vector types
                        
                        if field_type in ["vector", "float_vector", "float16_vector"]:
                            # Use vector type for dense vectors
                            placeholders.append("%s::vector")
                            update_set.append(f"{field_name} = EXCLUDED.{field_name}::vector")
                        elif field_type == "binary_vector":
                            # Use bit varying for binary vectors
                            # Convert binary vector to bit string
                            binary_str = ''.join(str(bit) for bit in vector)
                            values[-1] = binary_str  # Replace the vector with the bit string
                            placeholders.append("%s")
                            update_set.append(f"{field_name} = EXCLUDED.{field_name}")
                        elif field_type == "sparse_float_vector":
                            # Use jsonb for sparse vectors
                            # Convert sparse vector to JSON string
                            values[-1] = json.dumps(vector)  # Replace the vector with the JSON string
                            placeholders.append("%s")
                            update_set.append(f"{field_name} = EXCLUDED.{field_name}")
                        else:
                            # Default to vector type
                            placeholders.append("%s::vector")
                            update_set.append(f"{field_name} = EXCLUDED.{field_name}::vector")
            
            # Build SQL statement with ON CONFLICT DO UPDATE
            columns_str = ", ".join(columns)
            placeholders_str = ", ".join(placeholders)
            update_set_str = ", ".join(update_set)
            
            sql = f"""
            INSERT INTO {table_name} ({columns_str})
            VALUES ({placeholders_str})
            ON CONFLICT (id) DO UPDATE
            SET {update_set_str}
            """
            
            # Execute upsert
            self._cursor.execute(sql, values)
            upserted_ids.append(pk)

        return OmitZeroDict({
            "upsert_count": len(upserted_ids),
            "ids": upserted_ids
        })

    def search(
        self,
        collection_name: str,
        data: Optional[Union[List[list], list]] = None,
        filter: str = "",
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[dict] = None,
        timeout: Optional[float] = None,
        partition_names: Optional[List[str]] = None,
        anns_field: Optional[str] = None,
        **kwargs,
    ) -> List[List[dict]]:
        """Search for vectors in the collection
        
        Args:
            collection_name (str): The name of the collection to search in
            data (Optional[Union[List[list], list]]): The query vectors to search for.
                Can be a single vector or a list of vectors.
                Each vector should be a list of float values.
            filter (str): The filter expression to apply to the search results.
                Supports basic comparisons: =, !=, >, <, >=, <=
                Example: "value > 90", "tag = 'tag_1'"
            limit (int): The maximum number of results to return for each query vector (default: 10)
            output_fields (Optional[List[str]]): The fields to include in the results.
                If None, all fields are included.
            search_params (Optional[dict]): The search parameters (not used for PostgreSQL, included for compatibility)
                Example: {"nprobe": 10}
            timeout (Optional[float]): Timeout for the operation in seconds (not used for PostgreSQL, included for compatibility)
            partition_names (Optional[List[str]]): The names of the partitions to search in (not used for PostgreSQL, included for compatibility)
            anns_field (Optional[str]): The name of the vector field to search in.
                If None, the first vector field in the collection is used.
            **kwargs: Additional keyword arguments for compatibility with MilvusClient
        
        Returns:
            List[List[dict]]: A list of result lists, one for each query vector.
                Each result is a dict containing:
                - id: The ID of the record
                - distance: The distance between the query vector and the result vector
                - (other fields): The requested output fields
        """
        # Check if data is None or empty
        if data is None:
            return []
        
        # Convert numpy array to list if needed
        if isinstance(data, np.ndarray):
            data = data.tolist()
        
        # Check if data is empty after conversion
        if not data:
            return []

        # Get collection schema
        if collection_name not in self._schemas:
            self._load_schema(collection_name)

        schema = self._schemas[collection_name]
        # Get the specified vector field or use the first one
        if anns_field:
            vector_field = next((f for f in schema["fields"] if f["type"] in ["vector", "float_vector", "float16_vector", "binary_vector", "sparse_float_vector"] and f["name"] == anns_field), None)
            if not vector_field:
                raise ValueError(f"Vector field {anns_field} not found in collection {collection_name}")
        else:
            vector_field = next(f for f in schema["fields"] if f["type"] in ["vector", "float_vector", "float16_vector", "binary_vector", "sparse_float_vector"])

        vector_field_name = vector_field.get("name")
        vector_field_type = vector_field.get("type")

        # Get all vector fields and their types
        vector_fields = [f for f in schema["fields"] if f["type"] in ["vector", "float_vector", "float16_vector", "binary_vector", "sparse_float_vector"]]
        vector_field_names = [f.get("name") for f in vector_fields]
        vector_field_types = {f.get("name"): f.get("type") for f in vector_fields}

        results = []
        for vector in data:
            # Convert numpy array to list if needed
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()
            
            # Use PostgreSQL's vector similarity search
            # Use direct type cast to vector for the query vector
            table_name = collection_name
            
            # Get nprobe and ef_search from search_params if provided
            nprobe = 16  # Default value for IVF_FLAT
            ef_search = 64  # Default value for HNSW
            if search_params:
                if "nprobe" in search_params:
                    nprobe = search_params["nprobe"]
                if "ef" in search_params:
                    ef_search = search_params["ef"]
            
            # Set index-specific parameters for the current transaction if it's a dense vector search
            if vector_field_type in ["vector", "float_vector", "float16_vector"]:
                # Set ivfflat.probes for IVF_FLAT index
                self._cursor.execute(f"SET LOCAL ivfflat.probes = {nprobe}")
                # Set hnsw.ef_search for HNSW index
                self._cursor.execute(f"SET LOCAL hnsw.ef_search = {ef_search}")

            # Handle different vector types
            # Add all vector fields to the SELECT clause
            vector_fields_str = ", " + ", ".join(vector_field_names) if vector_field_names else ""
            
            if vector_field_type in ["vector", "float_vector", "float16_vector"]:
                # Get metric type from vector field
                metric_type = vector_field.get("metric_type", "L2")
                
                # Determine distance operator based on metric type
                if metric_type == "IP":
                    operator = "<#>"
                elif metric_type == "COSINE":
                    operator = "<=>"
                else:  # Default to L2
                    operator = "<->"
                
                # Use vector type for dense vectors
                query = f"""
                SELECT id, data, {vector_field_name} {operator} %s::vector as distance{vector_fields_str}
                FROM {table_name}
                """
                params = [vector]
            elif vector_field_type == "binary_vector":
                # Use bit varying for binary vectors
                # Note: pgvector doesn't support binary vector similarity directly
                # For now, we'll just return all results without similarity ranking
                query = f"""
                SELECT id, data, 0 as distance{vector_fields_str}
                FROM {table_name}
                """
                # Always initialize params as a list
                params = []
            elif vector_field_type == "sparse_float_vector":
                # Use jsonb for sparse vectors
                # Note: pgvector doesn't support sparse vector similarity directly
                # For now, we'll just return all results without similarity ranking
                query = f"""
                SELECT id, data, 0 as distance{vector_fields_str}
                FROM {table_name}
                """
                # Always initialize params as a list
                params = []
            else:
                # Default to vector type
                query = f"""
                SELECT id, data, {vector_field_name} <-> %s::vector as distance{vector_fields_str}
                FROM {table_name}
                """
                params = [vector]

            # Add filter if provided
            if filter:
                query = self._process_filter(filter, query, collection_name)

            # Order by distance and limit
            if vector_field_type in ["vector", "float_vector", "float16_vector"]:
                # For dense vectors, order by distance
                query += " ORDER BY distance LIMIT %s"
                params.append(limit)
            else:
                # For binary and sparse vectors, just limit the results
                query += " LIMIT %s"
                # Always ensure params has at least one element for LIMIT
                if not params:
                    params = []
                params.append(limit)

            # Execute query
            if self.print_sql:
                print("======", query)
            self._cursor.execute(query, params)
            rows = self._cursor.fetchall()
            
            # Format results
            hits = []
            for row in rows:
                data_dict = row[1]
                # Convert JSON string to dict if needed
                if isinstance(data_dict, str):
                    data_dict = json.loads(data_dict)
                
                hit = {
                    "id": row[0],
                    "distance": row[2]
                }
                # Add output fields
                if output_fields:
                    for field in output_fields:
                        if field in data_dict:
                            hit[field] = data_dict[field]
                        elif field in vector_field_names:
                            # Get vector field index in the row
                            field_index = vector_field_names.index(field) + 3  # 0: id, 1: data, 2: distance, 3+: vector fields
                            if field_index < len(row):
                                field_type = vector_field_types.get(field)
                                if field_type == "float16_vector":
                                    # Convert to numpy float16 bytes to match Milvus behavior
                                    vector = row[field_index]
                                    if vector:
                                        # Check if vector is a string representation of a list
                                        if isinstance(vector, str):
                                            # Convert string to list
                                            vector = ast.literal_eval(vector)
                                        np_vector = np.array(vector, dtype=np.float16)
                                        hit[field] = np_vector.tobytes()
                                    else:
                                        hit[field] = b''
                                else:
                                    hit[field] = row[field_index]
                else:
                    hit.update(data_dict)
                    # Add vector fields
                    for i, field_name in enumerate(vector_field_names):
                        field_index = i + 3  # 0: id, 1: data, 2: distance, 3+: vector fields
                        if field_index < len(row):
                            field_type = vector_field_types.get(field_name)
                            if field_type == "float16_vector":
                                # Convert to numpy float16 bytes to match Milvus behavior
                                vector = row[field_index]
                                if vector:
                                    # Check if vector is a string representation of a list
                                    if isinstance(vector, str):
                                        # Convert string to list
                                        vector = ast.literal_eval(vector)
                                    np_vector = np.array(vector, dtype=np.float16)
                                    hit[field_name] = np_vector.tobytes()
                                else:
                                    hit[field_name] = b''
                            else:
                                hit[field_name] = row[field_index]
                hits.append(hit)
            results.append(hits)

        return self._handler_search_result(results, output_fields)

    def query(
        self,
        collection_name: str,
        filter: str = "",
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        ids: Optional[Union[List, str, int]] = None,
        partition_names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[dict]:
        """Query data from the collection
        
        Args:
            collection_name (str): The name of the collection to query
            filter (str): The filter expression to apply to the query.
                Supports basic comparisons: =, !=, >, <, >=, <=
                Example: "value > 90", "tag = 'tag_1'"
            output_fields (Optional[List[str]]): The fields to include in the results.
                If None, all fields are included.
            timeout (Optional[float]): Timeout for the operation in seconds (not used for PostgreSQL, included for compatibility)
            ids (Optional[Union[List, str, int]]): The IDs of the records to query.
                Can be a single ID or a list of IDs.
                If provided, filter is ignored.
            partition_names (Optional[List[str]]): The names of the partitions to query in (not used for PostgreSQL, included for compatibility)
            **kwargs: Additional keyword arguments for compatibility with MilvusClient
        
        Returns:
            List[dict]: A list of result records.
                Each result is a dict containing:
                - id: The ID of the record
                - (other fields): The requested output fields or all fields if output_fields is None
        """
        # Use collection name as table name
        table_name = collection_name
        
        # Build query
        query = f"""
        SELECT id, data
        FROM {table_name}
        """
        
        # Add vector fields to select
        schema = self._schemas[collection_name]
        vector_fields = [f for f in schema["fields"] if f["type"] in ["vector", "float_vector", "float16_vector", "binary_vector", "sparse_float_vector"]]
        vector_field_names = [f.get("name") for f in vector_fields]
        vector_field_types = {f.get("name"): f.get("type") for f in vector_fields}
        
        if vector_field_names:
            vector_fields_str = ", " + ", ".join(vector_field_names)
            query = query.replace("SELECT id, data", f"SELECT id, data{vector_fields_str}")
        params = []

        # Add ids filter
        if ids:
            if isinstance(ids, (int, str)):
                ids = [ids]
            ids_str = [str(id) for id in ids]
            query += " WHERE id IN %s"
            params.append(tuple(ids_str))
        # Add filter
        elif filter:
            # Convert filter to JSONB query
            query = self._process_filter(filter, query, collection_name)

        # Execute query
        self._cursor.execute(query, params)
        rows = self._cursor.fetchall()

        # Format results
        results = []
        for row in rows:
            data_dict = row[1]
            # Convert JSON string to dict if needed
            if isinstance(data_dict, str):
                data_dict = json.loads(data_dict)
            
            result = {
                "id": row[0]
            }
            # Add output fields
            if output_fields:
                for field in output_fields:
                    if field in data_dict:
                        result[field] = data_dict[field]
                    elif field in vector_field_names:
                        # Get vector field index in the row
                        field_index = vector_field_names.index(field) + 2  # 0: id, 1: data, 2+: vector fields
                        if field_index < len(row):
                            field_type = vector_field_types.get(field)
                            if field_type == "float16_vector":
                                # Convert to numpy float16 bytes to match Milvus behavior
                                vector = row[field_index]
                                if vector:
                                    # Check if vector is a string representation of a list
                                    if isinstance(vector, str):
                                        # Convert string to list
                                        vector = ast.literal_eval(vector)
                                    np_vector = np.array(vector, dtype=np.float16)
                                    result[field] = np_vector.tobytes()
                                else:
                                    result[field] = b''
                            else:
                                result[field] = row[field_index]
            else:
                result.update(data_dict)
                # Add all vector fields
                for i, field_name in enumerate(vector_field_names):
                    field_index = i + 2  # 0: id, 1: data, 2+: vector fields
                    if field_index < len(row):
                        field_type = vector_field_types.get(field_name)
                        if field_type == "float16_vector":
                            # Convert to numpy float16 bytes to match Milvus behavior
                            vector = row[field_index]
                            if vector:
                                # Check if vector is a string representation of a list
                                if isinstance(vector, str):
                                    # Convert string to list
                                    vector = ast.literal_eval(vector)
                                np_vector = np.array(vector, dtype=np.float16)
                                result[field_name] = np_vector.tobytes()
                            else:
                                result[field_name] = b''
                        else:
                            result[field_name] = row[field_index]
            results.append(result)

        # 构建 HybridExtraList 返回类型
        # 准备 extra 信息
        extra = {
            "cost": 0,  # 简化处理，实际应该计算查询耗时
            "scanned_total_bytes": 0,  # 简化处理
            "cache_hit_ratio": 0.0  # 简化处理
        }

        # 创建 HybridExtraList
        hybrid_extra_list = HybridExtraList(
            [],  # 简化处理，没有延迟加载的字段
            results,  # 查询结果列表
            extra=extra,
            dynamic_fields=None,
            strict_float32=False,
            element_indices=None
        )

        return hybrid_extra_list

    def delete(
        self,
        collection_name: str,
        ids: Optional[Union[list, str, int]] = None,
        timeout: Optional[float] = None,
        filter: Optional[str] = None,
        partition_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, int]:
        """Delete data from the collection
        
        Args:
            collection_name (str): The name of the collection to delete data from
            ids (Optional[Union[list, str, int]]): The IDs of the records to delete.
                Can be a single ID or a list of IDs.
                If provided, filter is ignored.
            timeout (Optional[float]): Timeout for the operation in seconds (not used for PostgreSQL, included for compatibility)
            filter (Optional[str]): The filter expression to apply to the delete operation.
                Supports basic comparisons: =, !=, >, <, >=, <=
                Example: "value > 90", "tag = 'tag_1'"
                Ignored if ids is provided.
            partition_name (Optional[str]): The name of the partition (not used for PostgreSQL, included for compatibility)
            **kwargs: Additional keyword arguments for compatibility with MilvusClient
        
        Returns:
            Dict[str, int]: A dictionary containing:
                - delete_count (int): The number of records deleted
        """
        # Use collection name as table name
        table_name = collection_name
        
        # Build delete statement
        query = f"""
        DELETE FROM {table_name}
        """
        params = []

        # Add ids filter
        if ids:
            if isinstance(ids, (int, str)):
                ids = [ids]
            ids_str = [str(id) for id in ids]
            query += " WHERE id IN %s"
            params.append(tuple(ids_str))
        # Add filter
        elif filter:
            query = self._process_filter(filter, query, collection_name)

        # Execute delete
        self._cursor.execute(query, params)
        deleted_count = self._cursor.rowcount

        return OmitZeroDict({"delete_count": deleted_count})

    def update(
        self,
        collection_name: str,
        data: Union[Dict, List[Dict]],
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Dict:
        """Update data in the collection
        
        Args:
            collection_name (str): The name of the collection to update data in
            data (Union[Dict, List[Dict]]): The data to update. Can be a single dict or a list of dicts.
                Each dict should contain the primary key field and any other fields to update.
                Vector fields should be stored separately in the table.
            timeout (Optional[float]): Timeout for the operation in seconds (not used for PostgreSQL, included for compatibility)
            **kwargs: Additional keyword arguments for compatibility with MilvusClient
        
        Returns:
            Dict: A dictionary containing:
                - update_count (int): The number of records updated
        """
        if isinstance(data, dict):
            data = [data]

        if not data:
            return {"update_count": 0}

        # Get collection schema
        if collection_name not in self._schemas:
            self._load_schema(collection_name)

        schema = self._schemas[collection_name]
        primary_field = next(f for f in schema["fields"] if f["is_primary"])
        vector_fields = [f for f in schema["fields"] if f["type"] in ["vector", "float_vector", "float16_vector", "binary_vector", "sparse_float_vector"]]

        # Update data in collection table
        updated_count = 0
        for item in data:
            # Extract primary key
            pk = item.get(primary_field["name"])
            if pk is None:
                continue

            # Create a copy of the item and remove vector fields
            item_copy = item.copy()
            vector_data = {}
            for field in vector_fields:
                field_name = field.get("name")
                if field_name in item_copy:
                    vector_data[field_name] = item_copy.pop(field_name)

            # Build update SQL
            table_name = collection_name
            set_clauses = ["data = %s"]
            values = [json.dumps(item_copy)]
            
            # Add vector fields
            for field in vector_fields:
                field_name = field.get("name")
                field_type = field.get("type")
                
                if field_name in vector_data:
                    vector = vector_data[field_name]
                    if vector is not None and len(vector) > 0:
                        if isinstance(vector, np.ndarray):
                            vector = vector.tolist()
                        # Handle different vector types
                        if field_type in ["vector", "float_vector", "float16_vector"]:
                            # Use vector type for dense vectors
                            set_clauses.append(f"{field_name} = %s::vector")
                            values.append(vector)
                        elif field_type == "binary_vector":
                            # Use bit varying for binary vectors
                            # Convert binary vector to bit string
                            binary_str = ''.join(str(bit) for bit in vector)
                            set_clauses.append(f"{field_name} = %s")
                            values.append(binary_str)
                        elif field_type == "sparse_float_vector":
                            # Use jsonb for sparse vectors
                            # Convert sparse vector to JSON string
                            set_clauses.append(f"{field_name} = %s")
                            values.append(json.dumps(vector))
                        else:
                            # Default to vector type
                            set_clauses.append(f"{field_name} = %s::vector")
                            values.append(vector)
            
            # Add primary key to values
            # Determine primary key type
            pk_type = primary_field.get("type")
            if pk_type in ["int64", "long", "int32", "int"]:
                # Keep numeric types as is
                values.append(pk)
            else:
                # Convert to string for other types
                values.append(str(pk))
            
            # Build SQL statement
            set_clauses_str = ", ".join(set_clauses)
            sql = f"""
            UPDATE {table_name}
            SET {set_clauses_str}
            WHERE id = %s
            """
            
            # Execute update
            self._cursor.execute(sql, values)
            updated_count += self._cursor.rowcount

        return OmitZeroDict({
            "update_count": updated_count
        })

    def has_collection(self, collection_name: str, timeout: Optional[float] = None, **kwargs):
        """Check if a collection exists
        
        Args:
            collection_name (str): The name of the collection to check
            timeout (Optional[float]): Timeout for the operation in seconds (not used for PostgreSQL, included for compatibility)
            **kwargs: Additional keyword arguments for compatibility with MilvusClient
        
        Returns:
            bool: True if the collection exists, False otherwise
        """
        self._cursor.execute(
            "SELECT 1 FROM udb_collections WHERE name = %s",
            (collection_name,)
        )
        return self._cursor.fetchone() is not None

    def list_collections(self, **kwargs):
        """List all collections
        
        Args:
            **kwargs: Additional keyword arguments for compatibility with MilvusClient
        
        Returns:
            List[str]: A list of collection names
        """
        self._cursor.execute("SELECT name FROM udb_collections")
        return [row[0] for row in self._cursor.fetchall()]

    def drop_index(
        self,
        collection_name: str,
        index_name: str,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Drop an index
        
        Args:
            collection_name (str): The name of the collection to drop index from
            index_name (str): The name of the index to drop
            timeout (Optional[float]): Timeout for the operation in seconds (not used for PostgreSQL, included for compatibility)
            **kwargs: Additional keyword arguments for compatibility with MilvusClient
        
        Returns:
            None: This method does not return a value
        """
        # Get index information from udb_indexes table
        # First try to find by index_name
        self._cursor.execute(
            "SELECT field_name, index_name FROM udb_indexes WHERE collection_name = %s AND index_name = %s",
            (collection_name, index_name)
        )
        
        index_info = self._cursor.fetchone()
        if not index_info:
            # Try to find by field_name
            self._cursor.execute(
                "SELECT field_name, index_name FROM udb_indexes WHERE collection_name = %s AND field_name = %s",
                (collection_name, index_name)
            )
            index_info = self._cursor.fetchone()
            
            if not index_info:
                # Try to get by constructing the index name pattern
                self._cursor.execute(
                    "SELECT field_name, index_name FROM udb_indexes WHERE collection_name = %s",
                    (collection_name,)
                )
                field_names = [row[0] for row in self._cursor.fetchall()]
                
                # Check if index_name matches any field name
                if index_name not in field_names:
                    logger.warning(f"Index {index_name} does not exist in collection {collection_name}")
                    return
                
                # Find the index by field name
                self._cursor.execute(
                    "SELECT field_name, index_name FROM udb_indexes WHERE collection_name = %s AND field_name = %s",
                    (collection_name, index_name)
                )
                index_info = self._cursor.fetchone()
        
        if index_info:
            field_name, stored_index_name = index_info
            # Use the stored index name
            index_name = stored_index_name
        
        # Start transaction
        self._cursor.execute("BEGIN TRANSACTION")
        
        try:
            # Drop the index from the table
            try:
                self._cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
                logger.info(f"Dropped index {index_name} from collection {collection_name}")
            except Exception as e:
                logger.warning(f"Failed to drop index {index_name}: {e}")
            
            # Delete index from udb_indexes table
            self._cursor.execute(
                "DELETE FROM udb_indexes WHERE collection_name = %s AND index_name = %s",
                (collection_name, index_name)
            )
            
            # Commit transaction
            self._cursor.execute("COMMIT")
            
            logger.info(f"Dropped index for field {field_name} in collection {collection_name}")
        except Exception as e:
            # Rollback transaction if any error occurs
            self._cursor.execute("ROLLBACK")
            logger.error(f"Failed to drop index: {e}")
            raise

    def _handler_search_result(self, hits_list, output_fields):
        """将搜索结果转换为 Milvus 兼容的 SearchResult 格式
        
        Args:
            hits_list: 搜索结果列表，可以是：
                - List[List[dict]]: search 方法的多查询结果 [[hit1, hit2], [hit3, hit4]]
                - List[dict]: hybrid_search 的单查询结果 [hit1, hit2, hit3]
            output_fields: 输出字段列表
        
        Returns:
            SearchResult: Milvus 兼容的搜索结果对象
        """
        from pymilvus.client.search_result import SearchResult, HybridHits, Hit, SearchResultData

        # 如果是单层列表（hybrid_search 的结果），转换为双层列表（search 的结果格式）
        if hits_list and isinstance(hits_list, list) and len(hits_list) > 0:
            if isinstance(hits_list[0], dict):
                hits_list = [hits_list]

        if not hits_list:
            return SearchResult(SearchResultData())

        # 准备 SearchResultData
        search_result_data = SearchResultData()
        search_result_data.primary_field_name = "id"

        # 展平结果并记录每个查询的命中数
        all_pks = []
        all_scores = []
        topks = []

        for hits in hits_list:
            topks.append(len(hits))
            for hit in hits:
                all_pks.append(hit.get("id"))
                all_scores.append(hit.get("score", hit.get("distance", 0.0)))

        # 设置 ids 和 scores
        if all_pks and isinstance(all_pks[0], str):
            search_result_data.ids.str_id.data.extend(all_pks)
        else:
            search_result_data.ids.int_id.data.extend([int(pk) for pk in all_pks])
        search_result_data.scores.extend(all_scores)

        # 设置 topks
        search_result_data.topks.extend(topks)

        # 设置 output_fields
        if output_fields:
            search_result_data.output_fields.extend(output_fields)

        # 创建 SearchResult
        search_result = SearchResult(search_result_data)

        # 为每个查询向量创建 HybridHits
        start = 0
        for i, hits in enumerate(hits_list):
            end = start + len(hits)

            # 创建 HybridHits
            hybrid_hits = HybridHits(
                start=start,
                end=end,
                all_pks=all_pks,
                all_scores=all_scores,
                fields_data=[],
                output_fields=output_fields or [],
                highlight_results=[],
                pk_name="id"
            )

            # 填充 hybrid_hits
            for j, hit in enumerate(hits):
                hit_data = {"id": hit.get("id"), "distance": hit.get("score", hit.get("distance", 0.0))}
                hit_data["entity"] = hit.get("entity", {})

                # 添加其他字段到 entity
                for key, value in hit.items():
                    if key not in ["id", "distance", "score", "entity"]:
                        hit_data["entity"][key] = value

                hit_obj = Hit(hit_data, pk_name="id")
                if j < len(hybrid_hits):
                    hybrid_hits[j] = hit_obj

            # 替换 search_result 中的元素
            if i < len(search_result):
                search_result[i] = hybrid_hits

            start = end

        return search_result

    def hybrid_search(
        self,
        collection_name: str,
        reqs: List[AnnSearchRequest],
        ranker: Union[BaseRanker, Function],
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        partition_names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[List[dict]]:
        """Conducts multi vector similarity search with a rerank for rearrangement.
        
        Args:
            collection_name(``string``): The name of collection.
            reqs (``List[AnnSearchRequest]``): The vector search requests.
            ranker (``Union[BaseRanker, Function]``): The ranker.
            limit (``int``): The max number of returned record, also known as `topk`.

            partition_names (``List[str]``, optional): The names of partitions to search on.
            output_fields (``List[str]``, optional):
                The name of fields to return in the search result.  Can only get scalar fields.
            round_decimal (``int``, optional):
                The specified number of decimal places of returned distance.
                Defaults to -1 means no round to returned distance.
            timeout (``float``, optional): A duration of time in seconds to allow for the RPC.
                If timeout is set to None, the client keeps waiting until the server
                responds or an error occurs.
            **kwargs (``dict``): Optional search params

                * *offset* (``int``, optinal)
                    offset for pagination.

                * *consistency_level* (``str/int``, optional)
                    Which consistency level to use when searching in the collection.

                    Options of consistency level: Strong, Bounded, Eventually, Session, Customized.

                    Note: this parameter overwrites the same one specified when creating collection,
                    if no consistency level was specified, search will use the
                    consistency level when you create the collection.

        Returns:
            List[List[dict]]: A nested list of dicts containing the result data.

        Raises:
            MilvusException: If anything goes wrong
        """
        # 阶段1: 多路召回
        if not reqs:
            return []
        
        # Get collection schema
        if collection_name not in self._schemas:
            self._load_schema(collection_name)

        # 存储每个召回路的结果
        all_candidates = []
        
        # 执行每个搜索请求
        for req in reqs:
            # 提取参数
            data = req.data
            anns_field = req.anns_field
            search_params = getattr(req, "param", {}) or getattr(req, "search_params", {})
            filter_expr = getattr(req, "expr", "") or getattr(req, "filter", "")
            req_limit = getattr(req, "limit", limit)
            
            # 确保输出字段包含所有向量字段
            # 首先获取所有向量字段
            vector_fields = [f["name"] for f in self._schemas[collection_name]["fields"] 
                           if f["type"] in ["vector", "float_vector", "float16_vector", "binary_vector", "sparse_float_vector"]]
            
            # 合并用户指定的输出字段和向量字段
            combined_output_fields = []
            if output_fields:
                combined_output_fields = output_fields.copy()
            
            # 添加所有向量字段
            for field in vector_fields:
                if field not in combined_output_fields:
                    combined_output_fields.append(field)
            
            # 执行搜索
            results = self.search(
                collection_name=collection_name,
                data=data,
                filter=filter_expr,
                limit=req_limit,
                output_fields=combined_output_fields,
                search_params=search_params,
                timeout=timeout,
                partition_names=partition_names,
                anns_field=anns_field,
                **kwargs
            )
            
            # 存储结果
            if results and results[0]:
                all_candidates.append(results[0])

        # 阶段2: 结果融合重排
        # 去重
        unique_hits = {}
        # 存储每个结果在不同召回路中的分数
        hit_scores = {}
        
        # 获取每个搜索请求使用的度量类型、向量字段和查询向量
        metric_types = []
        anns_fields = []
        query_vectors = []
        for req in reqs:
            anns_field = req.anns_field
            anns_fields.append(anns_field)
            # 从 schema 中获取字段的度量类型
            field = next((f for f in self._schemas[collection_name]["fields"] if f["name"] == anns_field), None)
            if field and "metric_type" in field:
                metric_types.append(field["metric_type"])
            else:
                raise ValueError(f"Metric type not found for field {anns_field}")
            # 获取查询向量
            query_vectors.append(req.data[0])
        
        # 自定义重排类
        if not isinstance(ranker, (RRFRanker, WeightedRanker)):
            return ranker(*all_candidates)()
        
        # 收集所有唯一的命中结果和它们的向量字段值
        for i, candidates in enumerate(all_candidates):
            for hit in candidates:
                hit_id = hit.get("id")
                if hit_id not in unique_hits:
                    unique_hits[hit_id] = hit
                    hit_scores[hit_id] = {}
        
        # 为每个唯一结果计算所有召回路的分数
        if ranker:
            # 对每个唯一结果
            for hit_id, hit in unique_hits.items():
                # 对每个召回路计算分数
                for i in range(len(reqs)):
                    anns_field = anns_fields[i]
                    metric_type = metric_types[i]
                    query_vector = query_vectors[i]
                    
                    # 尝试从命中结果中获取该向量字段的值
                    vector_value = None
                    # 检查命中结果中是否直接包含向量字段
                    if anns_field in hit:
                        vector_value = hit[anns_field]
                    # 检查命中结果的 data 字段中是否包含向量字段
                    elif "data" in hit and isinstance(hit["data"], dict) and anns_field in hit["data"]:
                        vector_value = hit["data"][anns_field]
                    
                    # 计算 distance
                    distance = 0.0
                    if vector_value:
                        # 处理字符串形式的向量
                        if isinstance(vector_value, str):
                            import ast
                            try:
                                vector_value = ast.literal_eval(vector_value)
                            except:
                                # 如果解析失败，使用默认距离
                                distance = float('inf')
                        try:
                            vec1 = np.array(query_vector)
                            vec2 = np.array(vector_value)
                            
                            if metric_type == "L2":
                                # L2 距离
                                distance = np.linalg.norm(vec1 - vec2)
                            elif metric_type == "IP":
                                # 内积（负数，因为 pgvector 返回的是负数内积）
                                distance = -np.dot(vec1, vec2)
                            elif metric_type == "COSINE":
                                # 余弦距离（1 - 余弦相似度）
                                distance = 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                        except:
                            # 如果计算失败，使用默认距离
                            distance = float('inf')
                    
                    # 根据度量类型应用转换公式
                    if metric_type == "L2":
                        # L2 distance 转换公式 1.0 / (1.0 + distance)
                        raw_score = 1.0 / (1.0 + distance)
                    elif metric_type == "IP":
                        # IP distance 转换公式 -distance
                        raw_score = -distance
                    elif metric_type == "COSINE":
                        # COSIN distance 转换公式 1 - distance
                        raw_score = 1 - distance
                    else:
                        # 默认使用 L2 转换公式
                        raw_score = 1.0 / (1.0 + distance)
                    
                    # 应用 arctan 归一化
                    normalized_score = 2 * math.atan(raw_score) / math.pi
                    
                    # 存储该召回路的归一化分数
                    hit_scores[hit_id][i] = normalized_score
        
        # 转换为列表
        all_hits = list(unique_hits.values())
        
        # 使用 ranker 进行重排
        if ranker:
            # 检查是否是 WeightedRanker
            if isinstance(ranker, WeightedRanker):
                # 实现 WeightedRanker 逻辑
                weights = ranker._weights
                # norm_score = ranker._norm_score
                
                # 确保权重数量与召回路数量匹配
                if len(weights) < len(all_candidates):
                    # 如果权重不足，使用最后一个权重
                    weights.extend([weights[-1]] * (len(all_candidates) - len(weights)))
                
                # 计算每个结果的加权分数
                for hit in all_hits:
                    hit_id = hit.get("id")
                    scores = hit_scores.get(hit_id, {})
                    weighted_score = 0.0
                    
                    # 对每个召回路的分数应用权重
                    for i, score in scores.items():
                        if i < len(weights):
                            weighted_score += score * weights[i]
                    
                    # 由于每个召回路的分数已经归一化到 [0, 1] 范围，
                    # 加权后的分数也会在 [0, sum(weights)] 范围内
                    # 不需要再次归一化
                    hit["score"] = weighted_score
                
                # 按分数排序（降序）
                all_hits.sort(key=lambda x: x.get("score", 0), reverse=True)
            elif isinstance(ranker, RRFRanker):
                # 实现 RRFRanker 逻辑
                k = ranker._k or 60 # 默认 60
                
                hit_id_to_rank = {}
                # 为每个结果计算 RRF 分数
                for hit in all_hits:
                    hit_id = hit.get("id")
                    rrf_score = 0.0
                    
                    # 对每个召回路计算 RRF 分数
                    for i, candidates in enumerate(all_candidates):
                        # 查找该结果在当前召回路中的排名
                        
                        if not hit_id_to_rank.get((hit_id, i)):
                            rank = 1
                            found = False
                            for candidate in candidates:
                                # 记录当前召回路的排名
                                hit_id_to_rank[(candidate.get("id"), i)] = rank
                                if candidate.get("id") == hit_id:
                                    found = True
                                    break
                                rank += 1
                        else:
                            rank = hit_id_to_rank[(hit_id, i)]
                            found = True
                        # 如果找到，计算 RRF 分数
                        if found:
                            # 排名越大，score越小
                            rrf_score += 1.0 / (k + rank)
                    
                    # 存储 RRF 分数
                    hit["score"] = rrf_score
                
                # 按分数排序（降序）
                all_hits.sort(key=lambda x: x.get("score", 0), reverse=True)
            elif callable(ranker):
                # 如果 ranker 是函数，使用函数进行排序
                # 这里简化处理，实际应该调用函数进行排序
                all_hits.sort(key=lambda x: x.get("distance", 0))
            else:
                # 默认按距离排序
                all_hits.sort(key=lambda x: x.get("distance", 0))
        else:
            # 没有 ranker，默认按距离排序
            all_hits.sort(key=lambda x: x.get("distance", 0))
        
        # 限制结果数量
        all_hits = all_hits[:limit]
        search_result = self._handler_search_result(all_hits, output_fields)
        return search_result

    def drop_collection(self, collection_name: str, timeout: Optional[float] = None, **kwargs):
        """Drop a collection and all its data
        
        Args:
            collection_name (str): The name of the collection to drop
            timeout (Optional[float]): Timeout for the operation in seconds (not used for PostgreSQL, included for compatibility)
            **kwargs: Additional keyword arguments for compatibility with MilvusClient
        
        Returns:
            None: This method does not return a value
        """
        # Start transaction
        self._cursor.execute("BEGIN TRANSACTION")
        
        try:
            # Drop the collection table
            table_name = collection_name
            self._cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            # Delete collection record
            self._cursor.execute(
                "DELETE FROM udb_collections WHERE name = %s",
                (collection_name,)
            )
            
            # Delete indexes from udb_indexes table
            self._cursor.execute(
                "DELETE FROM udb_indexes WHERE collection_name = %s",
                (collection_name,)
            )
            
            # Commit transaction
            self._cursor.execute("COMMIT")
            
            # Remove from schema cache
            if collection_name in self._schemas:
                del self._schemas[collection_name]
            
            logger.info(f"Dropped collection {collection_name}")
        except Exception as e:
            # Rollback transaction if any error occurs
            self._cursor.execute("ROLLBACK")
            logger.error(f"Failed to drop collection {collection_name}: {e}")
            raise

    def describe_collection(self, collection_name: str, timeout: Optional[float] = None, **kwargs):
        """Describe a collection and return its schema
        
        Args:
            collection_name (str): The name of the collection to describe
            timeout (Optional[float]): Timeout for the operation in seconds (not used for PostgreSQL, included for compatibility)
            **kwargs: Additional keyword arguments for compatibility with MilvusClient
        
        Returns:
            dict: The schema of the collection, or None if the collection does not exist
                Contains:
                - fields: List of field definitions
                - (other schema properties)
        """
        self._cursor.execute(
            "SELECT schema FROM udb_collections WHERE name = %s",
            (collection_name,)
        )
        result = self._cursor.fetchone()
        if not result:
            return None
        # PostgreSQL may already return a dict for JSONB type
        if isinstance(result[0], str):
            return json.loads(result[0])
        return result[0]

    def _process_filter(self, filter: str, query: str, collection_name: str = None) -> str:
        """Process filter string to create valid SQL WHERE clause
        
        Args:
            filter (str): The filter string in Milvus syntax
            query (str): The existing SQL query
            collection_name (str): The name of the collection to get field types from
        
        Returns:
            str: The updated SQL query with filter applied
        """
        mfts = MilvusFilterToSQL()
        # Get field types from schema if collection_name is provided
        field_types = None
        if collection_name and collection_name in self._schemas:
            schema = self._schemas[collection_name]
            field_types = {}
            for field in schema.get("fields", []):
                field_name = field.get("name")
                field_type = field.get("type")
                if field_name and field_type:
                    field_types[field_name] = field_type
        return mfts.process_filter(filter, query, field_types=field_types)

    def _load_schema(self, collection_name: str):
        """Load collection schema from database"""
        self._cursor.execute(
            "SELECT schema FROM udb_collections WHERE name = %s",
            (collection_name,)
        )
        result = self._cursor.fetchone()
        if result:
            # UNVDB SQL may already return a dict for JSONB type
            if isinstance(result[0], str):
                self._schemas[collection_name] = json.loads(result[0])
            else:
                self._schemas[collection_name] = result[0]

    def close(self):
        """Close the connection"""
        if self._cursor:
            self._cursor.close()
        if self._conn:
            self._conn.close()
    
    def list_indexes(self, collection_name: str, field_name: Optional[str] = "", **kwargs):
        """List all indexes of collection. If `field_name` is not specified,
            return all the indexes of this collection, otherwise this interface will return
            all indexes on this field of the collection.
        
        Args:
            collection_name (str): The name of collection
            field_name (Optional[str]): The name of field.  If no field name is specified, all indexes
                of this collection will be returned.
            **kwargs: Additional keyword arguments for compatibility with MilvusClient
        
        Returns:
            List[str]: The name list of all indexes
        """
        # Check if collection exists
        if not self.has_collection(collection_name):
            return []
        
        # Query indexes from udb_indexes table
        query = "SELECT index_name FROM udb_indexes WHERE collection_name = %s"
        params = [collection_name]
        
        # If field_name is specified, filter by field_name
        if field_name:
            query += " AND field_name = %s"
            params.append(field_name)
        
        self._cursor.execute(query, params)
        indexes = self._cursor.fetchall()
        
        # Return stored index names
        index_name_list = []
        for index in indexes:
            index_name = index[0]
            index_name_list.append(index_name)
        
        return index_name_list

    # Other methods would need to be implemented similarly
    # For simplicity, we'll leave them as not implemented for now
    def create_index(
        self,
        collection_name: str,
        index_params: Optional[Union[dict, IndexParams]] = None,
        field_name: str = "vector",
        **kwargs,
    ):
        """Create index for a field
        
        Args:
            collection_name (str): The name of the collection to create index for
            index_params (Optional[Union[dict, IndexParams]]): The index parameters.
                If dict, should contain:
                    - index_type (str): The type of index (e.g., "IVF_FLAT")
                    - metric_type (str): The metric type (e.g., "L2", "IP", "COSINE")
                    - params (dict): Additional index parameters (e.g., {"nlist": 128})
                If IndexParams, will process each index in the list.
            field_name (str): The name of the field to create index for (default: "vector")
            **kwargs: Additional keyword arguments for compatibility with MilvusClient
        
        Returns:
            None: This method does not return a value
        """
        # Get collection schema
        if collection_name not in self._schemas:
            self._load_schema(collection_name)

        # Handle IndexParams type
        if isinstance(index_params, IndexParams):
            # For IndexParams, process each index in the list
            for index_param in index_params:
                # Extract field name from index_param
                current_field_name = index_param.field_name
                # Extract index configs
                configs = index_param.get_index_configs()
                configs = dict(configs)
                if "index_type" in configs:
                    index_type= configs.pop("index_type")
                else:
                    index_type = "IVF_FLAT"
                if "metric_type" in configs:
                    metric_type = configs.pop("metric_type")
                else:
                    metric_type = "L2"

                # Call create_index for each field
                self._create_index_for_field(
                    collection_name=collection_name,
                    field_name=current_field_name,
                    index_type=index_type,
                    metric_type=metric_type,
                    params=configs
                )
            return
        else:
            # Handle dict type or None
            # Get field info
            schema = self._schemas[collection_name]
            field = next((f for f in schema["fields"] if f["name"] == field_name), None)
            if not field:
                # If field not found in schema, assume it's a field in the data JSONB
                logger.info(f"Field {field_name} not found in schema, treating as JSONB field")
                is_vector_field = False
            else:
                is_vector_field = field["type"] in ["vector", "float_vector", "float16_vector", "binary_vector", "sparse_float_vector"]

            # Default index params
            if not index_params:
                if is_vector_field:
                    field_type = field["type"]
                    if field_type in ["vector", "float_vector", "float16_vector"]:
                        index_params = {
                            "index_type": "IVF_FLAT",
                            "metric_type": "L2",
                            "params": {"nlist": 128}
                        }
                    else:
                        # For binary and sparse vectors, use appropriate index type
                        index_params = {
                            "index_type": "BTREE" if field_type == "binary_vector" else "GIN",
                            "metric_type": "",
                            "params": {}
                        }
                else:
                    index_params = {
                        "index_type": "JSONB",
                        "metric_type": "",
                        "params": {}
                    }

            # Determine default index type based on field type
            if is_vector_field and field:
                field_type = field["type"]
                if field_type in ["vector", "float_vector", "float16_vector"]:
                    default_index_type = "IVF_FLAT"
                    default_metric_type = "L2"
                    default_params = {"nlist": 128}
                elif field_type == "binary_vector":
                    raise Exception(f"Index is not supported for {field_type} vector type.")
                    # default_index_type = "BTREE"
                    # default_metric_type = ""
                    # default_params = {}
                elif field_type == "sparse_float_vector":
                    default_index_type = "GIN"
                    default_metric_type = ""
                    default_params = {}
                else:
                    default_index_type = "JSONB"
                    default_metric_type = ""
                    default_params = {}
            else:
                default_index_type = "JSONB"
                default_metric_type = ""
                default_params = {}
            
            index_type = index_params.get("index_type", default_index_type)
            metric_type = index_params.get("metric_type", default_metric_type)
            params = index_params.get("params", default_params)
            
            # Call create_index for the field
            self._create_index_for_field(
                collection_name=collection_name,
                field_name=field_name,
                index_type=index_type,
                metric_type=metric_type,
                params=params
            )
            return
    
    @atomic
    def _create_index_for_field(
        self,
        collection_name: str,
        field_name: str,
        index_type: str,
        metric_type: str,
        params: dict,
    ):
        """Create index for a specific field"""
        # Get collection schema
        if collection_name not in self._schemas:
            self._load_schema(collection_name)

        schema = self._schemas[collection_name]
        field = next((f for f in schema["fields"] if f["name"] == field_name), None)
        if not field:
            # If field not found in schema, assume it's a field in the data JSONB
            logger.info(f"Field {field_name} not found in schema, treating as JSONB field")
            if field_name:
                raise Exception(f"Field {field_name} not found in schema, collection_name: {collection_name}")

            is_vector_field = False
        else:
            is_vector_field = field["type"] in ["vector", "float_vector", "float16_vector", "binary_vector", "sparse_float_vector"]

        # Generate index name
        index_name = f"idx_{collection_name}_{field_name}"

        # Check if index already exists
        self._cursor.execute(
            "SELECT id FROM udb_indexes WHERE collection_name = %s AND field_name = %s",
            (collection_name, field_name)
        )
        if self._cursor.fetchone():
            error_msg = f"Index already exists for field {field_name} in collection {collection_name}"
            logger.error(error_msg)
            raise Exception(error_msg)

        # Use collection name as table name
        table_name = collection_name

        # Create index based on field type
        if is_vector_field:
            # Check if field type is supported by pgvector
            field_type = field.get("type")
            if field_type in ["vector", "float_vector", "float16_vector"]:
                # Check if index type is IVF_FLAT
                if index_type == "IVF_FLAT":
                    # IVF_FLAT requires pgvector extension
                    # Test if pgvector extension is available
                    self._cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
                    if not self._cursor.fetchone():
                        raise Exception("pgvector extension is not installed. IVF_FLAT index requires pgvector extension.")
                    
                    nlist = params.get("nlist", 128)
                    # For pgvector extension, we need to specify the operator class
                    # based on the metric type
                    if metric_type == "L2":
                        opclass = "vector_l2_ops"
                    elif metric_type == "IP":
                        opclass = "vector_ip_ops"
                    elif metric_type == "COSINE":
                        opclass = "vector_cosine_ops"
                    else:
                        opclass = "vector_l2_ops"  # Default to L2
                    
                    self._cursor.execute(
                        f"""
                        CREATE INDEX {index_name}
                        ON {table_name} USING ivfflat ({field_name} {opclass})
                        WITH (lists = {nlist})
                        """
                    )
                    logger.info(f"Created ivfflat index for vector field {field_name} in collection {table_name}")
                elif index_type == "HNSW":
                    # HNSW requires pgvector extension
                    # Test if pgvector extension is available
                    self._cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
                    if not self._cursor.fetchone():
                        raise Exception("pgvector extension is not installed. HNSW index requires pgvector extension.")
                    
                    # Get HNSW parameters
                    m = params.get("M", 16)  # Number of connections per layer
                    ef_construction = params.get("efConstruction", 128)  # ef construction parameter
                    
                    # For pgvector extension, we need to specify the operator class
                    # based on the metric type
                    if metric_type == "L2":
                        opclass = "vector_l2_ops"
                    elif metric_type == "IP":
                        opclass = "vector_ip_ops"
                    elif metric_type == "COSINE":
                        opclass = "vector_cosine_ops"
                    else:
                        opclass = "vector_l2_ops"  # Default to L2
                    
                    self._cursor.execute(
                        f"""
                        CREATE INDEX {index_name}
                        ON {table_name} USING hnsw ({field_name} {opclass})
                        WITH (m = {m}, ef_construction = {ef_construction})
                        """
                    )
                    logger.info(f"Created HNSW index for vector field {field_name} in collection {table_name}")
                else:
                    # For other vector index types, just record the index
                    logger.info(f"Recording vector index for field {field_name} with type {index_type}")
                    raise Exception(f"Recording vector index for field {field_name} with type {index_type}")
            # elif field_type == "binary_vector":
                # For binary vectors, create B-tree index on bit varying type
                # try:
                #     self._cursor.execute(
                #         f"""
                #         CREATE INDEX {index_name}
                #         ON {table_name} ({field_name})
                #         """
                #     )
                #     logger.info(f"Created B-tree index for binary vector field {field_name} in collection {table_name}")
                # except Exception as e:
                #     logger.warning(f"Failed to create B-tree index for binary vector: {e}")
                #     logger.warning(f"Query performance for field {field_name} may be slow.")
            elif field_type == "sparse_float_vector":
                # For sparse vectors, create B-tree index on jsonb type
                try:
                    self._cursor.execute(
                        f"""
                        CREATE INDEX {index_name}
                        ON {table_name} USING gin ({field_name} jsonb_path_ops)
                        """
                    )
                    logger.info(f"Created GIN index for sparse vector field {field_name} in collection {table_name}")
                except Exception as e:
                    logger.warning(f"Failed to create GIN index for sparse vector: {e}")
                    logger.warning(f"Query performance for field {field_name} may be slow.")
            else:
                # For other vector types, raise exception
                raise Exception(f"Index is not supported for {field_type} vector type.")
        else:
            # Create B-tree index for non-vector fields in JSONB
            try:
                self._cursor.execute(
                    f"""
                    CREATE INDEX {index_name}
                    ON {table_name} ((data->>'{field_name}'))
                    """
                )
                logger.info(f"Created B-tree index for JSONB field {field_name} in collection {table_name}")
            except Exception as e:
                logger.warning(f"Failed to create B-tree index: {e}")
                logger.warning(f"Query performance for field {field_name} may be slow.")

        # Record index in udb_indexes table
        self._cursor.execute(
            """
            INSERT INTO udb_indexes (collection_name, field_name, index_name, index_type, params)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (collection_name, field_name, index_name, index_type, json.dumps(params))
        )
        
        # Refresh metric_type in udb_collection schema
        if is_vector_field and field and metric_type:
            # Get current schema from udb_collection
            self._cursor.execute(
                "SELECT schema FROM udb_collections WHERE name = %s",
                (collection_name,)
            )
            schema_json = self._cursor.fetchone()[0]
            if isinstance(schema_json, str):
                schema = json.loads(schema_json)
            
            # Update metric_type for the field
            for f in schema["fields"]:
                if f["name"] == field_name:
                    f["metric_type"] = metric_type
                    break
            
            # Save updated schema back to udb_collection
            self._cursor.execute(
                "UPDATE udb_collections SET schema = %s WHERE name = %s",
                (json.dumps(schema), collection_name)
            )
            
            # Update in-memory schema cache
            self._schemas[collection_name] = schema
            
            logger.info(f"Updated metric_type for field {field_name} to {metric_type} in collection {collection_name}")
        
        logger.info(f"Index created successfully for field {field_name} in collection {collection_name}")

    def load_collection(self, collection_name: str, *args, **kwargs):
        """Load collection and refresh schema
        
        Args:
            collection_name: Name of the collection to load
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        # Refresh schema for the collection
        if collection_name:
            self._load_schema(collection_name)
            logger.info(f"Collection {collection_name} loaded and schema refreshed")
        pass

    def release_collection(self, *args, **kwargs):
        """Release collection (not implemented)"""
        pass

    def describe_index(
        self,
        collection_name: str,
        index_name: str,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Dict:
        """Describe an index

        Args:
            collection_name (str): The name of the collection
            index_name (str): The name of the index to describe
            timeout (Optional[float]): Timeout for the operation in seconds (not used for PostgreSQL, included for compatibility)
            **kwargs: Additional keyword arguments for compatibility with MilvusClient

        Returns:
            Dict: A dictionary containing index information with the following keys:
                - index_name (str): The name of the index
                - field_name (str): The name of the field the index is on
                - index_type (str): The type of the index (e.g., "IVF_FLAT")
                - metric_type (str): The metric type (e.g., "L2", "IP", "COSINE")
                - params (dict): Additional index parameters
                - total_rows (int): Total number of rows in the collection
                - indexed_rows (int): Number of indexed rows
                - pending_index_rows (int): Number of pending index rows
                - state (str): Index state ("Finished", "InProgress", "Failed")

        Raises:
            Exception: If the index does not exist
        """
        # Check if collection exists
        if not self.has_collection(collection_name):
            raise Exception(f"Collection {collection_name} does not exist")

        # Query index information from udb_indexes table
        self._cursor.execute(
            """
            SELECT field_name, index_type, params
            FROM udb_indexes
            WHERE collection_name = %s AND index_name = %s
            """,
            (collection_name, index_name)
        )
        index_info = self._cursor.fetchone()

        if not index_info:
            raise Exception(f"Index {index_name} does not exist in collection {collection_name}")

        field_name, index_type, params_json = index_info
        if isinstance(params_json, dict):
            params = params_json
        elif isinstance(params_json, str):
            params = json.loads(params_json)
        else:
            params = {}

        # Get metric_type from params or default to "L2"
        metric_type = params.get("metric_type", "L2")

        # Get total rows in the collection
        self._cursor.execute(
            f"SELECT COUNT(*) FROM {collection_name}"
        )
        total_rows = self._cursor.fetchone()[0]

        # For PostgreSQL, all rows are indexed immediately
        indexed_rows = total_rows
        pending_index_rows = 0
        state = "Finished"

        return {
            "index_name": index_name,
            "field_name": field_name,
            "index_type": index_type,
            "metric_type": metric_type,
            "params": params,
            "total_rows": total_rows,
            "indexed_rows": indexed_rows,
            "pending_index_rows": pending_index_rows,
            "state": state
        }
