import logging
from typing import Dict, List

from pymilvus.milvus_client.index import IndexParams

logger = logging.getLogger(__name__)


class BaseMilvusClient:
    def __init__(self, **kwargs):
        pass

    @classmethod
    def prepare_index_params(cls, field_name: str = "", **kwargs) -> IndexParams:
        index_params = IndexParams()
        if field_name:
            index_params.add_index(field_name, **kwargs)
        return index_params
