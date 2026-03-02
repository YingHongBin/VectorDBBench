from typing import TypedDict

from pydantic import BaseModel

from ..api import DBCaseConfig, DBConfig

class HAKESSearchConfigDict(TypedDict):
    host: str
    port: int

class HAKESSearchConfig(DBConfig):
    host: str = 'localhost'
    port: int = 8080

    def to_dict(self) -> HAKESSearchConfigDict:
        return {
            "host": self.host,
            "port": self.port,
        }
    
class HAKESSearchCaseConfig(DBCaseConfig, BaseModel):
    k: int = 100
    nprobe: int = 10
    k_factor: int = 1
    metric_type: str = "L2"

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {
            "k": self.k,
            "nprobe": self.nprobe,
            "k_factor": self.k_factor,
            "metric_type": self.metric_type,
        }