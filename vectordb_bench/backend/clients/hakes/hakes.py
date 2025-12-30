from contextlib import contextmanager
from ..api import VectorDB, DBCaseConfig
from .config import HAKESConfigDict, HAKESCaseConfig

import numpy as np
from hakesclient.components.searcher import Searcher

class HakesClient(VectorDB):

    def __init__(
        self,
        dim: int,
        db_config: HAKESConfigDict,
        db_case_config: HAKESCaseConfig | None,
        collection_name: str = "poc",
        drop_old: bool = False,
        **kwargs,
    ) -> None:
        self.name = 'HAKES'
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name
        self.dim = dim
        self.search_param = self.case_config.search_param()

        host = 'http://' + self.db_config.get('host') + ':' + str(self.db_config.get('port'))
        self.searcher = Searcher([host])
        self.searcher.load_collection(self.collection_name)

        if drop_old:
            pass

    @contextmanager
    def init(self) -> None:
        yield

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        try:
            vecs = np.array(embeddings, dtype=np.float32)
            ids = np.array(metadata, dtype=np.int64)
            for i in range(len(vecs)):
                vec = vecs[i].reshape(1, -1)
                id = ids[i]
                response = self.searcher.add(self.collection_name, vec, id)
                if response is None:
                    raise Exception("HAKES add vector failed")
            # self.searcher.add(self.collection_name, vecs, ids)
            return len(embeddings), None
        except Exception as e:
            return 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
    ) -> list[int]:
        query = np.array([query], dtype=np.float32)
        result = self.searcher.search(
            self.collection_name,
            query,
            self.search_param['k'],
            self.search_param['nprobe'],
            self.search_param['k_factor'],
            # TODO: due to unknown reason, search_param['metric_type'] always be COSINE, which is not supported in HAKES, so hardcode to run and fix this issue later
            'IP'
            # self.search_param['metric_type']
        )
        if result is None:
            return None
        result = self.searcher.rerank(
            self.collection_name,
            query,
            self.search_param['k'],
            np.array(result["ids"]),
            'IP'
            # self.search_param['metric_type'],
        )
        return result['ids'][0]

    def optimize(self, data_size: int | None = None):
        pass
    
    def need_normalize_cosine(self) -> bool:
        """Wheather this database need to normalize dataset to support COSINE"""
        print("HAKES need to normalize dataset to support COSINE")
        return True