from contextlib import contextmanager
from ..api import VectorDB, DBCaseConfig
from .config import HAKESSearchConfigDict, HAKESSearchCaseConfig

import numpy as np
from hakes_client import HakesClient, ClientConfig

class HakesSearchClient(VectorDB):

    def __init__(
        self,
        dim: int,
        db_config: HAKESSearchConfigDict,
        db_case_config: HAKESSearchCaseConfig | None,
        collection_name: str = "poc",
        drop_old: bool = False,
        **kwargs,
    ) -> None:
        self.name = 'HAKESSEARCH'
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name
        self.dim = dim
        self.search_param = self.case_config.search_param()

        host = 'http://' + self.db_config.get('host') + ':' + str(self.db_config.get('port'))
        cfg = ClientConfig([host], [host])
        self.searcher = HakesClient(cfg)

        if drop_old:
            pass

    def __getstate__(self):
        """Control pickling: exclude unpicklable HakesClient instance."""
        state = self.__dict__.copy()
        # Remove unpicklable HakesClient which contains SimpleQueue
        state['searcher'] = None
        return state

    def __setstate__(self, state):
        """Control unpickling: recreate HakesClient in the new process."""
        self.__dict__.update(state)
        # Recreate the HakesClient connection in the child process
        host = 'http://' + self.db_config.get('host') + ':' + str(self.db_config.get('port'))
        cfg = ClientConfig([[host]], [host])
        self.searcher = HakesClient(cfg)

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
            response = self.searcher.add(len(vecs), 768, vecs, ids)
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
            query,
            self.search_param['k'],
            self.search_param['nprobe'],
            self.search_param['k_factor'],
            # TODO: due to unknown reason, search_param['metric_type'] always be COSINE, which is not supported in HAKES, so hardcode to run and fix this issue later
            'IP'
            # self.search_param['metric_type']
        )
        ids = result[0]['ids']
        if ids is None or len(ids) == 0:
            return None
        return ids[0]

    def optimize(self, data_size: int | None = None):
        pass
    
    def need_normalize_cosine(self) -> bool:
        """Wheather this database need to normalize dataset to support COSINE"""
        print("HAKES need to normalize dataset to support COSINE")
        return True