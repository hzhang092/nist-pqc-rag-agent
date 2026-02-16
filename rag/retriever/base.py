from dataclasses import dataclass
from typing import Protocol, List

@dataclass
class ChunkHit:
    score: float
    chunk_id: str
    doc_id: str
    start_page: int
    end_page: int
    text: str

class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> List[ChunkHit]:
        ...
