import os
import logging
from typing import List, Dict
from pathlib import Path
import json

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

logger = logging.getLogger(__name__)

CATALOG_PATH = os.getenv("CATALOG_PATH", "data/shl_catalog.json")
INDEX_PATH = os.getenv("INDEX_PATH", "data/faiss.index")
META_PATH = os.getenv("META_PATH", "data/catalog_meta.json")
EMBED_MODEL = "paraphrase-MiniLM-L3-v2"


def _item_to_text(item: Dict) -> str:
    """Create a rich text representation for embedding."""
    parts = [
        item.get("name", ""),
        f"Type: {item.get('test_type', '')}",
        f"Categories: {', '.join(item.get('keys', []))}",
        item.get("description", ""),
        f"Use cases: {', '.join(item.get('use_cases', []))}",
        f"Level: {', '.join(item.get('level', []))}",
        f"Domain: {', '.join(item.get('domain', []))}",
    ]
    return " | ".join(p for p in parts if p.strip())


class CatalogSearch:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.catalog: List[Dict] = []
        self.index: faiss.IndexFlatIP = None
        self._load()

    def _load(self):
        """Load catalog and build or restore FAISS index."""
        catalog_path = Path(CATALOG_PATH)
        if not catalog_path.exists():
            raise FileNotFoundError(f"Catalog not found at {catalog_path}")

        with open(catalog_path) as f:
            self.catalog = json.load(f)

        index_path = Path(INDEX_PATH)
        meta_path = Path(META_PATH)

        if index_path.exists() and meta_path.exists():
            logger.info("Loading existing FAISS index from disk…")
            self.index = faiss.read_index(str(index_path))
            with open(meta_path) as f:
                stored = json.load(f)
            # Sanity check: rebuild if catalog changed
            if len(stored) != len(self.catalog):
                logger.warning("Catalog size mismatch — rebuilding index.")
                self._build_index()
        else:
            logger.info("Building FAISS index from catalog…")
            self._build_index()

    def _build_index(self):
        texts = [_item_to_text(item) for item in self.catalog]
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine sim after L2-norm
        self.index.add(embeddings)

        # Persist
        os.makedirs(os.path.dirname(INDEX_PATH) or ".", exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "w") as f:
            json.dump(self.catalog, f)
        logger.info(f"FAISS index built with {len(self.catalog)} items.")

    def search(self, query: str, k: int = 15) -> List[Dict]:
        """Return top-k catalog items most similar to the query."""
        if not query.strip():
            return self.catalog[:k]

        q_emb = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        q_emb = q_emb.astype(np.float32)
        k = min(k, len(self.catalog))
        _, indices = self.index.search(q_emb, k)
        return [self.catalog[i] for i in indices[0] if i < len(self.catalog)]

    def get_all(self) -> List[Dict]:
        return self.catalog
