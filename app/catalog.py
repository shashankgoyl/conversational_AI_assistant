import json
import os
import logging
from typing import List, Dict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)
CATALOG_PATH = os.getenv("CATALOG_PATH", "data/shl_catalog.json")


def _item_to_text(item: Dict) -> str:
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
        self.catalog: List[Dict] = []
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=5000,
        )
        self._load()

    def _load(self):
        catalog_path = Path(CATALOG_PATH)
        if not catalog_path.exists():
            raise FileNotFoundError(f"Catalog not found at {catalog_path}")
        with open(catalog_path) as f:
            self.catalog = json.load(f)
        texts = [_item_to_text(item) for item in self.catalog]
        self._matrix = self.vectorizer.fit_transform(texts)
        logger.info(f"TF-IDF index built with {len(self.catalog)} items.")

    def search(self, query: str, k: int = 20) -> List[Dict]:
        if not query.strip():
            return self.catalog[:k]
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.catalog[i] for i in top_indices if scores[i] > 0]

    def get_all(self) -> List[Dict]:
        return self.catalog