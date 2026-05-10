#!/usr/bin/env python3
"""
Pre-build the FAISS index from shl_catalog.json.
Run this once before deploying (or in the Dockerfile) to avoid cold-start delay.

Usage:
    python scripts/build_index.py
"""

import sys
import os

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.catalog import CatalogSearch

if __name__ == "__main__":
    print("Building FAISS index…")
    cs = CatalogSearch()
    print(f"Done. Index contains {len(cs.catalog)} catalogue items.")
    print("Files written:")
    print("  data/faiss.index")
    print("  data/catalog_meta.json")
