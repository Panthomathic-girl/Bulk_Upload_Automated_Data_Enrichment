# app/data_enrichment_faiss/rag_faiss.py
import logging
from pathlib import Path
from typing import List, Dict, Optional

import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_PATH = Path("faiss_leads_index.faiss")
META_PATH = Path("faiss_leads_meta.pkl")


class LeadFAISSIndex:
    _model = None
    _index = None
    _meta: List[Dict] = None
    _id_to_idx: Dict[int, int] = None  # lead_id → position in _meta

    @classmethod
    def _load_model(cls):
        if cls._model is None:
            cls._model = SentenceTransformer(MODEL_NAME)
        return cls._model

    @classmethod
    def _record_to_text(cls, record: Dict) -> str:
        """Convert a lead/record dict to a meaningful searchable string - FULLY DYNAMIC"""
        parts = []
        for key, value in record.items():
            if key == "lead_id":        # Skip the primary key
                continue
            if value is None or value == "":
                continue
            # Convert numbers, booleans etc. to string
            parts.append(str(value).strip())
        return " | ".join(parts)

    @classmethod
    def _ensure_index(cls):
        if cls._index is not None:
            return

        if INDEX_PATH.exists() and META_PATH.exists():
            cls.load_index()
        else:
            dim = 384
            base_index = faiss.IndexFlatIP(dim)
            cls._index = faiss.IndexIDMap(base_index)
            cls._meta = []
            cls._id_to_idx = {}

    @classmethod
    def load_index(cls):
        if cls._index is not None:
            return

        cls._index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "rb") as f:
            cls._meta = pickle.load(f)
        cls._id_to_idx = {record["lead_id"]: idx for idx, record in enumerate(cls._meta)}
        logger.info(f"Loaded FAISS index with {cls._index.ntotal} records")

    @classmethod
    def _save_index(cls):
        faiss.write_index(cls._index, str(INDEX_PATH))
        with open(META_PATH, "wb") as f:
            pickle.dump(cls._meta, f)
        logger.info(f"Saved FAISS index with {len(cls._meta)} records")

    @classmethod
    def _rebuild_index(cls):
        if not cls._meta:
            cls._index = None
            cls._id_to_idx = {}
            for path in [INDEX_PATH, META_PATH]:
                if path.exists():
                    path.unlink()
            return

        model = cls._load_model()
        dim = 384
        base_index = faiss.IndexFlatIP(dim)
        new_index = faiss.IndexIDMap(base_index)

        vectors = []
        ids = []
        new_meta = []
        new_id_to_idx = {}

        for record in cls._meta:
            text = cls._record_to_text(record)
            if not text.strip():
                continue
            vec = model.encode(text, normalize_embeddings=True).astype(np.float32)
            lead_id = record["lead_id"]

            vectors.append(vec)
            ids.append(lead_id)
            new_meta.append(record)
            new_id_to_idx[lead_id] = len(new_meta) - 1

        if vectors:
            vectors_np = np.array(vectors, dtype=np.float32)
            ids_np = np.array(ids, dtype=np.int64)
            new_index.add_with_ids(vectors_np, ids_np)

        cls._index = new_index
        cls._meta = new_meta
        cls._id_to_idx = new_id_to_idx
        cls._save_index()

    # ===================== CRUD OPERATIONS =====================

    @classmethod
    def add_records(cls, records: List[Dict]) -> int:
        cls._ensure_index()
        model = cls._load_model()

        new_vectors = []
        new_ids = []
        added = 0

        for record in records:
            lead_id = record.get("lead_id")
            if lead_id is None or lead_id in cls._id_to_idx:
                continue

            text = cls._record_to_text(record)
            if not text.strip():
                continue

            vec = model.encode(text, normalize_embeddings=True).astype(np.float32)
            new_vectors.append(vec)
            new_ids.append(lead_id)
            cls._meta.append(record.copy())
            cls._id_to_idx[lead_id] = len(cls._meta) - 1
            added += 1

        if new_vectors:
            vectors_np = np.array(new_vectors, dtype=np.float32)
            ids_np = np.array(new_ids, dtype=np.int64)
            cls._index.add_with_ids(vectors_np, ids_np)
            cls._save_index()

        logger.info(f"Added {added} new records")
        return added

    @classmethod
    def upsert_records(cls, records: List[Dict]) -> Dict[str, int]:
        cls._ensure_index()
        model = cls._load_model()

        updated = 0
        skipped = 0

        to_update = {}
        for record in records:
            lead_id = record.get("lead_id")
            if lead_id is None:
                skipped += 1
                continue
            if lead_id not in cls._id_to_idx:
                skipped += 1
                continue

            text = cls._record_to_text(record)
            if not text.strip():
                skipped += 1
                continue

            vec = model.encode(text, normalize_embeddings=True).astype(np.float32)
            to_update[lead_id] = (vec, record.copy())

        if not to_update:
            logger.info(f"Upsert (update-only): No existing records to update. Skipped: {skipped}")
            return {"added": 0, "updated": 0, "skipped": skipped}

        ids_to_remove = list(to_update.keys())
        cls._index.remove_ids(np.array(ids_to_remove, dtype=np.int64))

        new_vectors = []
        new_ids = []
        for lead_id, (vec, record) in to_update.items():
            idx = cls._id_to_idx[lead_id]
            cls._meta[idx] = record
            new_vectors.append(vec)
            new_ids.append(lead_id)
            updated += 1

        if new_vectors:
            vectors_np = np.array(new_vectors, dtype=np.float32)
            ids_np = np.array(new_ids, dtype=np.int64)
            cls._index.add_with_ids(vectors_np, ids_np)
            cls._save_index()

        logger.info(f"Upsert (update-only): {updated} updated, {skipped} skipped (not found or invalid)")
        return {"updated": updated, "skipped": skipped}

    @classmethod
    def get_all_records(cls) -> List[Dict]:
        cls._ensure_index()
        return [record.copy() for record in cls._meta]

    @classmethod
    def get_by_lead_id(cls, lead_id: int) -> Optional[Dict]:
        cls._ensure_index()
        if lead_id not in cls._id_to_idx:
            return None
        idx = cls._id_to_idx[lead_id]
        return cls._meta[idx].copy()

    @classmethod
    def delete_by_lead_id(cls, lead_id: int) -> bool:
        cls._ensure_index()
        if lead_id not in cls._id_to_idx:
            return False

        cls._index.remove_ids(np.array([lead_id], dtype=np.int64))
        idx_to_remove = cls._id_to_idx.pop(lead_id)
        del cls._meta[idx_to_remove]
        cls._rebuild_index()
        logger.info(f"Deleted record lead_id={lead_id}")
        return True

    @classmethod
    def delete_all(cls):
        cls._index = None
        cls._meta = None
        cls._id_to_idx = None
        for path in [INDEX_PATH, META_PATH]:
            if path.exists():
                path.unlink()
        logger.info("All records and index deleted")

    @classmethod
    def search_similar(cls, query_record: Dict, top_k: int = 5, threshold: float = 0.7) -> List[Dict]:
        cls._ensure_index()
        if cls._index.ntotal == 0:
            return []

        model = cls._load_model()
        query_text = cls._record_to_text(query_record)
        if not query_text.strip():
            return []

        q_vec = model.encode(query_text, normalize_embeddings=True).astype(np.float32)[np.newaxis]
        scores, ids = cls._index.search(q_vec, top_k + 20)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1 or score < threshold:
                continue
            record = cls.get_by_lead_id(int(idx))
            if record:
                rec = record.copy()
                rec["similarity_score"] = round(float(score) * 100, 2)
                results.append(rec)
            if len(results) >= top_k:
                break
        return results