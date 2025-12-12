# app/data_enrichment_faiss/views.py
from fastapi import APIRouter, HTTPException, Body, Depends
from fastapi.responses import JSONResponse
import logging
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import asyncio
import traceback
import numpy as np

from .rag_faiss import LeadFAISSIndex

router = APIRouter(tags=["Automated Data Enrichment"])

executor = ThreadPoolExecutor(max_workers=10)
BATCH_SIZE = 100

logger = logging.getLogger(__name__)


def _search_batch_leads(batch: List[Dict], total_indexed: int) -> List[Dict]:
    try:
        LeadFAISSIndex._ensure_index()
        if LeadFAISSIndex._index.ntotal == 0:
            return [{"input_lead": lead, "duplicates_found": 0, "duplicates": []} for lead in batch]

        model = LeadFAISSIndex._load_model()
        query_texts = []
        valid_indices = []
        for i, lead in enumerate(batch):
            text = LeadFAISSIndex._record_to_text(lead)
            if text.strip():
                query_texts.append(text)
                valid_indices.append(i)

        if not query_texts:
            return [{"input_lead": lead, "duplicates_found": 0, "duplicates": []} for lead in batch]

        q_vecs = model.encode(query_texts, normalize_embeddings=True).astype(np.float32)

        top_k = total_indexed + 50
        threshold = 0.75

        scores, ids = LeadFAISSIndex._index.search(q_vecs, top_k + 20)

        batch_results = [None] * len(batch)
        for j, i in enumerate(valid_indices):
            lead_duplicates = []
            for score, idx in zip(scores[j], ids[j]):
                if idx == -1 or score < threshold:
                    continue
                record = LeadFAISSIndex.get_by_lead_id(int(idx))
                if record:
                    rec = record.copy()
                    rec["similarity_score"] = round(float(score) * 100, 2)
                    lead_duplicates.append(rec)

            batch_results[i] = {
                "input_lead": batch[i],
                "duplicates_found": len(lead_duplicates),
                "duplicates": lead_duplicates
            }

        for k in range(len(batch)):
            if batch_results[k] is None:
                batch_results[k] = {
                    "input_lead": batch[k],
                    "duplicates_found": 0,
                    "duplicates": []
                }

        return batch_results

    except Exception as e:
        logger.error(f"Error in batch similarity search: {str(e)}\n{traceback.format_exc()}")
        return [{
            "input_lead": lead,
            "duplicates_found": 0,
            "duplicates": [],
            "error": f"Search failed: {str(e)}"
        } for lead in batch]


# ===================== CRUD ENDPOINTS =====================

@router.post("/records")
async def add_records_batch(records: List[Dict]):
    """Add new records (skip if lead_id exists)"""
    count = LeadFAISSIndex.add_records(records)
    return {"status": "success", "added": count}


@router.put("/records")
async def upsert_records(
    records: Optional[List[Dict]] = Body(None),
    record: Optional[Dict] = Body(None)
):
    if records is None and record is None:
        raise HTTPException(status_code=400, detail="No data provided. Send 'records' list or 'record' object.")

    if record is not None:
        if not isinstance(record, dict) or "lead_id" not in record:
            raise HTTPException(status_code=400, detail="Single 'record' must be a dict with 'lead_id'")
        data_to_update = [record]
    else:
        if not isinstance(records, list):
            raise HTTPException(status_code=400, detail="'records' must be a list of dictionaries")
        data_to_update = records

    if not data_to_update:
        raise HTTPException(status_code=400, detail="No valid records to update")

    result = LeadFAISSIndex.upsert_records(data_to_update)
    return {
        "status": "success",
        "message": "Update completed",
        **result
    }


@router.get("/records")
async def fetch_all_records():
    records = LeadFAISSIndex.get_all_records()
    return {"count": len(records), "records": records}


@router.get("/records/{lead_id}")
async def fetch_by_lead_id(lead_id: int):
    record = LeadFAISSIndex.get_by_lead_id(lead_id)
    if not record:
        raise HTTPException(404, f"Record with lead_id={lead_id} not found")
    return {"record": record}


@router.delete("/records/{lead_id}")
async def delete_by_lead_id(lead_id: int):
    success = LeadFAISSIndex.delete_by_lead_id(lead_id)
    if not success:
        raise HTTPException(404, f"Record with lead_id={lead_id} not found")
    return {"status": "deleted", "lead_id": lead_id}


@router.delete("/records")
async def delete_all_records():
    LeadFAISSIndex.delete_all()
    return {"status": "all records deleted"}


# ===================== DEDUPLICATION =====================

@router.post("/duplicates")
async def process_deduplication(leads: List[Dict] = Body(...)):
    try:
        if not isinstance(leads, list):
            raise HTTPException(status_code=400, detail="Request body must be a list of leads")

        if not leads:
            raise HTTPException(status_code=400, detail="leads list is required")

        try:
            total_indexed = len(LeadFAISSIndex.get_all_records())
        except Exception as e:
            logger.error(f"Failed to get indexed count: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to access master records index")

        if total_indexed == 0:
            raise HTTPException(status_code=400, detail="No master records indexed yet. Use /records/batch first.")

        all_results = []
        total_leads = len(leads)
        total_batches = (total_leads + BATCH_SIZE - 1) // BATCH_SIZE

        loop = asyncio.get_event_loop()
        tasks = []
        for batch_idx in range(0, total_leads, BATCH_SIZE):
            batch = leads[batch_idx:batch_idx + BATCH_SIZE]
            tasks.append(loop.run_in_executor(executor, _search_batch_leads, batch, total_indexed))

        batch_results_list = await asyncio.gather(*tasks)

        for batch_res in batch_results_list:
            all_results.extend(batch_res)

        return {
            "status": "completed",
            "indexed_master_records": total_indexed,
            "leads_processed": len(leads),
            "total_batches": total_batches,
            "batch_size": BATCH_SIZE,
            "similarity_threshold": 0.75,
            "results": all_results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_deduplication: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error during deduplication")