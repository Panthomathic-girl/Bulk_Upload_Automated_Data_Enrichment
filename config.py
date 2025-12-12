# config.py
from pathlib import Path
from dotenv import load_dotenv
import os
from typing import Literal


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

class Settings:
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    PORT_NO: int = int(os.getenv("PORT_NO", "8000"))
    

    def __post_init__(self):
        if not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY required in .env")

class ModelConfig:
    LEAD_MODEL_FILE = "app/predictive_lead_score/models/deal_closure_model.pkl"
    ORDER_FORECAST_MODEL_FILE = Path("app/order_forecasting/models/order_forecast_model.pkl") 
    ENRICHMENT_FAISS_INDEX = Path("app/bulk_upload/models/faiss_leads_index.faiss")
    ENRICHMENT_META_FILE = Path("app/bulk_upload/models/faiss_leads_meta.pkl")
    ENRICHMENT_ID_TO_IDX_PATH = Path("app/bulk_upload/models/faiss_id_to_idx.pkl")
    ENRICHMENT_SENTENCE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
class SmartTaskConfig:
    TASKTYPE = Literal["call", "email", "meeting", "administrative", "follow up"]
    PriorityType = Literal["low", "medium", "high", "urgent"]
    StatusType = Literal["not started", "in progress", "completed", "waiting", "deferred"]
    
settings = Settings()



    # @classmethod
    # def delete_by_lead_id(cls, lead_id: int) -> bool:
    #     cls._ensure_index()
    #     if lead_id not in cls._id_to_idx:
    #         return False

    #     # Remove from FAISS
    #     cls._index.remove_ids(np.array([lead_id], dtype=np.int64))

    #     # Remove from metadata
    #     idx_to_remove = cls._id_to_idx.pop(lead_id)
    #     del cls._meta[idx_to_remove]

    #     # Rebuild index to keep everything consistent
    #     cls._rebuild_index()

    #     logger.info(f"Deleted record lead_id={lead_id}")
    #     return True