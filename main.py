from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from app.database import get_db, engine, Base
from app.models import MarketFeatures
from app.services.ml import ml_engine
from pydantic import BaseModel
from typing import Optional

# Create tables on startup
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"Warning: Database connection failed. Tables not created. Error: {e}")

app = FastAPI(title="AlphaTrade AI API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    timestamp: str
    price: float
    signal: str
    confidence: float
    rule_score: float
    ml_probability: float
    features: dict

def calculate_rule_score(row: MarketFeatures):
    bull_score = 0
    bear_score = 0
    reasons = []

    if row.pcr_oi > 1.2: 
        bull_score += 1.5
        reasons.append("High PCR")
    elif row.pcr_oi < 0.6: 
        bear_score += 1.5
        reasons.append("Low PCR")
    
    if row.delta_pcr_1m > 0.02: bull_score += 1.0
    elif row.delta_pcr_1m < -0.02: bear_score += 1.0

    if row.delta_price_5m > 0 and row.delta_oi_ce_5m < 0: 
        bull_score += 2.0
        reasons.append("Short Covering")
    
    if row.delta_price_5m < 0 and row.delta_oi_pe_5m < 0: 
        bear_score += 2.0
        reasons.append("Long Unwinding")

    if row.vix < 13 and row.delta_price_5m > 0: bull_score += 0.5
    if row.vix > 20 and row.delta_price_5m < 0: bear_score += 1.0

    return bull_score, bear_score, reasons

from fastapi.responses import FileResponse
import os

@app.get("/")
def read_root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "templates/index.html"))

@app.get("/health")
def health_check():
    return {"status": "running", "service": "AlphaTrade AI v2"}

@app.get("/predict/latest", response_model=PredictionResponse)
def predict_latest(db: Session = Depends(get_db)):
    """
    Main Endpoint: Fetches DB data -> Runs Rules -> Runs ML -> Returns Signal
    """
    try:
        row = db.query(MarketFeatures).order_by(MarketFeatures.timestamp.desc()).first()
    except Exception as e:
        print(f"DB Error: {e}")
        row = None
    
    if row is None:
        # If DB is empty (start of day), return neutral/waiting state
        return {
            "timestamp": "Waiting for Data...",
            "price": 0.0,
            "signal": "INITIALIZING",
            "confidence": 0.0,
            "rule_score": 0.0,
            "ml_probability": 0.0,
            "features": {
                "pcr": 0.0,
                "vix": 0.0,
                "reasons": ["System Initializing..."]
            }
        }

    # 1. Rule Engine
    bull_score, bear_score, reasons = calculate_rule_score(row)
    rule_net = bull_score - bear_score
    
    # 2. ML Engine
    features_dict = {
        'pcr_oi': row.pcr_oi,
        'pcr_vol': row.pcr_vol,
        'atm_iv': row.atm_iv,
        'vix': row.vix,
        'delta_oi_ce_5m': row.delta_oi_ce_5m,
        'delta_oi_pe_5m': row.delta_oi_pe_5m,
        'delta_pcr_1m': row.delta_pcr_1m,
        'delta_price_5m': row.delta_price_5m,
        'delta_oi_ce_15m': row.delta_oi_ce_15m,
        'delta_oi_pe_15m': row.delta_oi_pe_15m,
        'delta_price_15m': row.delta_price_15m
    }
    
    ml_prob = ml_engine.predict(features_dict)
    
    # 3. Hybrid Consensus Logic
    final_signal = "NEUTRAL"
    final_conf = 0.0
    
    if rule_net > 1.5 and ml_prob > 0.60:
        final_signal = "BUY"
        final_conf = (ml_prob + 0.1)
    elif rule_net < -1.5 and ml_prob < 0.40:
        final_signal = "SELL"
        final_conf = (1.0 - ml_prob + 0.1)
    else:
        if ml_prob > 0.75: final_signal = "WEAK_BUY"
        elif ml_prob < 0.25: final_signal = "WEAK_SELL"
        final_conf = abs(ml_prob - 0.5) * 2

    return {
        "timestamp": str(row.timestamp),
        "price": row.price,
        "signal": final_signal,
        "confidence": round(min(final_conf, 0.99), 2),
        "rule_score": round(rule_net, 2),
        "ml_probability": round(ml_prob, 2),
        "features": {
            "pcr": round(row.pcr_oi, 2),
            "vix": round(row.vix, 2),
            "reasons": reasons
        }
    }
