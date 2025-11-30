import time
import json
import joblib
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text

# --- CONFIGURATION ---
DB_USER = "postgres"
DB_PASS = "password"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "trading_db"
MODEL_PATH = "hybrid_model.pkl"
SCALER_PATH = "scaler.pkl"

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# --- GLOBAL STATE ---
ml_models = {}

# --- LIFESPAN MANAGER (Loads Model on Startup) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML Artifacts
    try:
        ml_models["model"] = joblib.load(MODEL_PATH)
        ml_models["scaler"] = joblib.load(SCALER_PATH)
        print(f"✅ ML Model & Scaler loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Warning: Could not load ML model: {e}")
        ml_models["model"] = None
        ml_models["scaler"] = None
    
    yield
    
    # Clean up (if needed)
    ml_models.clear()

# --- APP SETUP ---
app = FastAPI(title="AlphaTrade AI API", version="1.0", lifespan=lifespan)

# Enable CORS (Allows your frontend/React/HTML to talk to this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODELS ---
class PredictionResponse(BaseModel):
    timestamp: str
    price: float
    signal: str
    confidence: float
    rule_score: float
    ml_probability: float
    features: dict

# --- HELPER FUNCTIONS ---
def get_latest_features():
    query = """
    SELECT * FROM market_features 
    ORDER BY timestamp DESC 
    LIMIT 1
    """
    try:
        # specific for pandas reading sql
        df = pd.read_sql(query, engine)
        if df.empty:
            return None
        return df.iloc[0]
    except Exception as e:
        print(f"DB Error: {e}")
        return None

def calculate_rule_score(row):
    bull_score = 0
    bear_score = 0
    reasons = []

    # Re-using logic from Phase 3
    if row['pcr_oi'] > 1.2: 
        bull_score += 1.5
        reasons.append("High PCR")
    elif row['pcr_oi'] < 0.6: 
        bear_score += 1.5
        reasons.append("Low PCR")
    
    if row['delta_pcr_1m'] > 0.02: bull_score += 1.0
    elif row['delta_pcr_1m'] < -0.02: bear_score += 1.0

    if row['delta_price_5m'] > 0 and row['delta_oi_ce_5m'] < 0: 
        bull_score += 2.0
        reasons.append("Short Covering")
    
    if row['delta_price_5m'] < 0 and row['delta_oi_pe_5m'] < 0: 
        bear_score += 2.0
        reasons.append("Long Unwinding")

    if row['vix'] < 13 and row['delta_price_5m'] > 0: bull_score += 0.5
    if row['vix'] > 20 and row['delta_price_5m'] < 0: bear_score += 1.0

    return bull_score, bear_score, reasons

def get_ml_prediction(row):
    model = ml_models.get("model")
    scaler = ml_models.get("scaler")
    
    if not model or not scaler:
        return 0.5  # Neutral if no model

    feature_cols = [
        'pcr_oi', 'pcr_vol', 'atm_iv', 'vix',
        'delta_oi_ce_5m', 'delta_oi_pe_5m', 
        'delta_pcr_1m', 'delta_price_5m', 
        'delta_oi_ce_15m', 'delta_oi_pe_15m', 'delta_price_15m'
    ]
    
    # Extract features in correct order, handle missing cols with 0
    data = []
    for col in feature_cols:
        val = row.get(col, 0)
        data.append(val)
    
    X = np.array([data]).reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    # Predict Probability of Class 1 (Buy)
    prob = model.predict_proba(X_scaled)[0][1]
    return prob

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "running", "service": "AlphaTrade AI"}

@app.get("/predict/latest", response_model=PredictionResponse)
def predict_latest():
    """
    Main Endpoint: Fetches DB data -> Runs Rules -> Runs ML -> Returns Signal
    """
    row = get_latest_features()
    
    if row is None:
        raise HTTPException(status_code=503, detail="Market data unavailable")

    # 1. Rule Engine
    bull_score, bear_score, reasons = calculate_rule_score(row)
    rule_net = bull_score - bear_score
    
    # 2. ML Engine
    ml_prob = get_ml_prediction(row)  # 0.0 to 1.0
    
    # 3. Hybrid Consensus Logic
    final_signal = "NEUTRAL"
    final_conf = 0.0
    
    # Logic: ML is the filter, Rules are the trigger
    if rule_net > 1.5 and ml_prob > 0.60:
        final_signal = "BUY"
        final_conf = (ml_prob + 0.1) # Boost confidence
    elif rule_net < -1.5 and ml_prob < 0.40:
        final_signal = "SELL"
        final_conf = (1.0 - ml_prob + 0.1)
    else:
        # Weak signals
        if ml_prob > 0.75: final_signal = "WEAK_BUY"
        elif ml_prob < 0.25: final_signal = "WEAK_SELL"
        final_conf = abs(ml_prob - 0.5) * 2

    # formatting timestamp for JSON
    ts_str = str(row['timestamp'])

    return {
        "timestamp": ts_str,
        "price": row['price'],
        "signal": final_signal,
        "confidence": round(min(final_conf, 0.99), 2),
        "rule_score": round(rule_net, 2),
        "ml_probability": round(ml_prob, 2),
        "features": {
            "pcr": round(row['pcr_oi'], 2),
            "vix": round(row['vix'], 2),
            "reasons": reasons
        }
    }


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for testing (simplest)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)