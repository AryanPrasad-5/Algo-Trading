import joblib
import numpy as np
import os
from app.config import settings

class MLEngine:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_models()

    def load_models(self):
        try:
            if os.path.exists(settings.MODEL_PATH) and os.path.exists(settings.SCALER_PATH):
                self.model = joblib.load(settings.MODEL_PATH)
                self.scaler = joblib.load(settings.SCALER_PATH)
                print(f"✅ ML Model & Scaler loaded from {settings.MODEL_PATH}")
            else:
                print(f"⚠️ Warning: ML model files not found. Running in Rules-Only mode.")
        except Exception as e:
            print(f"⚠️ Warning: Could not load ML model: {e}")
            self.model = None
            self.scaler = None

    def predict(self, features: dict) -> float:
        """
        Returns probability of Buy (0.0 to 1.0). Returns 0.5 if no model.
        """
        if not self.model or not self.scaler:
            return 0.5

        feature_cols = [
            'pcr_oi', 'pcr_vol', 'atm_iv', 'vix',
            'delta_oi_ce_5m', 'delta_oi_pe_5m', 
            'delta_pcr_1m', 'delta_price_5m', 
            'delta_oi_ce_15m', 'delta_oi_pe_15m', 'delta_price_15m'
        ]
        
        data = []
        for col in feature_cols:
            val = features.get(col, 0)
            data.append(val)
        
        X = np.array([data]).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        prob = self.model.predict_proba(X_scaled)[0][1]
        return prob

ml_engine = MLEngine()
