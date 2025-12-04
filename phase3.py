import time
import json
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
from pytz import timezone

DB_USER = "postgres"
DB_PASS = "password"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "trading_db"

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS signal_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    signal VARCHAR(10),
    confidence NUMERIC,
    bull_score NUMERIC,
    bear_score NUMERIC,
    active_rules JSONB,
    price NUMERIC,
    pcr NUMERIC
);
"""

def init_db():
    with engine.connect() as conn:
        conn.execute(text(SCHEMA_SQL))
        conn.commit()

class RuleEngine:
    def __init__(self):
        self.engine = engine

    def get_latest_features(self):
        query = """
        SELECT * FROM market_features 
        ORDER BY timestamp DESC 
        LIMIT 1
        """
        return pd.read_sql(query, self.engine)

    def get_last_signal_ts(self):
        query = "SELECT MAX(timestamp) FROM signal_logs"
        with self.engine.connect() as conn:
            result = conn.execute(text(query)).scalar()
        return result

    def evaluate_rules(self, row):
        bull_score = 0
        bear_score = 0
        triggered = []

        # Rule 1: PCR Direction (Strong Signal)
        if row['pcr_oi'] > 1.2:
            bull_score += 1.5
            triggered.append("BULL_HIGH_PCR")
        elif row['pcr_oi'] < 0.6:
            bear_score += 1.5
            triggered.append("BEAR_LOW_PCR")
        
        # Rule 2: PCR Momentum (5min)
        if row['delta_pcr_1m'] > 0.02:
            bull_score += 1.0
            triggered.append("BULL_PCR_RISING")
        elif row['delta_pcr_1m'] < -0.02:
            bear_score += 1.0
            triggered.append("BEAR_PCR_FALLING")

        # Rule 3: Short Covering (Price Up + Call OI Down)
        if row['delta_price_5m'] > 0 and row['delta_oi_ce_5m'] < 0:
            bull_score += 2.0
            triggered.append("BULL_SHORT_COVERING")

        # Rule 4: Long Unwinding (Price Down + Put OI Down)
        if row['delta_price_5m'] < 0 and row['delta_oi_pe_5m'] < 0:
            bear_score += 2.0
            triggered.append("BEAR_LONG_UNWINDING")

        # Rule 5: Put Writing (Support Building - Price Stable/Up + Put OI Up)
        if row['delta_oi_pe_5m'] > 50000 and row['delta_price_5m'] > -5:
            bull_score += 1.0
            triggered.append("BULL_PUT_WRITING")

        # Rule 6: Call Writing (Resistance Building - Price Stable/Down + Call OI Up)
        if row['delta_oi_ce_5m'] > 50000 and row['delta_price_5m'] < 5:
            bear_score += 1.0
            triggered.append("BEAR_CALL_WRITING")

        # Rule 7: VIX Divergence
        if row['vix'] < 13 and row['delta_price_5m'] > 0:
            bull_score += 0.5
            triggered.append("BULL_LOW_VIX_STABLE")
        elif row['vix'] > 20 and row['delta_price_5m'] < 0:
            bear_score += 1.0
            triggered.append("BEAR_HIGH_VIX_PANIC")

        # Rule 8: Price Momentum (15m Trend)
        if row['delta_price_15m'] > 20:
            bull_score += 1.0
            triggered.append("BULL_MOMENTUM_15M")
        elif row['delta_price_15m'] < -20:
            bear_score += 1.0
            triggered.append("BEAR_MOMENTUM_15M")

        return bull_score, bear_score, triggered

    def decide_signal(self, bull, bear):
        total = bull + bear + 1e-9
        diff = bull - bear
        
        if diff > 2.0:
            return "STRONG_BULL", bull / total
        elif diff > 1.0:
            return "BULL", bull / total
        elif diff < -2.0:
            return "STRONG_BEAR", bear / total
        elif diff < -1.0:
            return "BEAR", bear / total
        else:
            return "NEUTRAL", 0.5

    def run(self):
        print("Starting Rule Engine...")
        while True:
            try:
                df = self.get_latest_features()
                if df.empty:
                    time.sleep(5)
                    continue

                latest_row = df.iloc[0]
                current_ts = latest_row['timestamp']
                last_processed_ts = self.get_last_signal_ts()

                if last_processed_ts == current_ts:
                    time.sleep(10)
                    continue

                bull_score, bear_score, triggers = self.evaluate_rules(latest_row)
                signal, confidence = self.decide_signal(bull_score, bear_score)

                log_entry = {
                    'timestamp': current_ts,
                    'signal': signal,
                    'confidence': round(confidence, 2),
                    'bull_score': bull_score,
                    'bear_score': bear_score,
                    'active_rules': json.dumps(triggers),
                    'price': float(latest_row['price']),
                    'pcr': float(latest_row['pcr_oi'])
                }

                pd.DataFrame([log_entry]).to_sql('signal_logs', self.engine, if_exists='append', index=False)
                
                print(f"[{current_ts}] Signal: {signal} ({confidence:.2f}) | Rules: {triggers}")

            except Exception as e:
                print(f"Rule Engine Error: {e}")
                time.sleep(5)
            
            time.sleep(10)

if __name__ == "__main__":
    init_db()
    engine_worker = RuleEngine()
    engine_worker.run()