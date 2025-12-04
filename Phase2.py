import time
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
from pytz import timezone

DB_USER = "postgres"
DB_PASS = "password"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "trading_db"

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS market_features (
    timestamp TIMESTAMPTZ PRIMARY KEY,
    price NUMERIC,
    vix NUMERIC,
    pcr_oi NUMERIC,
    pcr_vol NUMERIC,
    total_ce_oi NUMERIC,
    total_pe_oi NUMERIC,
    total_ce_vol NUMERIC,
    total_pe_vol NUMERIC,
    atm_iv NUMERIC,
    delta_oi_ce_1m NUMERIC,
    delta_oi_pe_1m NUMERIC,
    delta_pcr_1m NUMERIC,
    delta_price_1m NUMERIC,
    delta_oi_ce_5m NUMERIC,
    delta_oi_pe_5m NUMERIC,
    delta_price_5m NUMERIC,
    delta_oi_ce_15m NUMERIC,
    delta_oi_pe_15m NUMERIC,
    delta_price_15m NUMERIC
);
"""

def init_db():
    with engine.connect() as conn:
        conn.execute(text(SCHEMA_SQL))
        conn.commit()

class FeatureEngine:
    def __init__(self):
        self.engine = engine

    def get_latest_snapshot(self):
        query = """
        WITH latest_ts AS (
            SELECT MAX(timestamp) as max_ts FROM option_chain_snapshot
        )
        SELECT o.* FROM option_chain_snapshot o
        JOIN latest_ts l ON o.timestamp = l.max_ts
        """
        return pd.read_sql(query, self.engine)

    def get_market_data(self, timestamp):
        query = text("SELECT price, vix FROM market_data WHERE timestamp = :ts")
        result = pd.read_sql(query, self.engine, params={"ts": timestamp})
        if not result.empty:
            return result.iloc[0]['price'], result.iloc[0]['vix']
        return None, None

    def get_past_features(self, limit=16):
        query = f"""
        SELECT * FROM market_features 
        ORDER BY timestamp DESC 
        LIMIT {limit}
        """
        df = pd.read_sql(query, self.engine)
        return df.sort_values('timestamp').reset_index(drop=True)

    def compute_and_store(self):
        df_chain = self.get_latest_snapshot()
        
        if df_chain.empty:
            return

        current_ts = df_chain['timestamp'].iloc[0]
        
        existing_check = pd.read_sql(
            text("SELECT 1 FROM market_features WHERE timestamp = :ts"), 
            self.engine, 
            params={"ts": current_ts}
        )
        if not existing_check.empty:
            return

        underlying_price = df_chain['underlying_price'].iloc[0]
        
        ce_df = df_chain[df_chain['option_type'] == 'CE']
        pe_df = df_chain[df_chain['option_type'] == 'PE']

        total_ce_oi = ce_df['oi'].sum()
        total_pe_oi = pe_df['oi'].sum()
        total_ce_vol = ce_df['volume'].sum()
        total_pe_vol = pe_df['volume'].sum()

        pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        pcr_vol = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0

        atm_strike = round(underlying_price / 50) * 50
        atm_row = df_chain[df_chain['strike'] == atm_strike]
        atm_iv = atm_row['iv'].mean() if not atm_row.empty else 0

        price, vix = self.get_market_data(current_ts)
        if price is None:
            price = underlying_price
        if vix is None:
            vix = 0

        past_df = self.get_past_features(limit=16)

        delta_oi_ce_1m = 0
        delta_oi_pe_1m = 0
        delta_pcr_1m = 0
        delta_price_1m = 0
        
        delta_oi_ce_5m = 0
        delta_oi_pe_5m = 0
        delta_price_5m = 0
        
        delta_oi_ce_15m = 0
        delta_oi_pe_15m = 0
        delta_price_15m = 0

        if not past_df.empty:
            last_row = past_df.iloc[-1]
            delta_oi_ce_1m = total_ce_oi - last_row['total_ce_oi']
            delta_oi_pe_1m = total_pe_oi - last_row['total_pe_oi']
            delta_pcr_1m = pcr_oi - last_row['pcr_oi']
            delta_price_1m = price - last_row['price']

            if len(past_df) >= 5:
                row_5m = past_df.iloc[-5]
                delta_oi_ce_5m = total_ce_oi - row_5m['total_ce_oi']
                delta_oi_pe_5m = total_pe_oi - row_5m['total_pe_oi']
                delta_price_5m = price - row_5m['price']

            if len(past_df) >= 15:
                row_15m = past_df.iloc[-15]
                delta_oi_ce_15m = total_ce_oi - row_15m['total_ce_oi']
                delta_oi_pe_15m = total_pe_oi - row_15m['total_pe_oi']
                delta_price_15m = price - row_15m['price']

        feature_row = {
            'timestamp': current_ts,
            'price': price,
            'vix': vix,
            'pcr_oi': pcr_oi,
            'pcr_vol': pcr_vol,
            'total_ce_oi': total_ce_oi,
            'total_pe_oi': total_pe_oi,
            'total_ce_vol': total_ce_vol,
            'total_pe_vol': total_pe_vol,
            'atm_iv': atm_iv,
            'delta_oi_ce_1m': delta_oi_ce_1m,
            'delta_oi_pe_1m': delta_oi_pe_1m,
            'delta_pcr_1m': delta_pcr_1m,
            'delta_price_1m': delta_price_1m,
            'delta_oi_ce_5m': delta_oi_ce_5m,
            'delta_oi_pe_5m': delta_oi_pe_5m,
            'delta_price_5m': delta_price_5m,
            'delta_oi_ce_15m': delta_oi_ce_15m,
            'delta_oi_pe_15m': delta_oi_pe_15m,
            'delta_price_15m': delta_price_15m
        }

        pd.DataFrame([feature_row]).to_sql('market_features', self.engine, if_exists='append', index=False)
        print(f"Features stored for {current_ts} | PCR: {pcr_oi:.2f} | Price: {price}")

    def run(self):
        print("Starting Feature Engine...")
        while True:
            try:
                self.compute_and_store()
            except Exception as e:
                print(f"Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    init_db()
    engine_worker = FeatureEngine()
    engine_worker.run()