import os
import time
import requests
import pandas as pd
import yfinance as yf
import logging
from datetime import datetime
from sqlalchemy import create_engine, text
from pytz import timezone

DB_USER = "postgres"
DB_PASS = "password"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "trading_db"

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS market_data (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20),
    price NUMERIC,
    vix NUMERIC,
    PRIMARY KEY (timestamp, symbol)
);

CREATE TABLE IF NOT EXISTS option_chain_snapshot (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20),
    expiry DATE,
    strike NUMERIC,
    option_type VARCHAR(2),
    ltp NUMERIC,
    oi NUMERIC,
    volume NUMERIC,
    iv NUMERIC,
    underlying_price NUMERIC
);

CREATE INDEX IF NOT EXISTS idx_oc_ts_symbol ON option_chain_snapshot(timestamp, symbol);
"""

def init_db():
    with engine.connect() as conn:
        conn.execute(text(SCHEMA_SQL))
        conn.commit()

class DataIngestion:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.symbol = "NIFTY"
        self.indices_url = "https://www.nseindia.com/api/allIndices"
        self.chain_url = f"https://www.nseindia.com/api/option-chain-indices?symbol={self.symbol}"
        
        self._establish_session()

    def _establish_session(self):
        try:
            self.session.get("https://www.nseindia.com", timeout=10)
        except Exception as e:
            print(f"Session Init Error: {e}")

    def fetch_market_levels(self):
        try:
            ticker = yf.Ticker("^NSEI")
            vix_ticker = yf.Ticker("^INDIAVIX")
            
            nifty_data = ticker.history(period="1d", interval="1m")
            vix_data = vix_ticker.history(period="1d", interval="1m")

            if nifty_data.empty or vix_data.empty:
                return None, None

            current_price = nifty_data['Close'].iloc[-1]
            current_vix = vix_data['Close'].iloc[-1]
            
            return current_price, current_vix
        except Exception as e:
            print(f"Market Data Error: {e}")
            return None, None

    def fetch_option_chain(self, underlying_price):
        try:
            response = self.session.get(self.chain_url, timeout=10)
            if response.status_code == 401:
                self._establish_session()
                response = self.session.get(self.chain_url, timeout=10)
            
            data = response.json()
            records = data.get('records', {}).get('data', [])
            timestamp = datetime.now(timezone('Asia/Kolkata'))
            
            rows = []
            for record in records:
                expiry_date = record['expiryDate']
                strike = record['strikePrice']
                
                if 'CE' in record:
                    ce = record['CE']
                    rows.append({
                        'timestamp': timestamp,
                        'symbol': self.symbol,
                        'expiry': pd.to_datetime(expiry_date, format='%d-%b-%Y').date(),
                        'strike': strike,
                        'option_type': 'CE',
                        'ltp': ce.get('lastPrice', 0),
                        'oi': ce.get('openInterest', 0),
                        'volume': ce.get('totalTradedVolume', 0),
                        'iv': ce.get('impliedVolatility', 0),
                        'underlying_price': underlying_price
                    })
                
                if 'PE' in record:
                    pe = record['PE']
                    rows.append({
                        'timestamp': timestamp,
                        'symbol': self.symbol,
                        'expiry': pd.to_datetime(expiry_date, format='%d-%b-%Y').date(),
                        'strike': strike,
                        'option_type': 'PE',
                        'ltp': pe.get('lastPrice', 0),
                        'oi': pe.get('openInterest', 0),
                        'volume': pe.get('totalTradedVolume', 0),
                        'iv': pe.get('impliedVolatility', 0),
                        'underlying_price': underlying_price
                    })
            
            return pd.DataFrame(rows)

        except Exception as e:
            print(f"Chain Fetch Error: {e}")
            return pd.DataFrame()

    def save_market_data(self, price, vix):
        ts = datetime.now(timezone('Asia/Kolkata'))
        df = pd.DataFrame([{
            'timestamp': ts,
            'symbol': self.symbol,
            'price': price,
            'vix': vix
        }])
        df.to_sql('market_data', engine, if_exists='append', index=False)

    def save_option_chain(self, df):
        if not df.empty:
            df.to_sql('option_chain_snapshot', engine, if_exists='append', index=False)

    def run(self):
        print("Starting ingestion engine...")
        while True:
            start_time = time.time()
            
            price, vix = self.fetch_market_levels()
            
            if price:
                self.save_market_data(price, vix)
                chain_df = self.fetch_option_chain(price)
                self.save_option_chain(chain_df)
                print(f"[{datetime.now().time()}] Data saved. Price: {price}, VIX: {vix}, Chain Rows: {len(chain_df)}")
            else:
                print("Failed to fetch market data.")

            time_taken = time.time() - start_time
            sleep_time = max(0, 60 - time_taken)
            time.sleep(sleep_time)


if __name__ == "__main__":
    init_db()
    ingestor = DataIngestion()
    ingestor.run()