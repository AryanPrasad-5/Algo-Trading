import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime
from pytz import timezone
from sqlalchemy.orm import Session
from app.database import SessionLocal, engine
from app.models import MarketData, OptionChainSnapshot, MarketFeatures
from app.config import settings
from app.services.pcr import PCRCalculator

class DataIngestion:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.symbol = settings.SYMBOL
        self.indices_url = settings.NSE_INDICES_URL
        self.chain_url = f"{settings.NSE_CHAIN_URL_BASE}{self.symbol}"
        
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

    def save_market_data(self, db: Session, price, vix):
        ts = datetime.now(timezone('Asia/Kolkata'))
        market_data = MarketData(
            timestamp=ts,
            symbol=self.symbol,
            price=price,
            vix=vix
        )
        db.add(market_data)
        db.commit()
        return ts

    def save_option_chain(self, df):
        if not df.empty:
            df.to_sql('option_chain_snapshot', engine, if_exists='append', index=False)

    def calculate_and_save_features(self, db: Session, chain_df: pd.DataFrame, price: float, vix: float, timestamp):
        """
        Calculates features and saves to MarketFeatures table.
        """
        if chain_df.empty:
            return

        pcr_metrics = PCRCalculator.calculate_pcr(chain_df)
        atm_iv = PCRCalculator.calculate_atm_iv(chain_df, price)
        
        # Calculate deltas (simplified for now, ideally fetch previous record)
        # For a robust implementation, we would query the previous record from DB here.
        # We will initialize deltas to 0 for this step to keep it simple, 
        # or implement a fetch_previous_feature function.
        
        feature_record = MarketFeatures(
            timestamp=timestamp,
            symbol=self.symbol,
            price=price,
            vix=vix,
            pcr_oi=pcr_metrics['pcr_oi'],
            pcr_vol=pcr_metrics['pcr_vol'],
            atm_iv=atm_iv,
            delta_oi_ce_5m=0.0, # Placeholder
            delta_oi_pe_5m=0.0, # Placeholder
            delta_pcr_1m=0.0,   # Placeholder
            delta_price_5m=0.0, # Placeholder
            delta_oi_ce_15m=0.0,# Placeholder
            delta_oi_pe_15m=0.0,# Placeholder
            delta_price_15m=0.0 # Placeholder
        )
        db.add(feature_record)
        db.commit()

    def run(self):
        print("Starting ingestion engine...")
        db = SessionLocal()
        try:
            while True:
                start_time = time.time()
                
                price, vix = self.fetch_market_levels()
                
                if price:
                    ts = self.save_market_data(db, price, vix)
                    chain_df = self.fetch_option_chain(price)
                    self.save_option_chain(chain_df)
                    self.calculate_and_save_features(db, chain_df, price, vix, ts)
                    
                    print(f"[{datetime.now().time()}] Data saved. Price: {price}, VIX: {vix}, Chain Rows: {len(chain_df)}")
                else:
                    print("Failed to fetch market data.")

                time_taken = time.time() - start_time
                sleep_time = max(0, 60 - time_taken)
                time.sleep(sleep_time)
        finally:
            db.close()

if __name__ == "__main__":
    # Create tables if they don't exist
    from app.database import Base, engine
    Base.metadata.create_all(bind=engine)
    
    ingestor = DataIngestion()
    ingestor.run()
