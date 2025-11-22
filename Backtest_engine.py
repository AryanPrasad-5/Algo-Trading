import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

DB_USER = "postgres"
DB_PASS = "password"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "trading_db"
MODEL_PATH = "hybrid_model.pkl"
SCALER_PATH = "scaler.pkl"

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

class BacktestEngine:
    def __init__(self):
        self.engine = engine
        self.model = None
        self.scaler = None
        self.initial_capital = 100000
        self.capital = self.initial_capital
        self.trades = []
        self.current_position = None 
        self.equity_curve = []
        
        self.feature_cols = [
            'pcr_oi', 'pcr_vol', 'atm_iv', 'vix',
            'delta_oi_ce_5m', 'delta_oi_pe_5m', 
            'delta_pcr_1m', 'delta_price_5m', 
            'delta_oi_ce_15m', 'delta_oi_pe_15m', 'delta_price_15m'
        ]
        
        try:
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            print("ML Model loaded successfully.")
        except:
            print("Warning: ML Model not found. Running on Rules only.")

    def get_historical_data(self):
        print("Fetching historical data...")
        query = "SELECT * FROM market_features ORDER BY timestamp ASC"
        df = pd.read_sql(query, self.engine)
        df = df.fillna(0)
        return df

    def get_rule_signal(self, row):
        bull_score = 0
        bear_score = 0

        if row['pcr_oi'] > 1.2: bull_score += 1.5
        elif row['pcr_oi'] < 0.6: bear_score += 1.5
        
        if row['delta_pcr_1m'] > 0.02: bull_score += 1.0
        elif row['delta_pcr_1m'] < -0.02: bear_score += 1.0

        if row['delta_price_5m'] > 0 and row['delta_oi_ce_5m'] < 0: bull_score += 2.0
        if row['delta_price_5m'] < 0 and row['delta_oi_pe_5m'] < 0: bear_score += 2.0

        if row['vix'] < 13 and row['delta_price_5m'] > 0: bull_score += 0.5
        if row['vix'] > 20 and row['delta_price_5m'] < 0: bear_score += 1.0

        if bull_score > bear_score + 1.5: return 1  # Buy
        if bear_score > bull_score + 1.5: return -1 # Sell
        return 0

    def get_ml_probability(self, row):
        if not self.model: return 0.5
        
        features = row[self.feature_cols].values.reshape(1, -1)
        scaled = self.scaler.transform(features)
        prob = self.model.predict_proba(scaled)[0][1] 
        return prob

    def run(self):
        df = self.get_historical_data()
        if df.empty:
            print("No data found in DB.")
            return

        print(f"Running backtest on {len(df)} candles...")

        for i, row in df.iterrows():
            current_price = row['price']
            ts = row['timestamp']

            if self.current_position:
                entry_price = self.current_position['price']
                direction = self.current_position['type']
                
                if direction == 'LONG':
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price

                # TP/SL Logic (Scalping: TP 0.2%, SL 0.1%)
                if pnl_pct >= 0.002 or pnl_pct <= -0.001:
                    pnl_abs = self.capital * pnl_pct
                    self.capital += pnl_abs
                    self.trades.append({
                        'entry_ts': self.current_position['ts'],
                        'exit_ts': ts,
                        'type': direction,
                        'entry': entry_price,
                        'exit': current_price,
                        'pnl': pnl_abs,
                        'pnl_pct': pnl_pct * 100
                    })
                    self.current_position = None

            else:
                rule_sig = self.get_rule_signal(row)
                ml_prob = self.get_ml_probability(row)
                
                # Hybrid Logic
                # Long: Rule says Buy AND ML Prob > 60%
                # Short: Rule says Sell AND ML Prob < 40%
                
                if rule_sig == 1 and ml_prob > 0.60:
                    self.current_position = {'ts': ts, 'price': current_price, 'type': 'LONG'}
                elif rule_sig == -1 and ml_prob < 0.40:
                    self.current_position = {'ts': ts, 'price': current_price, 'type': 'SHORT'}

            self.equity_curve.append(self.capital)

        self.generate_report()

    def generate_report(self):
        if not self.trades:
            print("No trades triggered.")
            return

        trades_df = pd.DataFrame(self.trades)
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        
        total_trades = len(trades_df)
        win_rate = len(wins) / total_trades * 100
        total_pnl = trades_df['pnl'].sum()
        max_dd = (pd.Series(self.equity_curve).cummax() - pd.Series(self.equity_curve)).max()

        print("\n" + "="*30)
        print("BACKTEST RESULTS")
        print("="*30)
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate:     {win_rate:.2f}%")
        print(f"Total P&L:    {total_pnl:.2f}")
        print(f"Final Equity: {self.capital:.2f}")
        print(f"Max Drawdown: {max_dd:.2f}")
        print("="*30)

        print("\nRecent Trades:")
        print(trades_df.tail(5)[['entry_ts', 'type', 'pnl', 'pnl_pct']])

if __name__ == "__main__":
    bt = BacktestEngine()
    bt.run()