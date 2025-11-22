import pandas as pd
import numpy as np
import joblib
import time
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


DB_USER = "postgres"
DB_PASS = "password"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "trading_db"
MODEL_PATH = "hybrid_model.pkl"
SCALER_PATH = "scaler.pkl"

PROFIT_THRESHOLD_PCT = 0.0002 
LOOKAHEAD_MINUTES = 5

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

class AdvancedMLTrainer:
    def __init__(self):
        self.engine = engine
        self.model = None
        self.scaler = None
        
        self.feature_cols = [
            'pcr_oi', 'pcr_vol', 'atm_iv', 'vix',
            'delta_oi_ce_5m', 'delta_oi_pe_5m', 
            'delta_pcr_1m', 'delta_price_5m', 
            'delta_oi_ce_15m', 'delta_oi_pe_15m', 'delta_price_15m'
        ]

    def load_data(self):
        print("Loading data from Postgres...")
        query = """
        SELECT * FROM market_features 
        ORDER BY timestamp ASC
        """
        df = pd.read_sql(query, self.engine)
        return df

    def engineer_targets(self, df):
        """
        Create the target label: 
        1 (Buy) if Price in 5 mins > Current Price + Threshold
        0 (No Trade) otherwise.
        """
       
        df['future_return'] = df['price'].shift(-LOOKAHEAD_MINUTES) / df['price'] - 1
        
        
        df['target'] = (df['future_return'] > PROFIT_THRESHOLD_PCT).astype(int)
        
        
        df = df.dropna(subset=['target', 'future_return'])
        
        
        df = df.fillna(0)
        
        return df

    def train_model(self):
        df_raw = self.load_data()
        
        if len(df_raw) < 200:
            print("Not enough data to train a robust model (Need > 200 rows). Exiting.")
            return

        print(f"Preprocessing {len(df_raw)} rows...")
        df = self.engineer_targets(df_raw)
        
        X = df[self.feature_cols]
        y = df['target']

        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        
        print("\n--- Starting Walk-Forward Validation ---")
        tscv = TimeSeriesSplit(n_splits=5)
        
        rf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=4, n_jobs=-1, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
        
        
        model = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')

        fold = 1
        precisions = []
        
        for train_index, test_index in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            p = precision_score(y_test, preds, zero_division=0)
            r = recall_score(y_test, preds, zero_division=0)
            a = accuracy_score(y_test, preds)
            
            print(f"Fold {fold}: Precision: {p:.2f} | Recall: {r:.2f} | Acc: {a:.2f} | Samples: {len(X_test)}")
            precisions.append(p)
            fold += 1

        print(f"\nAverage Precision: {np.mean(precisions):.2f}")
        
        
        print("\nTraining Final Production Model on full dataset...")
        self.model = model
        self.model.fit(X_scaled, y)
        
    
        rf_part = self.model.named_estimators_['rf']
        importances = rf_part.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\nFeature Importance:")
        for f in range(X.shape[1]):
            print(f"{f+1}. {self.feature_cols[indices[f]]} ({importances[indices[f]]:.4f})")

        
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        print("\nModel & Scaler saved successfully.")

    def predict(self, features_dict):
        """
        Call this from your live trading loop.
        features_dict: Dictionary containing latest values of feature_cols
        """
        if self.model is None:
            try:
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
            except:
                return 0.0

        
        df = pd.DataFrame([features_dict])
        
       
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
                
        X = df[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        
        
        prob = self.model.predict_proba(X_scaled)[0][1]
        return prob

if __name__ == "__main__":
    trainer = AdvancedMLTrainer()
    trainer.train_model()
import pandas as pd
import numpy as np
import joblib
import time
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


DB_USER = "postgres"
DB_PASS = "password"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "trading_db"
MODEL_PATH = "hybrid_model.pkl"
SCALER_PATH = "scaler.pkl"


PROFIT_THRESHOLD_PCT = 0.0002 
LOOKAHEAD_MINUTES = 5

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

class AdvancedMLTrainer:
    def __init__(self):
        self.engine = engine
        self.model = None
        self.scaler = None
    
        self.feature_cols = [
            'pcr_oi', 'pcr_vol', 'atm_iv', 'vix',
            'delta_oi_ce_5m', 'delta_oi_pe_5m', 
            'delta_pcr_1m', 'delta_price_5m', 
            'delta_oi_ce_15m', 'delta_oi_pe_15m', 'delta_price_15m'
        ]

    def load_data(self):
        print("Loading data from Postgres...")
        query = """
        SELECT * FROM market_features 
        ORDER BY timestamp ASC
        """
        df = pd.read_sql(query, self.engine)
        return df

    def engineer_targets(self, df):
        """
        Create the target label: 
        1 (Buy) if Price in 5 mins > Current Price + Threshold
        0 (No Trade) otherwise.
        """
        
        df['future_return'] = df['price'].shift(-LOOKAHEAD_MINUTES) / df['price'] - 1
        
        
        df['target'] = (df['future_return'] > PROFIT_THRESHOLD_PCT).astype(int)
        
        
        df = df.dropna(subset=['target', 'future_return'])
        
        
        df = df.fillna(0)
        
        return df

    def train_model(self):
        df_raw = self.load_data()
        
        if len(df_raw) < 200:
            print("Not enough data to train a robust model (Need > 200 rows). Exiting.")
            return

        print(f"Preprocessing {len(df_raw)} rows...")
        df = self.engineer_targets(df_raw)
        
        X = df[self.feature_cols]
        y = df['target']

        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        
        print("\n--- Starting Walk-Forward Validation ---")
        tscv = TimeSeriesSplit(n_splits=5)
        
        rf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=4, n_jobs=-1, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
        
        
        model = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')

        fold = 1
        precisions = []
        
        for train_index, test_index in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            p = precision_score(y_test, preds, zero_division=0)
            r = recall_score(y_test, preds, zero_division=0)
            a = accuracy_score(y_test, preds)
            
            print(f"Fold {fold}: Precision: {p:.2f} | Recall: {r:.2f} | Acc: {a:.2f} | Samples: {len(X_test)}")
            precisions.append(p)
            fold += 1

        print(f"\nAverage Precision: {np.mean(precisions):.2f}")
        
        
        print("\nTraining Final Production Model on full dataset...")
        self.model = model
        self.model.fit(X_scaled, y)
        
        
        rf_part = self.model.named_estimators_['rf']
        importances = rf_part.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\nFeature Importance:")
        for f in range(X.shape[1]):
            print(f"{f+1}. {self.feature_cols[indices[f]]} ({importances[indices[f]]:.4f})")

        
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        print("\nModel & Scaler saved successfully.")

    def predict(self, features_dict):
        """
        Call this from your live trading loop.
        features_dict: Dictionary containing latest values of feature_cols
        """
        if self.model is None:
            try:
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
            except:
                return 0.0

        
        df = pd.DataFrame([features_dict])
        
        
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
                
        X = df[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        
       
        prob = self.model.predict_proba(X_scaled)[0][1]
        return prob

if __name__ == "__main__":
    trainer = AdvancedMLTrainer()
    trainer.train_model()