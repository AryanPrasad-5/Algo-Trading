from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DB_USER: str = "postgres"
    DB_PASS: str = "password"
    DB_HOST: str = "localhost"
    DB_PORT: str = "5432"
    DB_NAME: str = "trading_db"
    
    NSE_INDICES_URL: str = "https://www.nseindia.com/api/allIndices"
    NSE_CHAIN_URL_BASE: str = "https://www.nseindia.com/api/option-chain-indices?symbol="
    SYMBOL: str = "NIFTY"
    
    # ML Model Paths
    MODEL_PATH: str = "hybrid_model.pkl"
    SCALER_PATH: str = "scaler.pkl"

    @property
    def database_url(self) -> str:
        # Use SQLite for easier local setup
        return "sqlite:///./trading_db.sqlite"

    class Config:
        env_file = ".env"

settings = Settings()
