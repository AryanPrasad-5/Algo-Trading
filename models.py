from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Numeric, Index
from app.database import Base

class MarketData(Base):
    __tablename__ = "market_data"
    
    timestamp = Column(DateTime(timezone=True), primary_key=True)
    symbol = Column(String(20), primary_key=True)
    price = Column(Numeric)
    vix = Column(Numeric)

class OptionChainSnapshot(Base):
    __tablename__ = "option_chain_snapshot"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    symbol = Column(String(20))
    expiry = Column(Date)
    strike = Column(Numeric)
    option_type = Column(String(2))
    ltp = Column(Numeric)
    oi = Column(Numeric)
    volume = Column(Numeric)
    iv = Column(Numeric)
    underlying_price = Column(Numeric)
    
    __table_args__ = (
        Index('idx_oc_ts_symbol', 'timestamp', 'symbol'),
    )

class MarketFeatures(Base):
    __tablename__ = "market_features"
    
    timestamp = Column(DateTime(timezone=True), primary_key=True)
    symbol = Column(String(20), primary_key=True)
    price = Column(Float)
    vix = Column(Float)
    pcr_oi = Column(Float)
    pcr_vol = Column(Float)
    atm_iv = Column(Float)
    delta_oi_ce_5m = Column(Float)
    delta_oi_pe_5m = Column(Float)
    delta_pcr_1m = Column(Float)
    delta_price_5m = Column(Float)
    delta_oi_ce_15m = Column(Float)
    delta_oi_pe_15m = Column(Float)
    delta_price_15m = Column(Float)
