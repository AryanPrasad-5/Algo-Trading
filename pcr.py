import pandas as pd
import numpy as np

class PCRCalculator:
    @staticmethod
    def calculate_pcr(chain_df: pd.DataFrame) -> dict:
        """
        Calculates PCR and other aggregate metrics from the option chain DataFrame.
        """
        if chain_df.empty:
            return {
                "pcr_oi": 0,
                "pcr_vol": 0,
                "total_ce_oi": 0,
                "total_pe_oi": 0
            }
            
        total_ce_oi = chain_df[chain_df['option_type'] == 'CE']['oi'].sum()
        total_pe_oi = chain_df[chain_df['option_type'] == 'PE']['oi'].sum()
        
        total_ce_vol = chain_df[chain_df['option_type'] == 'CE']['volume'].sum()
        total_pe_vol = chain_df[chain_df['option_type'] == 'PE']['volume'].sum()
        
        pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        pcr_vol = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0
        
        return {
            "pcr_oi": round(pcr_oi, 4),
            "pcr_vol": round(pcr_vol, 4),
            "total_ce_oi": total_ce_oi,
            "total_pe_oi": total_pe_oi
        }

    @staticmethod
    def calculate_atm_iv(chain_df: pd.DataFrame, underlying_price: float) -> float:
        """
        Finds the ATM strike and returns the average IV of CE and PE.
        """
        if chain_df.empty:
            return 0.0
            
        # Find ATM strike (closest to underlying price)
        unique_strikes = chain_df['strike'].unique()
        atm_strike = unique_strikes[np.abs(unique_strikes - underlying_price).argmin()]
        
        atm_options = chain_df[chain_df['strike'] == atm_strike]
        
        ivs = atm_options['iv'].tolist()
        if not ivs:
            return 0.0
            
        return round(sum(ivs) / len(ivs), 2)
