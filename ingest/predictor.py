"""
Revenue Prediction Model
=========================
Uses linear regression to predict quarterly revenue from alternative data signals.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Optional
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')


# Wall Street Consensus Data (Mock - Replace with real FactSet/Bloomberg data)
WALL_STREET_CONSENSUS = {
    "STARBUCKS": {
        "2025_Q1": {"revenue_estimate": 9.1e9, "eps_estimate": 0.93}, # Oct-Dec (fiscal Q1)
        "2024_Q4": {"revenue_estimate": 9.0e9, "eps_estimate": 0.90},
    },
    "MCDONALD'S": {
        "2025_Q1": {"revenue_estimate": 6.2e9, "eps_estimate": 2.70},
    },
    "CHIPOTLE": {
        "2025_Q1": {"revenue_estimate": 2.8e9, "eps_estimate": 0.32},
    },
}

# Historical actual revenue (for model training)
HISTORICAL_REVENUE = {
    "STARBUCKS": {
        # Fiscal Year (starts in October)
        "2023_Q1": 8.71e9, "2023_Q2": 9.17e9, "2023_Q3": 9.37e9, "2023_Q4": 9.06e9,
        "2024_Q1": 9.43e9, "2024_Q2": 9.11e9, "2024_Q3": 9.07e9, "2024_Q4": 9.40e9,
    },
    "MCDONALD'S": {
        "2023_Q1": 5.90e9, "2023_Q2": 6.50e9, "2023_Q3": 6.69e9, "2023_Q4": 6.41e9,
        "2024_Q1": 6.17e9, "2024_Q2": 6.49e9, "2024_Q3": 6.87e9, "2024_Q4": 6.39e9,
    },
    "CHIPOTLE": {
        "2023_Q1": 2.39e9, "2023_Q2": 2.51e9, "2023_Q3": 2.47e9, "2023_Q4": 2.52e9,
        "2024_Q1": 2.70e9, "2024_Q2": 2.97e9, "2024_Q3": 2.79e9, "2024_Q4": 2.85e9,
    },
}


class RevenuePredictor:
    """Predicts quarterly revenue using alternative data signals."""

    def __init__(self, spend_path = "data/clean_spend_daily.csv", traffic_path = "data/clean_traffic_daily.csv"):
        base_dir = Path(__file__).resolve().parents[1]
        spend_path = Path(spend_path)
        traffic_path = Path(traffic_path)
        if not spend_path.is_absolute():
            spend_path = base_dir / spend_path
        if not traffic_path.is_absolute():
            traffic_path = base_dir / traffic_path

        # 1. Load Spend Data
        self.spend_df = pd.read_csv(spend_path)
        self.spend_df['date'] = pd.to_datetime(self.spend_df['date'])

        # 2. Load Traffic Data
        try:
            self.traffic_df = pd.read_csv(traffic_path)
            self.traffic_df['date'] = pd.to_datetime(self.traffic_df['date'])
            self.has_traffic = True
        except Exception as e:
            print(f"Warning: traffic data not found. Running in Spend-Only mode")
            self.has_traffic = False
            self.traffic_df = pd.DataFrame() # Empty DataFrame

        self.models = {}
        self.correlations = {}

    def _get_fiscal_quarter(self, date: datetime, ticker: str) -> str:
        # Starbucks Fiscal year starts in October = Q1
        if ticker == "STARBUCKS":
            if date.month >= 10: return f"{date.year + 1}_Q1"
            elif date.month >= 7: return f"{date.year}_Q4"
            elif date.month >= 4: return f"{date.year}_Q3"
            else: return f"{date.year}_Q2"
        
        # Standard Calendar year for MCD, CMG
        quarter = (date.month - 1) // 3 + 1
        return f"{date.year}_Q{quarter}"

    def _normalize_brand(self, brand: str) -> str:
        mapping = {
            "STARBUCKS (MERCHANT)": "STARBUCKS",
            "STARBUCKS CARD": "STARBUCKS",
            "CHIPOTLE MEXICAN": "CHIPOTLE",
            "DOMINO'S PIZZA": "DOMINO'S",
            "DUNKIN' DONUTS": "DUNKIN",
        }
        return mapping.get(brand, brand) # Normalize brand names
    
    def get_merged_data(self, brand: str) -> pd.DataFrame:
        # Filter Spend
        spend_vars = [b for b in self.spend_df['brand'].unique() if self._normalize_brand(b) == brand]
        df_spend = self.spend_df[self.spend_df['brand'].isin(spend_vars)].copy()
        
        if df_spend.empty: return pd.DataFrame()
        
        # Group Spend by Date (handle multiple merchant IDs)
        df_spend = df_spend.groupby('date').agg({
            'spend': 'sum',
            'transactions': 'sum'
        }).reset_index()
        
        # Merge Traffic (if available)
        if self.has_traffic:
            traffic_vars = [b for b in self.traffic_df['brand'].unique() if self._normalize_brand(b) == brand]
            df_traffic = self.traffic_df[self.traffic_df['brand'].isin(traffic_vars)].copy()
            
            if not df_traffic.empty:
                df_traffic = df_traffic.groupby('date').agg({'total_visits': 'sum' }).reset_index()
                # INNER JOIN: We only want days where we have BOTH signals
                merged = pd.merge(df_spend, df_traffic, on='date', how='inner')
                return merged

        return df_spend # return spend data if no traffic data is available

    def aggregate_quarterly(self, brand: str) -> pd.DataFrame:
        df = self.get_merged_data(brand)
        if df.empty: return pd.DataFrame()

        df['quarter'] = df['date'].apply(lambda d: self._get_fiscal_quarter(d, brand))

        # Aggregation Logic
        aggs = {
            'spend': 'sum',
            'transactions': 'sum',
            'date': ['min', 'max', 'count']
        }
        if 'total_visits' in df.columns:
            aggs['total_visits'] = 'sum'

        quarterly = df.groupby('quarter').agg(aggs).reset_index()

        # Flatten multi-level columns from aggregation
        quarterly.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                            for col in quarterly.columns]

        # Filter incomplete quarters (need at least 60 days of data)
        quarterly = quarterly[quarterly['date_count'] >= 60]
        return quarterly

    def train_model(self, brand: str) -> Tuple[float, float]:
        if brand not in HISTORICAL_REVENUE:
            return (0.0, 0.0)

        quarterly = self.aggregate_quarterly(brand)
        if quarterly.empty:
            return (0.0, 0.0)

        historical = HISTORICAL_REVENUE[brand]
        train_data = []
        
        for _, row in quarterly.iterrows():
            q = row['quarter']
            if q in historical:
                record = {
                    'quarter': q,
                    'spend': row['spend_sum'],
                    'transactions': row['transactions_sum'],
                    'actual_revenue': historical[q]
                }
                if 'total_visits_sum' in row.index:
                    record['visits'] = row['total_visits_sum']
                train_data.append(record)

        if len(train_data) < 3:
            return (0.0, 0.0)

        df = pd.DataFrame(train_data)

        # Select Features: Prefer visits if available, else Transactions
        features = ['spend', 'visits'] if 'visits' in df.columns else ['spend', 'transactions']
        X = df[features].values
        y = df['actual_revenue'].values

        model = LinearRegression()
        model.fit(X, y)
        self.models[brand] = {'model': model, 'features': features}

        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        self.correlations[brand] = r2

        return (r2, np.sqrt(r2))  # Return R² and R

    def predict_revenue(self, brand: str) -> Dict:
        if brand not in self.models:
            self.train_model(brand)
            if brand not in self.models: # If still not trained, return error
                return {'error': f'No model for {brand}'}

        quarterly = self.aggregate_quarterly(brand)
        if quarterly.empty: return {'error': f'No data for {brand}'}

        current_q = quarterly.iloc[-1] # Get the most recent quarter
        quarter = current_q['quarter']

        # Map feature names to actual column names
        feature_map = {
            'spend': 'spend_sum',
            'transactions': 'transactions_sum',
            'visits': 'total_visits_sum'
        }

        features = self.models[brand]['features']
        actual_cols = [feature_map.get(f, f) for f in features]

        # Check columns exist
        missing = [c for c in actual_cols if c not in current_q.index]
        if missing:
            return {'error': f'Missing columns {missing} for {brand}'}

        X = current_q[actual_cols].values.reshape(1, -1)
        predicted = self.models[brand]['model'].predict(X)[0]

        consensus = WALL_STREET_CONSENSUS.get(brand, {}).get(quarter, {}).get('revenue_estimate', predicted)
        delta_pct = ((predicted - consensus) / consensus) * 100 # Calculate percentage difference

        correlation = self.correlations.get(brand, 0)
        
        if correlation >= 0.9:
            signal_strength = "Very High"
        elif correlation >= 0.8:
            signal_strength = "High"
        elif correlation >= 0.7:
            signal_strength = "Medium"
        else:
            signal_strength = "Low"

        # Compatibility return Dictionary 
        return {
            'brand': brand,
            'quarter': quarter,
            'predicted_revenue': predicted,
            'wall_street_consensus': consensus, # Compatible Key
            'delta_pct': round(delta_pct, 2),
            'delta_direction': 'BEAT' if delta_pct > 0 else 'MISS',
            'signal_strength': signal_strength,
            'correlation': round(correlation, 3),
            'alt_data_spend': current_q['spend_sum'],
            'days_in_quarter': current_q['date_count'],
        }

    # help for Dashboard Charts 
    def get_trend_data(self, brand: str) -> pd.DataFrame:
        df = self.get_merged_data(brand)
        if df.empty: return pd.DataFrame()  # Return empty DataFrame if no data
        
        df = df.sort_values('date')

        # Calculate Avg ticket size
        df['avg_ticket_size'] = df['spend'] / df['transactions']
        df['avg_ticket_size'] = df['avg_ticket_size'].fillna(0)

        df['spend_7d_avg'] = df['spend'].rolling(7).mean()
        df['spend_30d_avg'] = df['spend'].rolling(30).mean()
        df['transactions_7d_avg'] = df['transactions'].rolling(7).mean()
        df['spend_ly'] = df['spend'].shift(365)
        df['spend_yoy_pct'] = ((df['spend'] - df['spend_ly']) / df['spend_ly'] * 100).round(2)
        
        if 'total_visits' in df.columns:
            df['visits_7d_avg'] = df['total_visits'].rolling(7).mean()

        return df

def generate_dashboard_metrics(spend_data_path="data/clean_spend_daily.csv") -> Dict:
    predictor = RevenuePredictor(spend_data_path)
    signals = []
    trends = {}

    for brand in ["STARBUCKS", "MCDONALD'S", "CHIPOTLE"]:
        signal = predictor.predict_revenue(brand)
        if 'error' not in signal:
            signals.append(signal)
            trends[brand] = predictor.get_trend_data(brand)


    return {
        'alpha_metrics': signals,
        'trend_data': {k: v.to_dict('records') for k, v in trends.items()},
        'last_updated': datetime.now().isoformat()
    }


if __name__ == "__main__":
    print("Testing Revenue Predictor...")
    predictor = RevenuePredictor()

    for brand in ["STARBUCKS", "MCDONALD'S", "CHIPOTLE"]:
        print(f"\n---Analyzing {brand}... ---")
        r2, corr = predictor.train_model(brand)
        print(f"R²: {r2:.3f}, Correlation: {corr:.3f}")

        signal = predictor.predict_revenue(brand)
        if 'error' not in signal:
            print(f"Predicted: ${signal['predicted_revenue']/1e9:.2f}B")
            print(f"Consensus: ${signal['wall_street_consensus']/1e9:.2f}B")
            print(f"Delta: {signal['delta_pct']:+.2f}%")
