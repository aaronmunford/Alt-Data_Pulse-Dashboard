"""
Revenue Prediction Model
=========================
Uses linear regression to predict quarterly revenue from alternative data signals.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Optional
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

try:
    from ingest.lseg_client import LSEGConsensusClient
except ImportError:  # Allows running from the ingest/ folder directly
    try:
        from lseg_client import LSEGConsensusClient
    except ImportError:
        LSEGConsensusClient = None

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

    def __init__(
        self,
        spend_path="data/clean_spend_daily.csv",
        traffic_path="data/clean_traffic_daily.csv",
        retail_sales_path="data/market_data_retail_sales.csv",
    ):
        base_dir = Path(__file__).resolve().parents[1]
        spend_path = Path(spend_path)
        traffic_path = Path(traffic_path)
        retail_sales_path = Path(retail_sales_path)
        if not spend_path.is_absolute():
            spend_path = base_dir / spend_path
        if not traffic_path.is_absolute():
            traffic_path = base_dir / traffic_path
        if not retail_sales_path.is_absolute():
            retail_sales_path = base_dir / retail_sales_path

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

        # 3. Load Retail Sales (Macro) Data
        try:
            self.retail_df = pd.read_csv(retail_sales_path)
            self.retail_df['DATE'] = pd.to_datetime(self.retail_df['DATE'])
            self.retail_df = self.retail_df.sort_values('DATE')
            # YoY retail sales percentage change (monthly).
            self.retail_df['retail_yoy'] = (
                self.retail_df['RSFSXMV'].pct_change(periods=12) * 100
            ).round(4)
            self.has_retail = True
        except Exception:
            print("Warning: retail sales data not found. Running without macro feature.")
            self.has_retail = False
            self.retail_df = pd.DataFrame()

        self.models = {}
        self.correlations = {}
        # LSEG client is optional; if unavailable, we'll fall back gracefully.
        self.consensus_client = LSEGConsensusClient() if LSEGConsensusClient else None
        # Debug flag to print feature inputs (set PREDICTOR_DEBUG=true).
        self.debug = os.getenv("PREDICTOR_DEBUG", "false").lower() == "true"

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
        # Uppercase for case-insensitive matching
        brand_upper = brand.upper()
        mapping = {
            "STARBUCKS (MERCHANT)": "STARBUCKS",
            "STARBUCKS CARD": "STARBUCKS",
            "STARBUCKS": "STARBUCKS",
            "CHIPOTLE MEXICAN": "CHIPOTLE",
            "CHIPOTLE": "CHIPOTLE",
            "DOMINO'S PIZZA": "DOMINO'S",
            "DOMINO'S": "DOMINO'S",
            "DUNKIN' DONUTS": "DUNKIN",
            "DUNKIN'": "DUNKIN",
            "DUNKIN": "DUNKIN",
            "MCDONALD'S": "MCDONALD'S",
            "BURGER KING": "BURGER KING",
        }
        return mapping.get(brand_upper, brand_upper)  # Return uppercase if no mapping

    def _get_quarterly_retail_yoy(self, quarter: str, brand: str) -> Optional[float]:
        """Get average retail sales YoY for a fiscal quarter."""
        if not self.has_retail or self.retail_df.empty:
            return None

        try:
            year, q_label = quarter.split('_')
            year = int(year)
            q_num = int(q_label[1])
        except Exception:
            return None

        if brand == "STARBUCKS":
            # Starbucks fiscal: Q1=Oct-Dec, Q2=Jan-Mar, Q3=Apr-Jun, Q4=Jul-Sep
            fiscal_to_months = {
                1: [(year - 1, 10), (year - 1, 11), (year - 1, 12)],
                2: [(year, 1), (year, 2), (year, 3)],
                3: [(year, 4), (year, 5), (year, 6)],
                4: [(year, 7), (year, 8), (year, 9)],
            }
        else:
            # Standard calendar quarters
            fiscal_to_months = {
                1: [(year, 1), (year, 2), (year, 3)],
                2: [(year, 4), (year, 5), (year, 6)],
                3: [(year, 7), (year, 8), (year, 9)],
                4: [(year, 10), (year, 11), (year, 12)],
            }

        months = fiscal_to_months.get(q_num, [])
        if not months:
            return None

        retail_subset = self.retail_df[
            self.retail_df['DATE'].apply(lambda d: (d.year, d.month) in months)
        ]

        if retail_subset.empty or retail_subset['retail_yoy'].isna().all():
            return None

        return float(retail_subset['retail_yoy'].mean())
    
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

                # Add macro retail YoY if we have data for this quarter.
                retail_yoy = self._get_quarterly_retail_yoy(q, brand)
                if retail_yoy is not None:
                    record['retail_yoy'] = retail_yoy
                train_data.append(record)

        if len(train_data) < 3:
            return (0.0, 0.0)

        df = pd.DataFrame(train_data)
        if self.debug and 'retail_yoy' in df.columns:
            print(f"[debug] {brand} retail_yoy by quarter:")
            print(df[['quarter', 'retail_yoy']].to_string(index=False))

        # Feature selection: prefer full feature set, fallback gracefully.
        # Only include a feature if ALL records have valid (non-NaN) values.
        has_valid_visits = 'visits' in df.columns and df['visits'].notna().all()
        has_valid_retail = 'retail_yoy' in df.columns and df['retail_yoy'].notna().all()

        if has_valid_visits and has_valid_retail:
            features = ['spend', 'visits', 'retail_yoy']
        elif has_valid_visits:
            features = ['spend', 'visits']
        elif has_valid_retail:
            features = ['spend', 'transactions', 'retail_yoy']
        else:
            features = ['spend', 'transactions']
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
            'visits': 'total_visits_sum',
            'retail_yoy': 'retail_yoy',
        }

        features = self.models[brand]['features']
        # Inject retail YoY if the model expects it.
        if 'retail_yoy' in features:
            retail_yoy = self._get_quarterly_retail_yoy(quarter, brand)
            if retail_yoy is None:
                return {'error': f'Missing retail_yoy data for {quarter}'}
            current_q = current_q.copy()
            current_q['retail_yoy'] = retail_yoy

        actual_cols = [feature_map.get(f, f) for f in features]

        # Check columns exist
        missing = [c for c in actual_cols if c not in current_q.index]
        if missing:
            return {'error': f'Missing columns {missing} for {brand}'}

        X = current_q[actual_cols].values.reshape(1, -1)
        predicted = self.models[brand]['model'].predict(X)[0]

        # --- Consensus lookup flow ---
        # 1) Try live LSEG SmartEstimate (or cached if live fails)
        # 2) If stale or missing, fall back to historical growth estimate
        # 3) If no history, use the model prediction as the consensus
        consensus_data = None
        is_stale = False
        if self.consensus_client:
            consensus_data = self.consensus_client.get_consensus(brand, quarter)
            is_stale = bool(consensus_data and consensus_data.get("is_stale"))
            if is_stale:
                consensus_data = None

        consensus = None
        if consensus_data:
            consensus = (
                consensus_data.get("revenue_smart_estimate")
                or consensus_data.get("revenue_mean")
            )

        if consensus is None:
            consensus, consensus_source = self._fallback_consensus_estimate(brand, quarter, predicted)
            consensus_data = consensus_data or {}
            consensus_data["source"] = consensus_source

        delta_pct = ((predicted - consensus) / consensus) * 100  # Calculate percentage difference

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
            'wall_street_consensus': consensus,  # Compatible Key
            'delta_pct': round(delta_pct, 2),
            'delta_direction': 'BEAT' if delta_pct > 0 else 'MISS',
            'signal_strength': signal_strength,
            'correlation': round(correlation, 3),
            'alt_data_spend': current_q['spend_sum'],
            'days_in_quarter': current_q['date_count'],
            'consensus_source': consensus_data.get('source') if consensus_data else 'model_prediction',
            'consensus_revenue_mean': consensus_data.get('revenue_mean') if consensus_data else None,
            'consensus_revenue_high': consensus_data.get('revenue_high') if consensus_data else None,
            'consensus_revenue_low': consensus_data.get('revenue_low') if consensus_data else None,
            'consensus_eps_smart': consensus_data.get('eps_smart_estimate') if consensus_data else None,
            'consensus_num_analysts': consensus_data.get('num_analysts') if consensus_data else None,
            'consensus_earnings_date': consensus_data.get('earnings_date') if consensus_data else None,
            'consensus_fetched_at': consensus_data.get('fetched_at') if consensus_data else None,
            'consensus_is_stale': is_stale,
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

    def _fallback_consensus_estimate(self, brand: str, quarter: str, predicted: float) -> Tuple[float, str]:
        """
        Estimate a consensus value when live/cached LSEG is unavailable.
        Uses historical YoY growth for the same quarter when possible.
        """
        history = HISTORICAL_REVENUE.get(brand)
        if not history:
            return predicted, "model_prediction"

        # Extract all values for the same fiscal quarter label (e.g., Q1).
        try:
            _, q_label = quarter.split("_")
        except ValueError:
            q_label = None

        quarter_history = {}
        for key, value in history.items():
            if q_label and key.endswith(f"_{q_label}"):
                year = int(key.split("_")[0])
                quarter_history[year] = value

        if len(quarter_history) < 2:
            # Not enough history for YoY growth; fall back to the latest actual.
            latest_key = sorted(history.keys())[-1]
            return history[latest_key], "historical_actual"

        # Compute average YoY growth for that quarter.
        years = sorted(quarter_history.keys())
        growth_rates = []
        for year in years[1:]:
            prev = quarter_history.get(year - 1)
            if prev:
                growth_rates.append(quarter_history[year] / prev - 1)

        avg_growth = float(np.mean(growth_rates)) if growth_rates else 0.0
        base_year = years[-1]
        estimate = quarter_history[base_year] * (1 + avg_growth)
        return estimate, "historical_growth"

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
        features = predictor.models.get(brand, {}).get('features', [])
        print(f"Features: {features}")
        print(f"R²: {r2:.3f}, Correlation: {corr:.3f}")

        signal = predictor.predict_revenue(brand)
        if 'error' not in signal:
            print(f"Predicted: ${signal['predicted_revenue']/1e9:.2f}B")
            print(f"Consensus: ${signal['wall_street_consensus']/1e9:.2f}B")
            print(f"Delta: {signal['delta_pct']:+.2f}%")
