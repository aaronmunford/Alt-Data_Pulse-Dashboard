"""
Revenue Prediction Model
=========================
Uses linear regression to predict quarterly revenue from alternative data signals.
The model learns the relationship between Consumer Edge spend data and actual
reported revenue to generate "live" revenue predictions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# WALL STREET CONSENSUS DATA (Mock - Replace with real FactSet/Bloomberg data)
# =============================================================================
WALL_STREET_CONSENSUS = {
    "STARBUCKS": {
        "2025_Q1": {"revenue_estimate": 9.1e9, "eps_estimate": 0.93},
        "2025_Q2": {"revenue_estimate": 9.4e9, "eps_estimate": 0.98},
        "2024_Q4": {"revenue_estimate": 9.0e9, "eps_estimate": 0.90},
    },
    "MCDONALD'S": {
        "2025_Q1": {"revenue_estimate": 6.2e9, "eps_estimate": 2.70},
        "2025_Q2": {"revenue_estimate": 6.5e9, "eps_estimate": 2.85},
    },
    "CHIPOTLE": {
        "2025_Q1": {"revenue_estimate": 2.8e9, "eps_estimate": 0.32},
        "2025_Q2": {"revenue_estimate": 3.0e9, "eps_estimate": 0.35},
    },
}

# Historical actual revenue (for model training)
# Source: Company 10-Q/10-K filings
HISTORICAL_REVENUE = {
    "STARBUCKS": {
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
    """
    Predicts quarterly revenue using alternative data signals.

    The model:
    1. Aggregates daily spend data to quarterly totals
    2. Trains a linear regression on (alt_data_spend -> actual_revenue)
    3. Applies the learned relationship to current quarter data
    """

    def __init__(self, spend_data_path: str = "data/clean_spend_daily.csv"):
        self.spend_df = pd.read_csv(spend_data_path)
        self.spend_df['date'] = pd.to_datetime(self.spend_df['date'])
        self.models: Dict[str, LinearRegression] = {}
        self.correlations: Dict[str, float] = {}
        self.scalers: Dict[str, Tuple[float, float]] = {}  # (mean, std) for each brand

    def _get_quarter(self, date: datetime) -> str:
        """Convert date to fiscal quarter string."""
        quarter = (date.month - 1) // 3 + 1
        return f"{date.year}_Q{quarter}"

    def _normalize_brand(self, brand: str) -> str:
        """Normalize brand names to match between datasets."""
        brand_mapping = {
            "STARBUCKS (MERCHANT)": "STARBUCKS",
            "STARBUCKS CARD": "STARBUCKS",
            "CHIPOTLE MEXICAN": "CHIPOTLE",
            "DOMINO'S PIZZA": "DOMINO'S",
            "DUNKIN' DONUTS": "DUNKIN",
            "DUNKIN'S DIAMONDS": "DUNKIN",
            "MTA SUBWAY": None,  # Not a QSR
        }
        return brand_mapping.get(brand, brand)

    def aggregate_quarterly(self, brand: str) -> pd.DataFrame:
        """Aggregate daily spend data to quarterly totals."""
        # Filter to brand (handling variations)
        brand_variations = [b for b in self.spend_df['brand'].unique()
                          if self._normalize_brand(b) == brand]

        if not brand_variations:
            return pd.DataFrame()

        df = self.spend_df[self.spend_df['brand'].isin(brand_variations)].copy()
        df['quarter'] = df['date'].apply(self._get_quarter)

        quarterly = df.groupby('quarter').agg({
            'spend': 'sum',
            'transactions': 'sum',
            'avg_ticket_size': 'mean',
            'date': ['min', 'max', 'count']
        }).reset_index()

        quarterly.columns = ['quarter', 'total_spend', 'total_transactions',
                           'avg_ticket_size', 'start_date', 'end_date', 'days_count']

        # Filter to quarters with sufficient data (at least 60 days)
        quarterly = quarterly[quarterly['days_count'] >= 60]

        return quarterly

    def train_model(self, brand: str) -> Tuple[float, float]:
        """
        Train linear regression model for a brand.
        Returns (r2_score, correlation).
        """
        if brand not in HISTORICAL_REVENUE:
            return (0.0, 0.0)

        quarterly = self.aggregate_quarterly(brand)
        if quarterly.empty:
            return (0.0, 0.0)

        # Match with historical revenue
        historical = HISTORICAL_REVENUE[brand]

        train_data = []
        for _, row in quarterly.iterrows():
            q = row['quarter']
            if q in historical:
                train_data.append({
                    'quarter': q,
                    'alt_spend': row['total_spend'],
                    'transactions': row['total_transactions'],
                    'ticket_size': row['avg_ticket_size'],
                    'actual_revenue': historical[q]
                })

        if len(train_data) < 3:
            return (0.0, 0.0)

        train_df = pd.DataFrame(train_data)

        # Feature engineering
        X = train_df[['alt_spend', 'transactions', 'ticket_size']].values
        y = train_df['actual_revenue'].values

        # Store scaling factors
        self.scalers[brand] = (X.mean(axis=0), X.std(axis=0))

        # Train model
        model = LinearRegression()
        model.fit(X, y)
        self.models[brand] = model

        # Calculate metrics
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        correlation = np.corrcoef(train_df['alt_spend'], y)[0, 1]
        self.correlations[brand] = correlation

        return (r2, correlation)

    def predict_revenue(self, brand: str, quarter: Optional[str] = None) -> Dict:
        """
        Predict revenue for current or specified quarter.

        Returns:
            {
                'predicted_revenue': float,
                'wall_street_consensus': float,
                'delta_pct': float,
                'signal_strength': str,
                'correlation': float,
                'confidence_interval': (low, high)
            }
        """
        if brand not in self.models:
            r2, corr = self.train_model(brand)
            if r2 == 0:
                return {'error': f'Insufficient data to train model for {brand}'}

        # Get current quarter data
        if quarter is None:
            today = datetime.now()
            quarter = self._get_quarter(today)

        quarterly = self.aggregate_quarterly(brand)
        current_q = quarterly[quarterly['quarter'] == quarter]

        if current_q.empty:
            # Fall back to latest available quarter
            current_q = quarterly.iloc[[-1]] if not quarterly.empty else None
            if current_q is not None:
                quarter = current_q['quarter'].values[0]

        if current_q is None or current_q.empty:
            return {'error': f'No data available for {brand} in {quarter}'}

        # Prepare features
        X = current_q[['total_spend', 'total_transactions', 'avg_ticket_size']].values

        # Predict
        predicted = self.models[brand].predict(X)[0]

        # Get Wall Street consensus
        consensus_data = WALL_STREET_CONSENSUS.get(brand, {}).get(quarter, {})
        consensus = consensus_data.get('revenue_estimate', predicted)  # Use prediction if no consensus

        # Calculate delta (the trade signal!)
        delta_pct = ((predicted - consensus) / consensus) * 100

        # Determine signal strength
        correlation = self.correlations.get(brand, 0)
        if correlation >= 0.9:
            signal_strength = "Very High"
        elif correlation >= 0.8:
            signal_strength = "High"
        elif correlation >= 0.7:
            signal_strength = "Medium"
        else:
            signal_strength = "Low"

        # Calculate confidence interval (rough estimate based on historical variance)
        std_pct = 0.05  # Assume 5% standard deviation
        ci_low = predicted * (1 - 2 * std_pct)
        ci_high = predicted * (1 + 2 * std_pct)

        return {
            'brand': brand,
            'quarter': quarter,
            'predicted_revenue': predicted,
            'wall_street_consensus': consensus,
            'delta_pct': round(delta_pct, 2),
            'delta_direction': 'BEAT' if delta_pct > 0 else 'MISS',
            'signal_strength': signal_strength,
            'correlation': round(correlation, 3),
            'confidence_interval': (ci_low, ci_high),
            'alt_data_spend': current_q['total_spend'].values[0],
            'days_in_quarter': current_q['days_count'].values[0],
        }

    def get_all_signals(self) -> pd.DataFrame:
        """Generate signals for all tracked brands."""
        results = []

        for brand in ["STARBUCKS", "MCDONALD'S", "CHIPOTLE"]:
            signal = self.predict_revenue(brand)
            if 'error' not in signal:
                results.append(signal)

        return pd.DataFrame(results)

    def get_trend_data(self, brand: str) -> pd.DataFrame:
        """
        Get daily trend data for charting.
        Returns spend and transaction trends with YoY comparison.
        """
        brand_variations = [b for b in self.spend_df['brand'].unique()
                          if self._normalize_brand(b) == brand]

        df = self.spend_df[self.spend_df['brand'].isin(brand_variations)].copy()
        df = df.sort_values('date')

        # Calculate rolling averages
        df['spend_7d_avg'] = df['spend'].rolling(7).mean()
        df['spend_30d_avg'] = df['spend'].rolling(30).mean()
        df['transactions_7d_avg'] = df['transactions'].rolling(7).mean()

        # YoY comparison
        df['spend_ly'] = df['spend'].shift(365)
        df['spend_yoy_pct'] = ((df['spend'] - df['spend_ly']) / df['spend_ly'] * 100).round(2)

        return df


def generate_dashboard_metrics(spend_data_path: str = "data/clean_spend_daily.csv") -> Dict:
    """
    Generate all metrics needed for the dashboard.

    Returns a dictionary with:
    - alpha_metrics: The main signal cards
    - trend_data: Data for the charts
    - insights: Auto-generated insights
    """
    predictor = RevenuePredictor(spend_data_path)

    # Get signals for main brands
    signals = predictor.get_all_signals()

    # Get trend data for charting
    trends = {}
    for brand in ["STARBUCKS", "MCDONALD'S", "CHIPOTLE"]:
        trends[brand] = predictor.get_trend_data(brand)

    # Generate insights
    insights = []
    for _, row in signals.iterrows():
        if row['delta_pct'] > 0:
            insight = f"{row['brand']}: Predicted +{row['delta_pct']}% beat. "
        else:
            insight = f"{row['brand']}: Predicted {row['delta_pct']}% miss. "

        # Add driver insight
        trend_df = trends.get(row['brand'], pd.DataFrame())
        if not trend_df.empty:
            latest = trend_df.iloc[-30:]  # Last 30 days
            if 'spend_yoy_pct' in latest.columns:
                avg_yoy = latest['spend_yoy_pct'].mean()
                insight += f"Spend YoY: {avg_yoy:+.1f}%"

        insights.append(insight)

    return {
        'alpha_metrics': signals.to_dict('records') if not signals.empty else [],
        'trend_data': {k: v.to_dict('records') for k, v in trends.items()},
        'insights': insights,
        'last_updated': datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Test the predictor
    print("=" * 60)
    print("REVENUE PREDICTION MODEL TEST")
    print("=" * 60)

    predictor = RevenuePredictor()

    for brand in ["STARBUCKS", "MCDONALD'S", "CHIPOTLE"]:
        print(f"\n--- {brand} ---")

        # Train model
        r2, corr = predictor.train_model(brand)
        print(f"Model R²: {r2:.3f}")
        print(f"Correlation: {corr:.3f}")

        # Get prediction
        signal = predictor.predict_revenue(brand)
        if 'error' not in signal:
            print(f"\nPredicted Revenue: ${signal['predicted_revenue']/1e9:.2f}B")
            print(f"Wall St Consensus: ${signal['wall_street_consensus']/1e9:.2f}B")
            print(f"Delta: {signal['delta_pct']:+.2f}% ({signal['delta_direction']})")
            print(f"Signal Strength: {signal['signal_strength']}")
        else:
            print(f"Error: {signal['error']}")
