"""
Traffic Congestion Forecasting System - Database Integration with Model Saving
Uses your test.db SQLite database with crash_data, snow, and jam_data_2017_2025 tables
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATABASE DATA LOADER
# ============================================================================

class DatabaseLoader:
    """Load and preprocess data from SQLite database"""
    
    def __init__(self, db_path='test.db'):
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"✓ Connected to database: {self.db_path}")
            return True
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def load_jam_data(self, start_date=None, end_date=None):
        """
        Load traffic jam data from jam_data_2017_2025 table
        Filters out -999 values
        """
        query = """
        SELECT 
            tmc,
            datetime,
            jam_factor,
            length,
            DISTRICT,
            MAINT_LOCA,
            SUBAREA_LO,
            ORG_CODE,
            road_type
        FROM jam_data_2017_2025
        WHERE 
            length != -999.0
            AND DISTRICT != '-999'
            AND MAINT_LOCA != '-999'
            AND SUBAREA_LO != '-999'
            AND ORG_CODE != '-999'
        """
        
        if start_date:
            query += f" AND datetime >= '{start_date}'"
        if end_date:
            query += f" AND datetime <= '{end_date}'"
        
        query += " ORDER BY datetime"
        
        print("Loading jam data...")
        df = pd.read_sql_query(query, self.conn)
        
        # Convert datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Clean numeric columns
        df['jam_factor'] = pd.to_numeric(df['jam_factor'], errors='coerce')
        df['length'] = pd.to_numeric(df['length'], errors='coerce')
        
        # Remove any remaining nulls
        df = df.dropna(subset=['jam_factor', 'length'])
        
        print(f"✓ Loaded {len(df):,} jam records")
        print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"  Unique locations (TMC): {df['tmc'].nunique()}")
        print(f"  Districts: {df['DISTRICT'].unique()}")
        
        return df
    
    def load_crash_data(self, start_date=None, end_date=None):
        """Load accident/crash data from crash_data table"""
        query = """
        SELECT 
            accident_severity,
            datetime,
            DISTRICT,
            MAINT_LOCA,
            SUBAREA_LO,
            ORG_CODE,
            crash_cost
        FROM crash_data
        WHERE datetime IS NOT NULL
        """
        
        if start_date:
            query += f" AND datetime >= '{start_date}'"
        if end_date:
            query += f" AND datetime <= '{end_date}'"
        
        query += " ORDER BY datetime"
        
        print("Loading crash data...")
        df = pd.read_sql_query(query, self.conn)
        
        # Convert datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Clean crash_cost
        df['crash_cost'] = pd.to_numeric(df['crash_cost'], errors='coerce')
        
        print(f"✓ Loaded {len(df):,} crash records")
        print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"  Severity distribution:")
        print(df['accident_severity'].value_counts().to_string())
        
        return df
    
    def load_snow_data(self, start_date=None, end_date=None):
        """Load snow depth data from snow table"""
        query = """
        SELECT 
            datetime,
            snow_depth,
            ORG_CODE,
            DISTRICT
        FROM snow
        WHERE datetime IS NOT NULL
        """
        
        if start_date:
            query += f" AND datetime >= '{start_date}'"
        if end_date:
            query += f" AND datetime <= '{end_date}'"
        
        query += " ORDER BY datetime"
        
        print("Loading snow data...")
        df = pd.read_sql_query(query, self.conn)
        
        # Convert datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Clean snow_depth
        df['snow_depth'] = pd.to_numeric(df['snow_depth'], errors='coerce')
        
        print(f"✓ Loaded {len(df):,} snow records")
        print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"  Average snow depth: {df['snow_depth'].mean():.2f}")
        
        return df
    
    def get_data_summary(self):
        """Get summary statistics of all tables"""
        summary = {}
        
        # Jam data summary
        jam_query = "SELECT COUNT(*) as count, MIN(datetime) as min_date, MAX(datetime) as max_date FROM jam_data_2017_2025"
        summary['jam_data'] = pd.read_sql_query(jam_query, self.conn).iloc[0].to_dict()
        
        # Crash data summary
        crash_query = "SELECT COUNT(*) as count, MIN(datetime) as min_date, MAX(datetime) as max_date FROM crash_data"
        summary['crash_data'] = pd.read_sql_query(crash_query, self.conn).iloc[0].to_dict()
        
        # Snow data summary
        snow_query = "SELECT COUNT(*) as count, MIN(datetime) as min_date, MAX(datetime) as max_date FROM snow"
        summary['snow_data'] = pd.read_sql_query(snow_query, self.conn).iloc[0].to_dict()
        
        return summary


# ============================================================================
# 2. FEATURE ENGINEERING FOR YOUR DATA
# ============================================================================

class TrafficFeatureEngineer:
    """Create features specific to your traffic data"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
    
    def create_temporal_features(self, df, datetime_col='datetime'):
        """Extract time-based features"""
        df = df.copy()
        
        # Basic time features
        df['hour'] = df[datetime_col].dt.hour
        df['day_of_week'] = df[datetime_col].dt.dayofweek
        df['day_of_month'] = df[datetime_col].dt.day
        df['month'] = df[datetime_col].dt.month
        df['year'] = df[datetime_col].dt.year
        df['week_of_year'] = df[datetime_col].dt.isocalendar().week
        df['quarter'] = df[datetime_col].dt.quarter
        
        # Boolean features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_evening_rush'] = ((df['hour'] >= 16) & (df['hour'] <= 18)).astype(int)
        df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Season
        df['season'] = df['month'].apply(lambda x: 
            'winter' if x in [12, 1, 2] else
            'spring' if x in [3, 4, 5] else
            'summer' if x in [6, 7, 8] else 'fall'
        )
        
        return df
    
    def create_lag_features(self, df, target_col='jam_factor', group_col='tmc', lags=[1, 2, 3, 6, 12, 24]):
        """Create lagged features for time series by location"""
        df = df.copy()
        df = df.sort_values(['tmc', 'datetime'])
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df.groupby(group_col)[target_col].shift(lag)
        
        # Rolling statistics by location
        for window in [3, 6, 12, 24]:
            df[f'{target_col}_rolling_mean_{window}'] = df.groupby(group_col)[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'{target_col}_rolling_std_{window}'] = df.groupby(group_col)[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            df[f'{target_col}_rolling_max_{window}'] = df.groupby(group_col)[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).max()
            )
        
        return df
    
    def merge_crash_features(self, jam_df, crash_df, time_window_hours=24):
        """Merge crash data as features"""
        jam_df = jam_df.copy()
        jam_df = jam_df.reset_index(drop=True)
        
        jam_df['crashes_24h'] = 0
        jam_df['fatal_crashes_24h'] = 0
        jam_df['total_crash_cost_24h'] = 0
        
        print(f"Merging crash data (time window: {time_window_hours}h)...")
        
        crash_df = crash_df.copy()
        crash_df['date'] = crash_df['datetime'].dt.date
        
        crash_summary = crash_df.groupby(['DISTRICT', 'date']).agg({
            'accident_severity': 'count',
            'crash_cost': 'sum'
        }).rename(columns={'accident_severity': 'crash_count'}).reset_index()
        
        fatal_crashes = crash_df[crash_df['accident_severity'] == 'FATAL'].groupby(['DISTRICT', 'date']).size().reset_index(name='fatal_count')
        crash_summary = crash_summary.merge(fatal_crashes, on=['DISTRICT', 'date'], how='left')
        crash_summary['fatal_count'] = crash_summary['fatal_count'].fillna(0)
        
        jam_df['date'] = jam_df['datetime'].dt.date
        jam_df = jam_df.merge(
            crash_summary,
            on=['DISTRICT', 'date'],
            how='left'
        )
        
        jam_df['crashes_24h'] = jam_df['crash_count'].fillna(0)
        jam_df['fatal_crashes_24h'] = jam_df['fatal_count'].fillna(0)
        jam_df['total_crash_cost_24h'] = jam_df['crash_cost'].fillna(0)
        
        jam_df = jam_df.drop(['date', 'crash_count', 'fatal_count', 'crash_cost'], axis=1, errors='ignore')
        
        print(f"✓ Crash features merged")
        print(f"  Average crashes per record: {jam_df['crashes_24h'].mean():.2f}")
        
        return jam_df
    
    def merge_snow_features(self, jam_df, snow_df):
        """Merge snow depth data"""
        jam_df = jam_df.copy()
        jam_df = jam_df.reset_index(drop=True)
        snow_df = snow_df.copy()
        
        jam_df['datetime_hour'] = jam_df['datetime'].dt.floor('H')
        snow_df['datetime_hour'] = snow_df['datetime'].dt.floor('H')
        
        merged = jam_df.merge(
            snow_df[['datetime_hour', 'DISTRICT', 'snow_depth']],
            on=['datetime_hour', 'DISTRICT'],
            how='left',
            suffixes=('', '_snow')
        )
        
        merged['snow_depth'] = merged['snow_depth'].fillna(0)
        
        merged['has_snow'] = (merged['snow_depth'] > 0).astype(int)
        merged['heavy_snow'] = (merged['snow_depth'] > 5).astype(int)
        
        merged = merged.drop('datetime_hour', axis=1)
        
        print(f"✓ Snow features merged")
        print(f"  Records with snow: {merged['has_snow'].sum():,} ({merged['has_snow'].mean()*100:.1f}%)")
        
        return merged
    
    def encode_categorical_features(self, df, categorical_cols=None):
        """Encode categorical variables"""
        if categorical_cols is None:
            categorical_cols = ['DISTRICT', 'MAINT_LOCA', 'SUBAREA_LO', 'ORG_CODE', 'road_type', 'season']
        
        df = df.copy()
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[f'{col}_encoded'] = df[col].apply(
                        lambda x: self.label_encoders[col].transform([str(x)])[0] 
                        if str(x) in self.label_encoders[col].classes_ else -1
                    )
        
        return df
    
    def create_congestion_target(self, df):
        """Create target variable for congestion prediction"""
        df = df.copy()
        
        def categorize_congestion(jam_factor):
            if jam_factor < 3:
                return 0
            elif jam_factor < 5:
                return 1
            elif jam_factor < 7:
                return 2
            else:
                return 3
        
        df['congestion_level'] = df['jam_factor'].apply(categorize_congestion)
        df['is_congested'] = (df['jam_factor'] >= 5).astype(int)
        
        return df
    
    def save(self, filepath):
        """Save the feature engineer state"""
        state = {
            'label_encoders': {}
        }
        
        # Save label encoders
        for col, encoder in self.label_encoders.items():
            state['label_encoders'][col] = {
                'classes': encoder.classes_.tolist()
            }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"✓ Feature engineer saved to: {filepath}")
    
    def load(self, filepath):
        """Load the feature engineer state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Restore label encoders
        for col, encoder_data in state['label_encoders'].items():
            encoder = LabelEncoder()
            encoder.classes_ = np.array(encoder_data['classes'])
            self.label_encoders[col] = encoder
        
        print(f"✓ Feature engineer loaded from: {filepath}")


# ============================================================================
# 3. FORECASTING MODEL
# ============================================================================

class CongestionForecaster:
    """Traffic congestion forecasting model"""
    
    def __init__(self, model_type='gradient_boosting'):
        self.model_type = model_type
        self.model = None
        self.feature_cols = None
        self.scaler = StandardScaler()
        self.metadata = {}
        
    def initialize_model(self):
        """Initialize ML model"""
        if self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                min_samples_split=10,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
        
    def select_features(self, df):
        """Select relevant features for modeling"""
        time_features = [
            'hour', 'day_of_week', 'month', 'week_of_year', 'quarter',
            'is_weekend', 'is_rush_hour', 'is_morning_rush', 'is_evening_rush',
            'is_night', 'is_business_hours',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
        ]
        
        location_features = [
            'DISTRICT_encoded', 'MAINT_LOCA_encoded', 'SUBAREA_LO_encoded',
            'ORG_CODE_encoded', 'road_type_encoded', 'season_encoded'
        ]
        
        traffic_features = ['length']
        weather_features = ['snow_depth', 'has_snow', 'heavy_snow']
        crash_features = ['crashes_24h', 'fatal_crashes_24h', 'total_crash_cost_24h']
        lag_features = [col for col in df.columns if 'lag' in col or 'rolling' in col]
        
        all_features = (time_features + location_features + traffic_features + 
                       weather_features + crash_features + lag_features)
        
        self.feature_cols = [col for col in all_features if col in df.columns]
        
        return self.feature_cols
    
    def train(self, df, target_col='jam_factor', test_size=0.2):
        """Train the forecasting model"""
        print("\n" + "="*70)
        print("TRAINING FORECASTING MODEL")
        print("="*70)
        
        df_clean = df.dropna(subset=[target_col])
        feature_cols = self.select_features(df_clean)
        print(f"\nUsing {len(feature_cols)} features")
        
        X = df_clean[feature_cols].fillna(0)
        y = df_clean[target_col]
        
        split_idx = int(len(df_clean) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training samples: {len(X_train):,}")
        print(f"Testing samples: {len(X_test):,}")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nTraining {self.model_type} model...")
        self.initialize_model()
        self.model.fit(X_train_scaled, y_train)
        
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        # Store metadata
        self.metadata = {
            'train_date': datetime.now().isoformat(),
            'model_type': self.model_type,
            'n_features': len(feature_cols),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'test_r2': r2_score(y_test, test_pred)
        }
        
        print("\n=== MODEL PERFORMANCE ===")
        print(f"\nTraining Set:")
        print(f"  MAE:  {self.metadata['train_mae']:.3f}")
        print(f"  RMSE: {self.metadata['train_rmse']:.3f}")
        print(f"  R²:   {self.metadata['train_r2']:.3f}")
        
        print(f"\nTest Set:")
        print(f"  MAE:  {self.metadata['test_mae']:.3f}")
        print(f"  RMSE: {self.metadata['test_rmse']:.3f}")
        print(f"  R²:   {self.metadata['test_r2']:.3f}")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n=== TOP 15 IMPORTANT FEATURES ===")
            print(importance_df.head(15).to_string(index=False))
        
        return self
    
    def predict(self, df):
        """Make predictions"""
        X = df[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def save_model(self, model_dir='saved_models'):
        """Save the complete model package"""
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"{self.model_type}_{timestamp}"
        model_path = os.path.join(model_dir, model_name)
        os.makedirs(model_path, exist_ok=True)
        
        # Save model
        model_file = os.path.join(model_path, 'model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler
        scaler_file = os.path.join(model_path, 'scaler.pkl')
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature columns and metadata
        config = {
            'feature_cols': self.feature_cols,
            'model_type': self.model_type,
            'metadata': self.metadata
        }
        config_file = os.path.join(model_path, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✓ Model saved to: {model_path}")
        print(f"  - model.pkl")
        print(f"  - scaler.pkl")
        print(f"  - config.json")
        
        return model_path
    
    def load_model(self, model_path):
        """Load a saved model package"""
        # Load config
        config_file = os.path.join(model_path, 'config.json')
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        self.feature_cols = config['feature_cols']
        self.model_type = config['model_type']
        self.metadata = config['metadata']
        
        # Load model
        model_file = os.path.join(model_path, 'model.pkl')
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler
        scaler_file = os.path.join(model_path, 'scaler.pkl')
        with open(scaler_file, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"✓ Model loaded from: {model_path}")
        print(f"  Model type: {self.model_type}")
        print(f"  Features: {len(self.feature_cols)}")
        print(f"  Train date: {self.metadata.get('train_date', 'Unknown')}")
        print(f"  Test R²: {self.metadata.get('test_r2', 'Unknown')}")
        
        return self

    def forecast_location(self, df, location_filter, hours_ahead=72):
        """Forecast congestion for a specific location"""
        mask = pd.Series([True] * len(df), index=df.index)
        for key, value in location_filter.items():
            if key in df.columns:
                mask &= (df[key] == value)
        
        location_data = df[mask].sort_values('datetime').tail(168)
        
        if len(location_data) == 0:
            print(f"No data found for location: {location_filter}")
            return None
        
        last_timestamp = location_data['datetime'].max()
        future_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1),
            periods=hours_ahead,
            freq='H'
        )
        
        forecast_df = pd.DataFrame({'datetime': future_timestamps})
        
        for key, value in location_filter.items():
            forecast_df[key] = value
        
        static_cols = ['DISTRICT', 'MAINT_LOCA', 'SUBAREA_LO', 'ORG_CODE', 'road_type', 'length']
        for col in static_cols:
            if col in location_data.columns and col not in forecast_df.columns:
                forecast_df[col] = location_data[col].iloc[-1]
        
        fe = TrafficFeatureEngineer()
        forecast_df = fe.create_temporal_features(forecast_df)
        
        try:
            forecast_df = fe.encode_categorical_features(forecast_df)
        except Exception as e:
            print(f"Warning: Could not encode all categorical features: {e}")
            for col in ['DISTRICT', 'MAINT_LOCA', 'SUBAREA_LO', 'ORG_CODE', 'road_type', 'season']:
                if col in forecast_df.columns and f'{col}_encoded' not in forecast_df.columns:
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    if col in location_data.columns:
                        le.fit(location_data[col].astype(str))
                        try:
                            forecast_df[f'{col}_encoded'] = le.transform(forecast_df[col].astype(str))
                        except:
                            forecast_df[f'{col}_encoded'] = 0
                    else:
                        forecast_df[f'{col}_encoded'] = 0
        
        forecast_df['snow_depth'] = 0
        forecast_df['has_snow'] = 0
        forecast_df['heavy_snow'] = 0
        
        forecast_df['crashes_24h'] = location_data['crashes_24h'].mean() if 'crashes_24h' in location_data.columns else 0
        forecast_df['fatal_crashes_24h'] = location_data['fatal_crashes_24h'].mean() if 'fatal_crashes_24h' in location_data.columns else 0
        forecast_df['total_crash_cost_24h'] = location_data['total_crash_cost_24h'].mean() if 'total_crash_cost_24h' in location_data.columns else 0
        
        lag_cols = [col for col in self.feature_cols if 'lag' in col or 'rolling' in col]
        for col in lag_cols:
            if col in location_data.columns:
                forecast_df[col] = location_data[col].iloc[-1]
            else:
                forecast_df[col] = 0
        
        try:
            predictions = self.predict(forecast_df)
            forecast_df['predicted_jam_factor'] = predictions
        except Exception as e:
            print(f"Warning: Prediction error: {e}")
            print("Using fallback predictions based on time patterns...")
            hourly_avg = location_data.groupby('hour')['jam_factor'].mean().to_dict()
            forecast_df['predicted_jam_factor'] = forecast_df['hour'].map(
                lambda h: hourly_avg.get(h, location_data['jam_factor'].mean())
            )
        
        forecast_df['predicted_congestion_level'] = forecast_df['predicted_jam_factor'].apply(
            lambda x: 'Free Flow' if x < 3 else
                     'Light' if x < 5 else
                     'Moderate' if x < 7 else 'Heavy'
        )
        
        return forecast_df[['datetime', 'predicted_jam_factor', 'predicted_congestion_level']]


# ============================================================================
# 4. MAIN EXECUTION PIPELINE
# ============================================================================

def main_pipeline(db_path='test.db', sample_size=None, save_models=True):
    """Complete pipeline using your database"""
    
    print("="*70)
    print("TRAFFIC CONGESTION FORECASTING SYSTEM")
    print("Database: Missouri Traffic Data")
    print("="*70)
    
    # Step 1: Load Data
    print("\n[1/6] LOADING DATA FROM DATABASE...")
    print("-"*70)
    
    loader = DatabaseLoader(db_path)
    if not loader.connect():
        return None
    
    summary = loader.get_data_summary()
    print("\nDatabase Summary:")
    for table, stats in summary.items():
        print(f"  {table}: {stats['count']:,} records ({stats['min_date']} to {stats['max_date']})")
    
    jam_data = loader.load_jam_data()
    crash_data = loader.load_crash_data()
    snow_data = loader.load_snow_data()
    
    loader.close()
    
    if sample_size and len(jam_data) > sample_size:
        print(f"\nSampling {sample_size:,} records for faster processing...")
        jam_data = jam_data.sample(sample_size, random_state=42).sort_values('datetime')
    
    # Step 2: Feature Engineering
    print("\n[2/6] FEATURE ENGINEERING...")
    print("-"*70)
    
    fe = TrafficFeatureEngineer()
    
    print("Creating temporal features...")
    jam_data = fe.create_temporal_features(jam_data)
    
    print("Creating target variable...")
    jam_data = fe.create_congestion_target(jam_data)
    
    print("Encoding categorical features...")
    jam_data = fe.encode_categorical_features(jam_data)
    
    print("Merging snow data...")
    jam_data = fe.merge_snow_features(jam_data, snow_data)
    
    print("Merging crash data...")
    try:
        sample_for_crashes = jam_data.sample(min(50000, len(jam_data)), random_state=42).reset_index(drop=True)
        sample_for_crashes = fe.merge_crash_features(sample_for_crashes, crash_data, time_window_hours=24)
        
        crash_cols = ['crashes_24h', 'fatal_crashes_24h', 'total_crash_cost_24h']
        for col in crash_cols:
            if col not in jam_data.columns:
                jam_data[col] = 0
        
        if len(sample_for_crashes) < len(jam_data):
            jam_data.loc[sample_for_crashes.index, crash_cols] = sample_for_crashes[crash_cols].values
        else:
            jam_data = sample_for_crashes
            
    except Exception as e:
        print(f"Warning: Could not merge crash data: {e}")
        print("Continuing with zero crash features...")
        jam_data['crashes_24h'] = 0
        jam_data['fatal_crashes_24h'] = 0
        jam_data['total_crash_cost_24h'] = 0
    
    print("Creating lag features (this may take a moment)...")
    jam_data = fe.create_lag_features(jam_data, target_col='jam_factor', group_col='tmc')
    
    jam_data = jam_data.dropna(subset=['jam_factor_lag_1'])
    
    print(f"\n✓ Final feature matrix: {jam_data.shape}")
    print(f"  Features: {jam_data.shape[1]}")
    print(f"  Records: {jam_data.shape[0]:,}")
    
    # Step 3: Train Model
    print("\n[3/6] MODEL TRAINING...")
    print("-"*70)
    
    forecaster = CongestionForecaster(model_type='gradient_boosting')
    forecaster.train(jam_data, target_col='jam_factor', test_size=0.2)
    
    # Save models if requested
    if save_models:
        print("\n[3.5/6] SAVING MODELS...")
        print("-"*70)
        
        model_path = forecaster.save_model()
        
        # Save feature engineer
        fe_path = os.path.join(os.path.dirname(model_path), 'feature_engineer.pkl')
        fe.save(fe_path)
    
    # Step 4: Generate Forecasts for Sample Locations
    print("\n[4/6] GENERATING FORECASTS...")
    print("-"*70)
    
    top_locations = jam_data.groupby('tmc')['jam_factor'].mean().nlargest(5)
    
    print(f"\nForecasting for top 5 congested locations:")
    print(top_locations.to_string())
    
    forecasts = {}
    for tmc in top_locations.index[:3]:
        location_info = jam_data[jam_data['tmc'] == tmc].iloc[-1]
        location_filter = {
            'tmc': tmc,
            'DISTRICT': location_info['DISTRICT']
        }
        
        print(f"\n→ Forecasting for TMC: {tmc} ({location_info['DISTRICT']} - {location_info['MAINT_LOCA']})")
        forecast = forecaster.forecast_location(jam_data, location_filter, hours_ahead=72)
        
        if forecast is not None:
            forecasts[tmc] = forecast
            
            print(f"  ✓ 72-hour forecast generated")
            print(f"  Average predicted jam factor: {forecast['predicted_jam_factor'].mean():.2f}")
            print(f"  Congestion distribution:")
            print(forecast['predicted_congestion_level'].value_counts().to_string())
            
            worst = forecast.nlargest(5, 'predicted_jam_factor')
            print(f"\n  ⚠️  Top 5 worst congestion periods:")
            for _, row in worst.iterrows():
                print(f"    • {row['datetime']:%Y-%m-%d %H:%M} - Jam Factor: {row['predicted_jam_factor']:.2f} ({row['predicted_congestion_level']})")
    
    # Step 5: Save Results
    print("\n[5/6] SAVING RESULTS...")
    print("-"*70)
    
    for tmc, forecast in forecasts.items():
        filename = f"forecast_{tmc.replace('+', '_')}_{datetime.now():%Y%m%d}.csv"
        forecast.to_csv(filename, index=False)
        print(f"✓ Saved: {filename}")
    
    report_data = {
        'timestamp': [datetime.now()],
        'total_records': [len(jam_data)],
        'features_used': [len(forecaster.feature_cols)],
        'locations_forecasted': [len(forecasts)],
        'model_type': [forecaster.model_type],
        'test_r2': [forecaster.metadata.get('test_r2', 0)],
        'test_mae': [forecaster.metadata.get('test_mae', 0)]
    }
    report_df = pd.DataFrame(report_data)
    report_df.to_csv('model_report.csv', index=False)
    print(f"✓ Saved: model_report.csv")
    
    # Step 6: Analysis and Insights
    print("\n[6/6] ANALYSIS & INSIGHTS...")
    print("-"*70)
    
    print("\n📊 OVERALL TRAFFIC PATTERNS:")
    print(f"  Average jam factor: {jam_data['jam_factor'].mean():.2f}")
    print(f"  Std deviation: {jam_data['jam_factor'].std():.2f}")
    print(f"  Max jam factor: {jam_data['jam_factor'].max():.2f}")
    
    print("\n🚦 RUSH HOUR ANALYSIS:")
    rush_hour_avg = jam_data[jam_data['is_rush_hour'] == 1]['jam_factor'].mean()
    non_rush_avg = jam_data[jam_data['is_rush_hour'] == 0]['jam_factor'].mean()
    print(f"  Rush hour avg: {rush_hour_avg:.2f}")
    print(f"  Non-rush hour avg: {non_rush_avg:.2f}")
    print(f"  Difference: {rush_hour_avg - non_rush_avg:.2f} ({(rush_hour_avg/non_rush_avg - 1)*100:.1f}% worse)")
    
    if 'has_snow' in jam_data.columns:
        print("\n❄️  WEATHER IMPACT:")
        snow_avg = jam_data[jam_data['has_snow'] == 1]['jam_factor'].mean()
        no_snow_avg = jam_data[jam_data['has_snow'] == 0]['jam_factor'].mean()
        print(f"  With snow: {snow_avg:.2f}")
        print(f"  Without snow: {no_snow_avg:.2f}")
        print(f"  Snow impact: {snow_avg - no_snow_avg:.2f} ({(snow_avg/no_snow_avg - 1)*100:.1f}% worse)")
    
    if 'crashes_24h' in jam_data.columns:
        print("\n🚨 CRASH IMPACT:")
        with_crashes = jam_data[jam_data['crashes_24h'] > 0]['jam_factor'].mean()
        no_crashes = jam_data[jam_data['crashes_24h'] == 0]['jam_factor'].mean()
        print(f"  With recent crashes: {with_crashes:.2f}")
        print(f"  Without crashes: {no_crashes:.2f}")
        print(f"  Crash impact: {with_crashes - no_crashes:.2f} ({(with_crashes/no_crashes - 1)*100:.1f}% worse)")
    
    print("\n🗺️  TOP 5 DISTRICTS BY CONGESTION:")
    district_avg = jam_data.groupby('DISTRICT')['jam_factor'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    print(district_avg.head().to_string())
    
    print("\n⏰ CONGESTION BY HOUR OF DAY:")
    hourly_avg = jam_data.groupby('hour')['jam_factor'].mean().sort_values(ascending=False)
    print("  Worst hours:")
    print(hourly_avg.head(5).to_string())
    print("  Best hours:")
    print(hourly_avg.tail(5).to_string())
    
    print("\n" + "="*70)
    print("✨ FORECASTING COMPLETE!")
    print("="*70)
    
    return forecaster, forecasts, jam_data


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_saved_model(model_path):
    """
    Load a previously saved model
    
    Usage:
        forecaster = load_saved_model('saved_models/gradient_boosting_20250101_120000')
    """
    forecaster = CongestionForecaster()
    forecaster.load_model(model_path)
    return forecaster


def quick_forecast(db_path='test.db', tmc=None, district=None, hours=72):
    """
    Quick forecast for a specific location
    
    Usage:
        quick_forecast('test.db', tmc='119+04099', hours=72)
    """
    print("Loading data and model...")
    
    loader = DatabaseLoader(db_path)
    loader.connect()
    
    jam_data = loader.load_jam_data()
    
    if tmc:
        jam_data = jam_data[jam_data['tmc'] == tmc]
    if district:
        jam_data = jam_data[jam_data['DISTRICT'] == district]
    
    if len(jam_data) == 0:
        print("No data found for specified location")
        return None
    
    print(f"Found {len(jam_data):,} records")
    
    fe = TrafficFeatureEngineer()
    jam_data = fe.create_temporal_features(jam_data)
    jam_data = fe.create_congestion_target(jam_data)
    jam_data = fe.encode_categorical_features(jam_data)
    
    jam_data['snow_depth'] = 0
    jam_data['has_snow'] = 0
    jam_data['heavy_snow'] = 0
    jam_data['crashes_24h'] = 0
    jam_data['fatal_crashes_24h'] = 0
    jam_data['total_crash_cost_24h'] = 0
    
    jam_data = fe.create_lag_features(jam_data, target_col='jam_factor', group_col='tmc')
    jam_data = jam_data.dropna(subset=['jam_factor_lag_1'])
    
    print("Training model...")
    forecaster = CongestionForecaster()
    forecaster.train(jam_data, test_size=0.2)
    
    location_info = jam_data.iloc[-1]
    location_filter = {'tmc': location_info['tmc'], 'DISTRICT': location_info['DISTRICT']}
    
    forecast = forecaster.forecast_location(jam_data, location_filter, hours_ahead=hours)
    
    print(f"\n✓ Forecast generated for {hours} hours")
    print(forecast.head(10))
    
    loader.close()
    return forecast


def analyze_location_patterns(db_path='test.db', tmc=None, save_plot=False):
    """
    Analyze historical patterns for a location
    
    Usage:
        analyze_location_patterns('test.db', tmc='119+04099')
    """
    loader = DatabaseLoader(db_path)
    loader.connect()
    
    jam_data = loader.load_jam_data()
    
    if tmc:
        jam_data = jam_data[jam_data['tmc'] == tmc]
        print(f"Analyzing TMC: {tmc}")
    
    if len(jam_data) == 0:
        print("No data found")
        return
    
    fe = TrafficFeatureEngineer()
    jam_data = fe.create_temporal_features(jam_data)
    
    print("\n" + "="*70)
    print(f"LOCATION ANALYSIS: {jam_data['tmc'].iloc[0]}")
    print("="*70)
    
    print(f"\nData range: {jam_data['datetime'].min()} to {jam_data['datetime'].max()}")
    print(f"Total records: {len(jam_data):,}")
    
    print("\n📊 JAM FACTOR STATISTICS:")
    print(f"  Mean: {jam_data['jam_factor'].mean():.2f}")
    print(f"  Median: {jam_data['jam_factor'].median():.2f}")
    print(f"  Std Dev: {jam_data['jam_factor'].std():.2f}")
    print(f"  Min: {jam_data['jam_factor'].min():.2f}")
    print(f"  Max: {jam_data['jam_factor'].max():.2f}")
    
    print("\n⏰ HOURLY PATTERNS:")
    hourly = jam_data.groupby('hour')['jam_factor'].mean().sort_values(ascending=False)
    print("  Peak hours:")
    print(hourly.head(5).to_string())
    
    print("\n📅 DAILY PATTERNS:")
    daily = jam_data.groupby('day_of_week')['jam_factor'].mean()
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for i, day in enumerate(days):
        if i in daily.index:
            print(f"  {day}: {daily[i]:.2f}")
    
    print("\n📆 MONTHLY PATTERNS:")
    monthly = jam_data.groupby('month')['jam_factor'].mean().sort_values(ascending=False)
    print(monthly.to_string())
    
    loader.close()


def export_forecasts_for_all_districts(db_path='test.db', hours=72):
    """
    Generate and export forecasts for all districts
    
    Usage:
        export_forecasts_for_all_districts('test.db', hours=72)
    """
    print("Loading and preparing data...")
    
    loader = DatabaseLoader(db_path)
    loader.connect()
    
    jam_data = loader.load_jam_data()
    snow_data = loader.load_snow_data()
    
    fe = TrafficFeatureEngineer()
    jam_data = fe.create_temporal_features(jam_data)
    jam_data = fe.create_congestion_target(jam_data)
    jam_data = fe.encode_categorical_features(jam_data)
    jam_data = fe.merge_snow_features(jam_data, snow_data)
    
    jam_data['crashes_24h'] = 0
    jam_data['fatal_crashes_24h'] = 0
    jam_data['total_crash_cost_24h'] = 0
    
    jam_data = fe.create_lag_features(jam_data, target_col='jam_factor', group_col='tmc')
    jam_data = jam_data.dropna(subset=['jam_factor_lag_1'])
    
    print("Training model...")
    forecaster = CongestionForecaster()
    forecaster.train(jam_data, test_size=0.2)
    
    districts = jam_data['DISTRICT'].unique()
    print(f"\nGenerating forecasts for {len(districts)} districts...")
    
    all_forecasts = []
    
    for district in districts:
        if district == '-999':
            continue
        
        district_data = jam_data[jam_data['DISTRICT'] == district]
        top_tmc = district_data.groupby('tmc')['jam_factor'].count().idxmax()
        
        location_info = district_data[district_data['tmc'] == top_tmc].iloc[-1]
        location_filter = {'tmc': top_tmc, 'DISTRICT': district}
        
        print(f"  Forecasting {district} (TMC: {top_tmc})...")
        forecast = forecaster.forecast_location(jam_data, location_filter, hours_ahead=hours)
        
        if forecast is not None:
            forecast['district'] = district
            forecast['tmc'] = top_tmc
            all_forecasts.append(forecast)
    
    combined = pd.concat(all_forecasts, ignore_index=True)
    
    filename = f"all_districts_forecast_{datetime.now():%Y%m%d_%H%M}.csv"
    combined.to_csv(filename, index=False)
    print(f"\n✓ All forecasts saved to: {filename}")
    print(f"  Total forecasts: {len(combined):,} records")
    print(f"  Districts covered: {combined['district'].nunique()}")
    
    loader.close()
    return combined


# ============================================================================
# RUN THE PIPELINE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    db_path = 'test.db'
    
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║     TRAFFIC CONGESTION FORECASTING SYSTEM                         ║
║     Missouri Traffic Data Analysis                                ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\nStarting full pipeline with model saving...")
    print("Note: Using sample of 100,000 records for demonstration.")
    print("Remove sample_size parameter to process all data.\n")
    
    try:
        forecaster, forecasts, data = main_pipeline(
            db_path=db_path,
            sample_size=100000,  # Remove this to process all data
            save_models=True     # Set to False to skip model saving
        )
        
        print("\n" + "="*70)
        print("NEXT STEPS:")
        print("="*70)
        print("""
1. Review the generated forecast CSV files

2. Load a saved model for predictions:
   
   from your_script import load_saved_model
   forecaster = load_saved_model('saved_models/gradient_boosting_YYYYMMDD_HHMMSS')
   # Now use forecaster.predict() or forecaster.forecast_location()

3. Use quick_forecast() for specific locations:
   
   from your_script import quick_forecast
   forecast = quick_forecast('test.db', tmc='119+04099', hours=72)

4. Analyze location patterns:
   
   from your_script import analyze_location_patterns
   analyze_location_patterns('test.db', tmc='119+04099')

5. Generate forecasts for all districts:
   
   from your_script import export_forecasts_for_all_districts
   export_forecasts_for_all_districts('test.db', hours=72)

Models are saved in the 'saved_models' directory with:
  - model.pkl (trained model)
  - scaler.pkl (feature scaler)
  - config.json (model configuration and metadata)
  - feature_engineer.pkl (feature engineering pipeline)
        """)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)