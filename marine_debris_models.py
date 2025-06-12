"""
Marine Debris Classification Models
==================================

Advanced machine learning models and analytics for marine debris classification.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import tensorflow as tf
from pathlib import Path
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from config import *

class MarineDebrisFeatureEngineer:
    """Advanced feature engineering for marine debris data."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.engineered_features = pd.DataFrame()
        
    def create_temporal_features(self) -> pd.DataFrame:
        """Create time-based features."""
        df = self.data.copy()
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day_of_Year'] = df['Date'].dt.dayofyear
            df['Week_of_Year'] = df['Date'].dt.isocalendar().week
            df['Is_Weekend'] = df['Date'].dt.weekday >= 5
            
            # Seasonal encoding
            df['Season_Numeric'] = df['Season'].map({
                'Spring': 1, 'Summer': 2, 'Fall': 3, 'Winter': 4
            })
            
        return df
    
    def create_geographic_features(self) -> pd.DataFrame:
        """Create location-based features."""
        df = self.data.copy()
        
        # Coastal region classification
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            # US coastal regions (simplified)
            conditions = [
                (df['Longitude'] > -90) & (df['Latitude'] > 24),  # Atlantic
                (df['Longitude'] <= -90) & (df['Latitude'] > 25),  # Gulf
                (df['Longitude'] < -110) & (df['Latitude'] > 32),  # Pacific
            ]
            choices = ['Atlantic', 'Gulf', 'Pacific']
            df['Coastal_Region'] = np.select(conditions, choices, default='Other')
            
            # Distance from equator
            df['Distance_from_Equator'] = abs(df['Latitude'])
            
            # Coastal proximity (simplified)
            df['Coastal_Proximity'] = np.random.choice(['Urban', 'Suburban', 'Rural'], len(df))
            
        return df
    
    def create_debris_features(self) -> pd.DataFrame:
        """Create debris-specific features."""
        df = self.data.copy()
        
        # Debris type columns
        debris_cols = [col for col in df.columns if any(debris_type in col.lower() 
                      for debris_type in ['plastic', 'metal', 'glass', 'rubber', 'cloth'])]
        
        if debris_cols:
            # Total debris count
            df['Total_Debris'] = df[debris_cols].sum(axis=1)
            
            # Plastic dominance
            plastic_cols = [col for col in debris_cols if 'plastic' in col.lower()]
            if plastic_cols:
                df['Plastic_Total'] = df[plastic_cols].sum(axis=1)
                df['Plastic_Percentage'] = (df['Plastic_Total'] / df['Total_Debris']).fillna(0)
            
            # Material diversity (number of different materials found)
            df['Material_Diversity'] = (df[debris_cols] > 0).sum(axis=1)
            
            # Debris density (debris per unit area)
            if 'Width' in df.columns and 'Length' in df.columns:
                df['Area'] = df['Width'] * df['Length']
                df['Debris_Density'] = df['Total_Debris'] / df['Area']
        
        return df
    
    def create_environmental_features(self) -> pd.DataFrame:
        """Create environment-based features."""
        df = self.data.copy()
        
        # Weather impact score
        weather_impact = {
            'Sunny': 1, 'Cloudy': 2, 'Windy': 3, 'Rainy': 4
        }
        if 'Weather' in df.columns:
            df['Weather_Impact_Score'] = df['Weather'].map(weather_impact)
        
        # Storm activity score
        storm_impact = {
            'None': 0, 'Low': 1, 'Medium': 2, 'High': 3
        }
        if 'Storm_Activity' in df.columns:
            df['Storm_Impact_Score'] = df['Storm_Activity'].map(storm_impact)
        
        # Combined environmental risk
        if 'Weather_Impact_Score' in df.columns and 'Storm_Impact_Score' in df.columns:
            df['Environmental_Risk'] = df['Weather_Impact_Score'] + df['Storm_Impact_Score']
        
        return df
    
    def create_target_variables(self) -> pd.DataFrame:
        """Create target variables for classification."""
        df = self.data.copy()
        
        # Debris source classification (simplified)
        # This would typically be based on domain knowledge
        if 'Coastal_Proximity' in df.columns:
            df['Debris_Source'] = df['Coastal_Proximity'].map({
                'Urban': 'land_based',
                'Suburban': 'land_based', 
                'Rural': 'ocean_based'
            })
        else:
            # Random assignment for demo
            df['Debris_Source'] = np.random.choice(['land_based', 'ocean_based'], len(df))
        
        # Primary material type
        debris_cols = [col for col in df.columns if any(debris_type in col.lower() 
                      for debris_type in ['plastic', 'metal', 'glass', 'rubber', 'cloth'])]
        
        if debris_cols:
            # Find dominant material
            material_sums = df[debris_cols].sum()
            dominant_materials = []
            
            for idx, row in df.iterrows():
                debris_counts = row[debris_cols]
                if debris_counts.sum() > 0:
                    max_material = debris_counts.idxmax()
                    if 'plastic' in max_material.lower():
                        dominant_materials.append('Plastic')
                    elif 'metal' in max_material.lower():
                        dominant_materials.append('Metal')
                    elif 'glass' in max_material.lower():
                        dominant_materials.append('Glass')
                    elif 'rubber' in max_material.lower():
                        dominant_materials.append('Rubber')
                    elif 'cloth' in max_material.lower():
                        dominant_materials.append('Cloth')
                    else:
                        dominant_materials.append('Other')
                else:
                    dominant_materials.append('Other')
            
            df['Primary_Material'] = dominant_materials
        
        # Pollution severity level
        if 'Total_Debris' in df.columns:
            debris_quantiles = df['Total_Debris'].quantile([0.25, 0.5, 0.75])
            df['Pollution_Level'] = pd.cut(
                df['Total_Debris'],
                bins=[-np.inf, debris_quantiles[0.25], debris_quantiles[0.5], 
                      debris_quantiles[0.75], np.inf],
                labels=['Low', 'Medium', 'High', 'Critical']
            )
                    
        return df
    
    def engineer_all_features(self) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        print("🔧 Starting feature engineering...")
        
        df = self.data.copy()
        df = self.create_temporal_features()
        df = self.create_geographic_features()
        df = self.create_debris_features()
        df = self.create_environmental_features()
        df = self.create_target_variables()
        
        self.engineered_features = df
        print(f"✅ Feature engineering completed. Shape: {df.shape}")
        return df

class MarineDebrisModelTrainer:
    """Train and evaluate machine learning models."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.models = {}
        self.results = {}
        
    def prepare_features_and_targets(self) -> tuple:
        """Prepare feature matrix and target variables."""
        df = self.data.copy()
        
        # Select features (excluding target variables and non-numeric columns)
        exclude_cols = ['Date', 'Debris_Source', 'Primary_Material', 'Pollution_Level',
                       'Country', 'State', 'Organization', 'Shoreline_Name', 'Season',
                       'Weather', 'Storm_Activity', 'Coastal_Region', 'Coastal_Proximity']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
        
        X = df[feature_cols].fillna(0)
        
        # Target variables
        targets = {}
        if 'Debris_Source' in df.columns:
            targets['debris_source'] = df['Debris_Source']
        if 'Primary_Material' in df.columns:
            targets['material_type'] = df['Primary_Material']
        if 'Pollution_Level' in df.columns:
            targets['pollution_level'] = df['Pollution_Level']
            
        return X, targets, feature_cols
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series, 
                           task_name: str) -> RandomForestClassifier:
        """Train Random Forest classifier."""
        print(f"🌲 Training Random Forest for {task_name}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        rf = RandomForestClassifier(**RANDOM_FOREST_CONFIG)
        rf.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store results
        self.models[f'rf_{task_name}'] = rf
        self.results[f'rf_{task_name}'] = {
            'accuracy': accuracy,
            'classification_report': report,
            'feature_importance': dict(zip(X.columns, rf.feature_importances_))
        }
        
        print(f"✅ Random Forest {task_name} - Accuracy: {accuracy:.3f}")
        return rf
    
    def train_xgboost(self, X: pd.DataFrame, y: pd.Series, 
                     task_name: str) -> xgb.XGBClassifier:
        """Train XGBoost classifier."""
        print(f"🚀 Training XGBoost for {task_name}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Encode labels if necessary
        if y_train.dtype == 'object':
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)
        else:
            y_train_encoded = y_train
            y_test_encoded = y_test
            le = None
        
        # Train model
        xgb_model = xgb.XGBClassifier(**XGBOOST_CONFIG)
        xgb_model.fit(X_train, y_train_encoded)
        
        # Predictions
        y_pred = xgb_model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test_encoded, y_pred)
        
        # Store results
        self.models[f'xgb_{task_name}'] = {'model': xgb_model, 'label_encoder': le}
        self.results[f'xgb_{task_name}'] = {
            'accuracy': accuracy,
            'feature_importance': dict(zip(X.columns, xgb_model.feature_importances_))
        }
        
        print(f"✅ XGBoost {task_name} - Accuracy: {accuracy:.3f}")
        return xgb_model
    
    def train_neural_network(self, X: pd.DataFrame, y: pd.Series, 
                           task_name: str) -> tf.keras.Model:
        """Train neural network classifier."""
        print(f"🧠 Training Neural Network for {task_name}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Encode labels
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        
        # Convert to categorical if multiclass
        n_classes = len(np.unique(y_train_encoded))
        if n_classes > 2:
            y_train_categorical = tf.keras.utils.to_categorical(y_train_encoded)
            y_test_categorical = tf.keras.utils.to_categorical(y_test_encoded)
        else:
            y_train_categorical = y_train_encoded
            y_test_categorical = y_test_encoded
        
        # Build model
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(n_classes if n_classes > 2 else 1, 
                  activation='softmax' if n_classes > 2 else 'sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy' if n_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        history = model.fit(
            X_train_scaled, y_train_categorical,
            batch_size=32,
            epochs=100,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        if n_classes > 2:
            y_pred = np.argmax(model.predict(X_test_scaled), axis=1)
        else:
            y_pred = (model.predict(X_test_scaled) > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_test_encoded, y_pred)
        
        # Store results
        self.models[f'nn_{task_name}'] = {
            'model': model, 
            'scaler': scaler, 
            'label_encoder': le,
            'history': history
        }
        self.results[f'nn_{task_name}'] = {'accuracy': accuracy}
        
        print(f"✅ Neural Network {task_name} - Accuracy: {accuracy:.3f}")
        return model
    
    def train_all_models(self):
        """Train all models for all tasks."""
        print("🤖 Starting model training...")
        
        X, targets, feature_cols = self.prepare_features_and_targets()
        
        for task_name, y in targets.items():
            if y.notna().sum() > 0:  # Only train if we have valid targets
                print(f"\n📊 Training models for {task_name}")
                
                # Remove rows with missing targets
                mask = y.notna()
                X_clean = X[mask]
                y_clean = y[mask]
                
                # Train models
                self.train_random_forest(X_clean, y_clean, task_name)
                self.train_xgboost(X_clean, y_clean, task_name)
                self.train_neural_network(X_clean, y_clean, task_name)
        
        print("✅ All models trained successfully!")
    
    def save_models(self, models_dir: Path):
        """Save all trained models."""
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model_data in self.models.items():
            model_path = models_dir / f"{model_name}.joblib"
            joblib.dump(model_data, model_path)
            print(f"💾 Saved {model_name} to {model_path}")
        
        # Save results
        results_path = models_dir / "model_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = {}
            for key, value in self.results.items():
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.floating):
                        json_results[key][k] = float(v)
                    elif isinstance(v, np.integer):
                        json_results[key][k] = int(v)
                    elif isinstance(v, dict):
                        json_results[key][k] = {str(kk): float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                                               for kk, vv in v.items()}
                    else:
                        json_results[key][k] = v
            
            json.dump(json_results, f, indent=2)
        print(f"💾 Saved results to {results_path}")

# Import necessary libraries that might be missing
try:
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    print("Note: TensorFlow features may not be available.") 