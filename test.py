"""
Marine Debris Classification Testing Suite
==========================================

Test suite for evaluating trained marine debris classification models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

try:
    from config import *
except ImportError:
    # Default paths if config not available
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR / "models" / "trained"
    RESULTS_DIR = BASE_DIR / "results"
    DATA_DIR = BASE_DIR / "data"

class MarineDebrisModelTester:
    """Test trained marine debris classification models."""
    
    def __init__(self):
        self.models = {}
        self.test_data = None
        self.test_results = {}
    
    def load_models(self, models_dir: Path = MODELS_DIR):
        """Load all saved models."""
        print("📂 Loading trained models...")
        
        if not models_dir.exists():
            print(f"❌ Models directory not found: {models_dir}")
            return False
        
        model_files = list(models_dir.glob("*.joblib"))
        
        if not model_files:
            print("❌ No model files found!")
            return False
        
        for model_file in model_files:
            try:
                model_name = model_file.stem
                self.models[model_name] = joblib.load(model_file)
                print(f"✅ Loaded {model_name}")
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
        
        return len(self.models) > 0
    
    def load_test_data(self, data_path: Path = None):
        """Load test data from processed dataset."""
        if data_path is None:
            # Try to load processed data
            processed_file = DATA_DIR / "processed" / "marine_debris_processed.csv"
            if processed_file.exists():
                data_path = processed_file
            else:
                # Create sample test data
                return self._create_test_data()
        
        try:
            print(f"📊 Loading test data from: {data_path}")
            self.test_data = pd.read_csv(data_path)
            print(f"✅ Test data loaded: {self.test_data.shape}")
            return True
        except Exception as e:
            print(f"❌ Failed to load test data: {e}")
            return self._create_test_data()
    
    def _create_test_data(self):
        """Create sample test data for model testing."""
        print("🔧 Creating sample test data...")
        
        np.random.seed(42)
        n_samples = 1000
        
        # Create test data with same structure as training data
        test_data = {
            'Latitude': np.random.uniform(25, 48, n_samples),
            'Longitude': np.random.uniform(-125, -70, n_samples),
            'Survey_Year': np.random.choice(range(2015, 2024), n_samples),
            'Slope': np.random.uniform(0, 45, n_samples),
            'Width': np.random.uniform(10, 500, n_samples),
            'Length': np.random.uniform(100, 5000, n_samples),
            'Plastic_Bags': np.random.poisson(5, n_samples),
            'Plastic_Bottles': np.random.poisson(8, n_samples),
            'Metal_Cans': np.random.poisson(4, n_samples),
            'Glass_Bottles': np.random.poisson(2, n_samples),
            'Year': np.random.choice(range(2015, 2024), n_samples),
            'Month': np.random.choice(range(1, 13), n_samples),
            'Day_of_Year': np.random.choice(range(1, 366), n_samples),
            'Weather_Impact_Score': np.random.choice(range(1, 5), n_samples),
            'Storm_Impact_Score': np.random.choice(range(0, 4), n_samples),
            'Distance_from_Equator': np.random.uniform(25, 48, n_samples),
            'Season_Numeric': np.random.choice(range(1, 5), n_samples),
        }
        
        # Add derived features
        test_data['Total_Debris'] = (test_data['Plastic_Bags'] + 
                                   test_data['Plastic_Bottles'] + 
                                   test_data['Metal_Cans'] + 
                                   test_data['Glass_Bottles'])
        
        test_data['Area'] = test_data['Width'] * test_data['Length']
        test_data['Debris_Density'] = test_data['Total_Debris'] / test_data['Area']
        test_data['Environmental_Risk'] = (test_data['Weather_Impact_Score'] + 
                                         test_data['Storm_Impact_Score'])
        
        # Add target variables for testing
        test_data['Debris_Source'] = np.random.choice(['land_based', 'ocean_based'], n_samples)
        test_data['Primary_Material'] = np.random.choice(['Plastic', 'Metal', 'Glass', 'Other'], n_samples)
        test_data['Pollution_Level'] = np.random.choice(['Low', 'Medium', 'High', 'Critical'], n_samples)
        
        self.test_data = pd.DataFrame(test_data)
        print(f"✅ Sample test data created: {self.test_data.shape}")
        return True
    
    def prepare_test_features(self, target_task: str):
        """Prepare features for testing specific task."""
        df = self.test_data.copy()
        
        # Define feature columns (same as training)
        exclude_cols = ['Debris_Source', 'Primary_Material', 'Pollution_Level']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
        
        X = df[feature_cols].fillna(0)
        
        # Get target variable
        target_map = {
            'debris_source': 'Debris_Source',
            'material_type': 'Primary_Material', 
            'pollution_level': 'Pollution_Level'
        }
        
        y = None
        if target_task in target_map and target_map[target_task] in df.columns:
            y = df[target_map[target_task]]
        
        return X, y, feature_cols
    
    def test_random_forest_model(self, model_name: str):
        """Test Random Forest model."""
        if model_name not in self.models:
            print(f"❌ Model {model_name} not found!")
            return None
        
        print(f"🌲 Testing {model_name}...")
        
        # Extract task name
        task_name = model_name.replace('rf_', '')
        
        # Prepare features
        X_test, y_true, feature_cols = self.prepare_test_features(task_name)
        
        if y_true is None:
            print(f"❌ No target variable available for {task_name}")
            return None
        
        # Get model
        model = self.models[model_name]
        
        # Make predictions
        try:
            # Ensure feature columns match
            model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else feature_cols
            X_test_aligned = X_test[[col for col in model_features if col in X_test.columns]]
            
            # Add missing columns with zeros
            for col in model_features:
                if col not in X_test_aligned.columns:
                    X_test_aligned[col] = 0
            
            # Reorder columns to match model training
            X_test_aligned = X_test_aligned[model_features]
            
            y_pred = model.predict(X_test_aligned)
            y_proba = model.predict_proba(X_test_aligned) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            results = {
                'model_type': 'Random Forest',
                'task': task_name,
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1_score': f1_score(y_true, y_pred, average='weighted'),
                'predictions': y_pred,
                'probabilities': y_proba,
                'true_labels': y_true
            }
            
            self.test_results[model_name] = results
            
            print(f"✅ {model_name} - Accuracy: {results['accuracy']:.3f}")
            return results
            
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            return None
    
    def test_xgboost_model(self, model_name: str):
        """Test XGBoost model."""
        if model_name not in self.models:
            print(f"❌ Model {model_name} not found!")
            return None
        
        print(f"🚀 Testing {model_name}...")
        
        # Extract task name
        task_name = model_name.replace('xgb_', '')
        
        # Prepare features
        X_test, y_true, feature_cols = self.prepare_test_features(task_name)
        
        if y_true is None:
            print(f"❌ No target variable available for {task_name}")
            return None
        
        # Get model and label encoder
        model_data = self.models[model_name]
        model = model_data['model'] if isinstance(model_data, dict) else model_data
        label_encoder = model_data.get('label_encoder') if isinstance(model_data, dict) else None
        
        try:
            # Encode labels if encoder available
            if label_encoder:
                y_true_encoded = label_encoder.transform(y_true)
            else:
                y_true_encoded = y_true
            
            # Make predictions
            y_pred_encoded = model.predict(X_test)
            
            # Decode predictions if encoder available
            if label_encoder:
                y_pred = label_encoder.inverse_transform(y_pred_encoded)
            else:
                y_pred = y_pred_encoded
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            results = {
                'model_type': 'XGBoost',
                'task': task_name,
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1_score': f1_score(y_true, y_pred, average='weighted'),
                'predictions': y_pred,
                'true_labels': y_true
            }
            
            self.test_results[model_name] = results
            
            print(f"✅ {model_name} - Accuracy: {results['accuracy']:.3f}")
            return results
            
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            return None
    
    def test_all_models(self):
        """Test all loaded models."""
        print("🧪 Testing all models...")
        
        if not self.models:
            print("❌ No models loaded!")
            return False
        
        if self.test_data is None:
            print("❌ No test data available!")
            return False
        
        for model_name in self.models.keys():
            if model_name.startswith('rf_'):
                self.test_random_forest_model(model_name)
            elif model_name.startswith('xgb_'):
                self.test_xgboost_model(model_name)
            elif model_name.startswith('nn_'):
                print(f"⚠️ Neural network testing not implemented for {model_name}")
        
        return len(self.test_results) > 0
    
    def plot_results(self, save_plots: bool = True):
        """Plot testing results."""
        if not self.test_results:
            print("❌ No test results to plot!")
            return
        
        print("📈 Generating result plots...")
        
        # Set style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # 1. Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(self.test_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            values = [self.test_results[model][metric] for model in models]
            
            bars = ax.bar(models, values, color=plt.cm.Set3(np.linspace(0, 1, len(models))))
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            # Rotate x-axis labels
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(RESULTS_DIR / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Confusion Matrices
        n_models = len([m for m in self.test_results.keys() if 'predictions' in self.test_results[m]])
        if n_models > 0:
            fig, axes = plt.subplots(1, min(n_models, 3), figsize=(5*min(n_models, 3), 4))
            if n_models == 1:
                axes = [axes]
            
            plot_idx = 0
            for model_name, results in self.test_results.items():
                if 'predictions' in results and plot_idx < 3:
                    ax = axes[plot_idx] if n_models > 1 else axes[0]
                    
                    cm = confusion_matrix(results['true_labels'], results['predictions'])
                    labels = sorted(set(results['true_labels']))
                    
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=labels, yticklabels=labels, ax=ax)
                    ax.set_title(f'{model_name} Confusion Matrix')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    
                    plot_idx += 1
            
            plt.tight_layout()
            if save_plots:
                plt.savefig(RESULTS_DIR / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print("✅ Result plots generated successfully!")
    
    def print_testing_summary(self):
        """Print comprehensive testing summary."""
        print("\n" + "="*60)
        print("🧪 MARINE DEBRIS MODEL TESTING SUMMARY")
        print("="*60)
        
        if not self.test_results:
            print("❌ No test results available!")
            return
        
        # Overall statistics
        print(f"📊 Total Models Tested: {len(self.test_results)}")
        print(f"📈 Test Data Size: {len(self.test_data) if self.test_data is not None else 'N/A'}")
        
        # Performance summary
        print("\n📋 MODEL PERFORMANCE SUMMARY:")
        print("-" * 60)
        
        for model_name, results in self.test_results.items():
            print(f"\n🤖 {model_name.upper()}")
            print(f"   Type: {results['model_type']}")
            print(f"   Task: {results['task']}")
            print(f"   Accuracy:  {results['accuracy']:.3f}")
            print(f"   Precision: {results['precision']:.3f}")
            print(f"   Recall:    {results['recall']:.3f}")
            print(f"   F1-Score:  {results['f1_score']:.3f}")
        
        # Best performers
        print("\n🏆 BEST PERFORMERS:")
        print("-" * 30)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in metrics:
            best_model = max(self.test_results.keys(), 
                           key=lambda x: self.test_results[x][metric])
            best_score = self.test_results[best_model][metric]
            print(f"   {metric.replace('_', ' ').title()}: {best_model} ({best_score:.3f})")
        
        print("\n✅ Testing completed successfully!")

def main():
    """Main testing function."""
    print("🧪 Marine Debris Model Testing Suite")
    print("=" * 50)
    
    # Initialize tester
    tester = MarineDebrisModelTester()
    
    # Load models
    if not tester.load_models():
        print("❌ Failed to load models. Exiting...")
        return
    
    # Load test data
    if not tester.load_test_data():
        print("❌ Failed to load test data. Exiting...")
        return
    
    # Test all models
    if not tester.test_all_models():
        print("❌ Failed to test models. Exiting...")
        return
    
    # Generate plots
    tester.plot_results()
    
    # Print summary
    tester.print_testing_summary()

if __name__ == "__main__":
    main() 