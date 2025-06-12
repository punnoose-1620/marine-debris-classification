"""
Marine Debris Classification Project Configuration
==================================================

This file contains all configuration settings for the marine debris classification project.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Base project directory
BASE_DIR = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Model directories
MODELS_DIR = BASE_DIR / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
MODEL_CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# Results directories
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# Visualization directories
VISUALIZATIONS_DIR = BASE_DIR / "visualizations"
REPORTS_DIR = BASE_DIR / "reports"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
                  MODELS_DIR, TRAINED_MODELS_DIR, MODEL_CHECKPOINTS_DIR,
                  RESULTS_DIR, FIGURES_DIR, METRICS_DIR, VISUALIZATIONS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Dataset information
DATASET_CONFIG = {
    'kaggle_dataset': 'your-username/marine-debris-dataset',  # Update with actual dataset
    'raw_data_file': 'nasaa.csv',
    'processed_data_file': 'marine_debris_processed.csv',
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42
}

# Column mappings and feature groups
COLUMN_MAPPINGS = {
    'geographic_features': [
        'Country', 'State', 'Latitude', 'Longitude', 'Shoreline_Name'
    ],
    'temporal_features': [
        'Date', 'Season', 'Survey_Year'
    ],
    'environmental_features': [
        'Weather', 'Storm_Activity', 'Slope', 'Width', 'Length'
    ],
    'debris_categories': [
        'Plastic_Bags', 'Plastic_Bottles', 'Plastic_Food_Containers',
        'Plastic_Utensils', 'Plastic_Straws', 'Plastic_Lids',
        'Metal_Cans', 'Metal_Bottle_Caps', 'Glass_Bottles',
        'Rubber_Gloves', 'Cloth_Items', 'Paper_Items'
    ],
    'organizational_features': [
        'Organization', 'Survey_Type'
    ]
}

# Target variable configurations
TARGET_VARIABLES = {
    'debris_source': {
        'name': 'debris_source',
        'type': 'binary',
        'classes': ['land_based', 'ocean_based']
    },
    'material_type': {
        'name': 'primary_material',
        'type': 'multiclass',
        'classes': ['Plastic', 'Metal', 'Glass', 'Rubber', 'Cloth', 'Paper', 'Other']
    },
    'pollution_severity': {
        'name': 'pollution_level',
        'type': 'ordinal',
        'classes': ['Low', 'Medium', 'High', 'Critical']
    }
}

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# Random Forest Configuration
RANDOM_FOREST_CONFIG = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

# XGBoost Configuration
XGBOOST_CONFIG = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

# Neural Network Configuration
NEURAL_NETWORK_CONFIG = {
    'hidden_layers': [128, 64, 32],
    'activation': 'relu',
    'dropout_rate': 0.3,
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5
}

# Clustering Configuration
CLUSTERING_CONFIG = {
    'kmeans': {
        'n_clusters': 5,
        'random_state': 42,
        'n_init': 10
    },
    'dbscan': {
        'eps': 0.5,
        'min_samples': 5
    },
    'umap': {
        'n_neighbors': 15,
        'min_dist': 0.1,
        'n_components': 2,
        'random_state': 42
    }
}

# Cross-validation configuration
CV_CONFIG = {
    'cv_folds': 5,
    'scoring_metrics': ['accuracy', 'precision', 'recall', 'f1'],
    'random_state': 42
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Color palettes
COLOR_PALETTES = {
    'debris_types': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'],
    'severity_levels': ['#2ECC71', '#F39C12', '#E74C3C', '#9B59B6'],
    'geographic': ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
}

# Plot settings
PLOT_SETTINGS = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'font_size': 12,
    'title_size': 16,
    'save_format': 'png'
}

# Map settings
MAP_SETTINGS = {
    'default_zoom': 6,
    'tile_layer': 'OpenStreetMap',
    'marker_size': 5,
    'heatmap_radius': 15,
    'center_coordinates': [39.8283, -98.5795]  # Center of US
}

# ============================================================================
# API KEYS AND EXTERNAL SERVICES
# ============================================================================

# API Keys (load from environment variables for security)
API_KEYS = {
    'kaggle_username': os.getenv('KAGGLE_USERNAME'),
    'kaggle_key': os.getenv('KAGGLE_KEY'),
    'mapbox_token': os.getenv('MAPBOX_ACCESS_TOKEN'),
    'google_maps_key': os.getenv('GOOGLE_MAPS_API_KEY')
}

# ============================================================================
# MODEL PERFORMANCE THRESHOLDS
# ============================================================================

PERFORMANCE_THRESHOLDS = {
    'accuracy': {
        'minimum': 0.75,
        'good': 0.85,
        'excellent': 0.90
    },
    'precision': {
        'minimum': 0.70,
        'good': 0.80,
        'excellent': 0.90
    },
    'recall': {
        'minimum': 0.70,
        'good': 0.80,
        'excellent': 0.90
    },
    'f1_score': {
        'minimum': 0.70,
        'good': 0.80,
        'excellent': 0.90
    }
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': RESULTS_DIR / 'marine_debris_analysis.log'
}

# ============================================================================
# PROCESSING CONFIGURATION
# ============================================================================

PROCESSING_CONFIG = {
    'chunk_size': 10000,  # For processing large datasets
    'n_jobs': -1,  # Number of parallel jobs
    'memory_limit': '8GB',  # Memory limit for processing
    'cache_results': True
}

# ============================================================================
# DEPLOYMENT CONFIGURATION
# ============================================================================

DEPLOYMENT_CONFIG = {
    'port': 5000,
    'host': '0.0.0.0',
    'debug': False,
    'model_version': '1.0.0',
    'api_version': 'v1'
}

# Export key variables for easy import
__all__ = [
    'BASE_DIR', 'DATA_DIR', 'MODELS_DIR', 'RESULTS_DIR',
    'DATASET_CONFIG', 'COLUMN_MAPPINGS', 'TARGET_VARIABLES',
    'RANDOM_FOREST_CONFIG', 'XGBOOST_CONFIG', 'NEURAL_NETWORK_CONFIG',
    'CLUSTERING_CONFIG', 'CV_CONFIG', 'COLOR_PALETTES', 'PLOT_SETTINGS',
    'MAP_SETTINGS', 'API_KEYS', 'PERFORMANCE_THRESHOLDS'
] 