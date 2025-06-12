# Marine Debris Source Classification Project - Complete Setup Prompt

Create a comprehensive Marine Debris Source Classification project using the NASA/NOAA marine debris dataset. This project should predict the source and type of marine debris based on survey data including location, environmental conditions, and debris characteristics.

## Project Requirements

### 1. Main Python file for Python Notebook (`marine_debris_classification.py`)
Create a single, comprehensive python code file (which will manually be converted to Jupyter notebook) with the following sections:

#### Data Analysis & Preprocessing
- Load and explore the nasaa.csv dataset from Kaggle using the kaggle directory for python
- Data cleaning and feature engineering from columns including:
  - Geographic features: Country, State, Latitude/Longitude, Shoreline_Name
  - Temporal features: Date, Season, Survey_Year
  - Environmental: Weather conditions, Storm_Activity, Slope, Width, Length
  - Debris categories: Plastic types, Metal, Glass, Rubber, Cloth, etc.
  - Organization conducting survey
- Handle missing values and outliers
- Create visualizations showing debris distribution by location, time, and type for prior analysis

#### Feature Engineering
- Extract meaningful features from location data (coastal regions, urban vs rural)
- Time-based features (seasonality, year trends)
- Environmental condition encoding
- Debris type aggregation and classification
- Create target variables for:
  - Primary debris source (land-based vs ocean-based)
  - Debris material classification (Plastic, Metal, Glass, etc.)
  - Pollution severity levels

#### Machine Learning Models
Implement and compare multiple classification approaches:
- **Random Forest Classifier** for debris source prediction
- **XGBoost** for material type classification
- **Neural Network** for multi-class debris categorization
- **Clustering analysis** for identifying debris hotspots
- Model evaluation with cross-validation, confusion matrices, feature importance

#### Advanced Analytics
- Geospatial analysis with folium maps showing debris distributions
- Time series analysis of debris trends
- Correlation analysis between environmental factors and debris types
- Predictive modeling for future debris accumulation

#### Results & Insights
- Model performance comparison
- Key factors influencing debris classification
- Geographic and temporal patterns
- Actionable insights for cleanup efforts

### 2. Project Structure & Files

#### README.md
Create a comprehensive README with:
- Project overview and objectives
- Dataset description and source attribution
- Installation and setup instructions
- Usage guide and examples
- Model performance metrics
- Key findings and insights
- Future improvements
- References (to atleast 6 similar research papers) and acknowledgments

#### setup.sh (Bash Script)
Create a complete setup script that:
- Creates a virtual environment named 'marine-debris-env'
- Installs all required packages:
  ```
  pandas numpy matplotlib seaborn plotly folium
  scikit-learn xgboost tensorflow keras
  jupyter jupyterlab
  geopandas contextily
  shap lime
  kaggle
  ```
- Creates necessary directory structure:
  ```
  marine-debris-classification/
  ├── data/
  ├── notebooks/
  ├── models/
  ├── results/
  ├── visualizations/
  └── reports/
  ```
- Sets up .gitignore for git repository setup
- Activates environment and launches Jupyter Lab

#### requirements.txt
Include all necessary dependencies with versions (update versions as required to latest compatible versions):
- Data manipulation: pandas>=1.5.0, numpy>=1.21.0
- Visualization: matplotlib>=3.5.0, seaborn>=0.11.0, plotly>=5.0.0
- ML frameworks: scikit-learn>=1.1.0, xgboost>=1.6.0, tensorflow>=2.8.0
- Geospatial: folium>=0.12.0, geopandas>=0.11.0
- Other: jupyter, shap, lime, kaggle

#### config.py
Configuration file with:
- Data paths and column mappings
- Model hyperparameters
- Visualization settings (including folder paths)
- API keys (if needed for mapping services)

#### test.py
Testing file that:
- Loads saved models
- Loads dataset
- Tests saved models with random data from dataset
- Plots results of testing for each model
- Prints testing summary

### 3. Advanced Features to Include

#### Model Interpretability
- SHAP values for feature importance
- LIME for local explanations
- Feature importance visualizations

#### Interactive Dashboards
- Plotly Dash components for data exploration
- Interactive maps showing debris hotspots
- Time series plots with filtering capabilities

#### Deployment Ready
- Model serialization and loading functions
- API endpoint structure for predictions
- Docker configuration for containerization

#### Geospatial Analysis
- Coastal proximity analysis
- Ocean current correlation
- Population density impact assessment

### 4. Data Science Best Practices
- Comprehensive data validation
- Robust error handling
- Modular, reusable code structure
- Extensive documentation and comments
- Unit tests for critical functions
- Version control integration

### 5. Specific Analysis Goals
The project should answer:
- What are the primary sources of marine debris in different regions?
- How do seasonal and weather patterns affect debris accumulation?
- Which environmental factors best predict debris type and quantity?
- Can we identify pollution hotspots and their characteristics?
- What recommendations can be made for targeted cleanup efforts?

### 6. Output Deliverables
- Trained models saved in joblib/pickle format
- Comprehensive analysis report
- Interactive visualizations and maps
- Model performance benchmarks
- Feature importance analysis
- Deployment-ready prediction functions

Ensure the entire project is production-ready, well-documented, and provides actionable insights for marine conservation efforts. The code should be clean, efficient, and suitable for both educational and professional use.