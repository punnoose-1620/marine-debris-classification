#!/bin/bash

# Marine Debris Classification Project Setup Script
echo "🌊 Setting up Marine Debris Source Classification Project..."

# Create virtual environment
echo "📦 Creating virtual environment 'marine-debris-env'..."
python3 -m venv marine-debris-env

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source marine-debris-env/bin/activate

# Upgrade pip
echo "🔄 Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "📚 Installing required packages..."
pip install -r requirements.txt

# Create project directory structure
echo "📁 Creating project directory structure..."
mkdir -p data
mkdir -p notebooks
mkdir -p models
mkdir -p results
mkdir -p visualizations
mkdir -p reports

# Create .gitignore file
echo "🚫 Creating .gitignore file..."
cat > .gitignore << EOF
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
marine-debris-env/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data files
*.csv
*.json
*.parquet
*.h5
*.pkl
*.pickle

# Model files
*.joblib
*.h5
*.pb
*.onnx

# Logs
*.log

# API keys
api_keys.txt
.env.local

# Large files
*.zip
*.tar.gz
*.rar
EOF

# Create data subdirectories
echo "📊 Creating data subdirectories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/external

# Create models subdirectories
echo "🤖 Creating model subdirectories..."
mkdir -p models/trained
mkdir -p models/checkpoints

# Create results subdirectories
echo "📈 Creating results subdirectories..."
mkdir -p results/figures
mkdir -p results/metrics

# Setup Kaggle API (if kaggle.json exists)
if [ -f ~/.kaggle/kaggle.json ]; then
    echo "🔑 Kaggle API credentials found!"
    chmod 600 ~/.kaggle/kaggle.json
else
    echo "⚠️  Kaggle API credentials not found. Please:"
    echo "   1. Go to https://www.kaggle.com/account"
    echo "   2. Download kaggle.json"
    echo "   3. Place it in ~/.kaggle/"
    echo "   4. Run: chmod 600 ~/.kaggle/kaggle.json"
fi

# Create a simple activation script
echo "🔧 Creating activation script..."
cat > activate_env.sh << EOF
#!/bin/bash
source marine-debris-env/bin/activate
echo "🌊 Marine Debris Classification environment activated!"
echo "📊 Project directory: $(pwd)"
echo "🚀 Run 'jupyter lab' to start Jupyter Lab"
EOF
chmod +x activate_env.sh

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate the environment: source marine-debris-env/bin/activate"
echo "   2. Or use the helper script: source activate_env.sh"
echo "   3. Set up Kaggle API credentials if not already done"
echo "   4. Run the main script: python marine_debris_classification.py"
echo "   5. Or start Jupyter Lab: jupyter lab"
echo ""
echo "🌊 Happy analyzing marine debris data!" 