#!/usr/bin/env python3
"""
Startup script for Customer Churn Prediction System with detailed error reporting
"""

import sys
import os
import subprocess
import traceback
from pathlib import Path

def check_python_environment():
    """Check Python version and basic environment"""
    print("=== Python Environment Check ===")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")  # First 3 entries
    print()

def install_requirements():
    """Install required packages"""
    print("=== Installing Requirements ===")
    requirements_file = "requirements.txt"
    
    if os.path.exists(requirements_file):
        try:
            print("Installing packages from requirements.txt...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", requirements_file
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Requirements installed successfully")
            else:
                print(f"✗ Error installing requirements: {result.stderr}")
                return False
        except Exception as e:
            print(f"✗ Failed to install requirements: {e}")
            return False
    else:
        print("requirements.txt not found, installing basic packages...")
        packages = ["flask", "pandas", "numpy", "scikit-learn"]
        for package in packages:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], check=True)
                print(f"✓ Installed {package}")
            except Exception as e:
                print(f"✗ Failed to install {package}: {e}")
                return False
    
    print()
    return True

def test_imports():
    """Test if all required packages can be imported"""
    print("=== Testing Package Imports ===")
    
    packages = {
        'flask': 'Flask web framework',
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'sklearn': 'Machine learning'
    }
    
    all_imported = True
    
    for package, description in packages.items():
        try:
            if package == 'flask':
                import flask
                print(f"✓ {package} ({description}): {flask.__version__}")
            elif package == 'pandas':
                import pandas as pd
                print(f"✓ {package} ({description}): {pd.__version__}")
            elif package == 'numpy':
                import numpy as np
                print(f"✓ {package} ({description}): {np.__version__}")
            elif package == 'sklearn':
                import sklearn
                print(f"✓ {package} ({description}): {sklearn.__version__}")
        except ImportError as e:
            print(f"✗ {package} ({description}): {e}")
            all_imported = False
    
    print()
    return all_imported

def check_dataset_files():
    """Check if dataset files exist"""
    print("=== Checking Dataset Files ===")
    
    files_to_check = [
        'churn_cleaned.csv',
        'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    ]
    
    found_files = []
    
    for file in files_to_check:
        if os.path.exists(file):
            try:
                import pandas as pd
                df = pd.read_csv(file, nrows=5)  # Read first 5 rows
                print(f"✓ {file}: Found ({df.shape[0]} rows, {df.shape[1]} columns)")
                print(f"  Sample columns: {list(df.columns[:5])}")
                found_files.append(file)
            except Exception as e:
                print(f"✗ {file}: Error reading - {e}")
        else:
            print(f"✗ {file}: Not found")
    
    print()
    return found_files

def test_model_initialization():
    """Test model initialization without starting Flask"""
    print("=== Testing Model Initialization ===")
    
    try:
        # Import the main module
        sys.path.insert(0, os.getcwd())
        import main
        
        print("✓ main.py imported successfully")
        
        # Test dataset loading
        dataset_files = check_dataset_files()
        if not dataset_files:
            print("✗ No dataset files available for testing")
            return False
        
        # Use the first available dataset
        dataset_file = dataset_files[0]
        print(f"Testing with {dataset_file}...")
        
        # Test the predictor
        predictor = main.ChurnPredictor()
        
        # Load and preprocess data
        df = predictor.load_and_preprocess_data(dataset_file)
        if df is None:
            print("✗ Failed to load and preprocess data")
            return False
        
        print(f"✓ Data loaded and preprocessed. Shape: {df.shape}")
        
        # Train the model
        accuracy = predictor.train_model(df)
        print(f"✓ Model trained successfully. Accuracy: {accuracy:.2%}")
        
        # Test prediction
        test_data = {
            'gender': 'Female',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'No',
            'tenure': 12,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'DSL',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 70.0,
            'TotalCharges': 840.0
        }
        
        result = predictor.predict_churn(test_data)
        if result:
            print(f"✓ Test prediction successful:")
            print(f"  Churn prediction: {result['churn_prediction']}")
            print(f"  Churn probability: {result['churn_probability']:.2%}")
            print(f"  Confidence: {result['confidence']:.2%}")
        else:
            print("✗ Test prediction failed")
            return False
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        traceback.print_exc()
        print()
        return False

def start_flask_app():
    """Start the Flask application"""
    print("=== Starting Flask Application ===")
    
    try:
        print("Starting Flask development server...")
        print("The application will be available at: http://localhost:5000")
        print("Press Ctrl+C to stop the server")
        print()
        
        # Import and run the main application
        import main
        
        # Set environment variables for development
        os.environ['FLASK_ENV'] = 'development'
        os.environ['FLASK_DEBUG'] = '1'
        
        # Run the Flask app
        main.app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
        
    except KeyboardInterrupt:
        print("\nFlask server stopped by user")
    except Exception as e:
        print(f"✗ Error starting Flask app: {e}")
        traceback.print_exc()

def main():
    """Main startup sequence"""
    print("🚀 Customer Churn Prediction System Startup")
    print("=" * 50)
    print()
    
    # Step 1: Check Python environment
    check_python_environment()
    
    # Step 2: Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements. Please install manually:")
        print("pip install flask pandas numpy scikit-learn")
        return
    
    # Step 3: Test imports
    if not test_imports():
        print("❌ Import tests failed. Please check your Python environment.")
        return
    
    # Step 4: Check dataset files
    dataset_files = check_dataset_files()
    if not dataset_files:
        print("⚠️  No dataset files found. The app will run in demo mode.")
        print("Please ensure your dataset files are in the current directory.")
    
    # Step 5: Test model initialization
    if not test_model_initialization():
        print("❌ Model initialization failed. Please check the errors above.")
        return
    
    print("✅ All tests passed! Starting Flask application...")
    print()
    
    # Step 6: Start Flask app
    start_flask_app()

if __name__ == "__main__":
    main()