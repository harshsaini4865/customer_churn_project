#!/usr/bin/env python3
"""
Comprehensive startup script for Customer Churn Prediction System
Handles initialization, error checking, and provides detailed feedback
"""

import sys
import os
import subprocess
import traceback
import time
from pathlib import Path

def check_environment():
    """Check Python environment and dependencies"""
    print("🔍 Checking environment...")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check if we're in the right directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Check for required files
    required_files = ['main.py', 'templates/index.html']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ Required files found")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    required_packages = [
        'flask',
        'pandas', 
        'numpy',
        'scikit-learn'
    ]
    
    failed_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                failed_packages.append(package)
    
    if failed_packages:
        print(f"❌ Failed to install: {failed_packages}")
        return False
    
    return True

def initialize_model():
    """Initialize the model and dataset"""
    print("🧠 Initializing model...")
    
    try:
        # Import the main module
        import main
        
        # Check for dataset files
        dataset_files = ['churn_cleaned.csv', 'WA_Fn-UseC_-Telco-Customer-Churn.csv']
        dataset_found = None
        
        for file in dataset_files:
            if os.path.exists(file):
                dataset_found = file
                break
        
        if not dataset_found:
            print("⚠️  No dataset files found. Running in demo mode.")
            return False
        
        print(f"📊 Loading dataset: {dataset_found}")
        
        # Create predictor instance
        predictor = main.ChurnPredictor()
        
        # Load and preprocess data
        df = predictor.load_and_preprocess_data(dataset_found)
        if df is None:
            print("❌ Failed to load dataset")
            return False
        
        print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Train model
        accuracy = predictor.train_model(df)
        print(f"✅ Model trained with accuracy: {accuracy:.2%}")
        
        # Store the predictor globally
        main.predictor = predictor
        main.model = predictor.model
        main.scaler = predictor.scaler
        main.label_encoders = predictor.label_encoders
        main.feature_columns = predictor.feature_columns
        main.model_accuracy = accuracy
        
        return True
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        traceback.print_exc()
        return False

def test_endpoints():
    """Test Flask endpoints"""
    print("🧪 Testing endpoints...")
    
    try:
        import requests
        import time
        
        # Wait a moment for the server to start
        time.sleep(2)
        
        # Test the predict endpoint
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
        
        response = requests.post('http://localhost:5000/predict', 
                               json=test_data, 
                               timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ /predict endpoint working: {result}")
        else:
            print(f"❌ /predict endpoint failed: {response.status_code}")
            
    except ImportError:
        print("⚠️  requests module not available, skipping endpoint test")
    except Exception as e:
        print(f"❌ Endpoint test failed: {e}")

def main():
    """Main startup sequence"""
    print("🚀 Customer Churn Prediction System")
    print("=" * 60)
    
    # Step 1: Check environment
    if not check_environment():
        print("❌ Environment check failed")
        return
    
    # Step 2: Install dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed")
        return
    
    # Step 3: Initialize model
    model_initialized = initialize_model()
    
    if model_initialized:
        print("✅ Model initialized successfully!")
    else:
        print("⚠️  Running in demo mode (no real predictions)")
    
    # Step 4: Start Flask app
    print("🌐 Starting Flask development server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("📝 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        import main
        main.app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()