#!/usr/bin/env python3
"""
Test script to identify errors in the Customer Churn Prediction System
"""

import sys
import os
import traceback

def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    try:
        import flask
        print(f"✓ Flask {flask.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Flask import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        print("✓ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ Scikit-learn import failed: {e}")
        return False
    
    return True

def test_dataset_files():
    """Test if dataset files exist and are readable"""
    print("\nTesting dataset files...")
    
    files_to_check = [
        'churn_cleaned.csv',
        'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            try:
                import pandas as pd
                df = pd.read_csv(file)
                print(f"✓ {file} found and readable. Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)}")
                if 'Churn' in df.columns:
                    churn_counts = df['Churn'].value_counts()
                    print(f"  Churn distribution: {dict(churn_counts)}")
                return file
            except Exception as e:
                print(f"✗ Error reading {file}: {e}")
        else:
            print(f"✗ {file} not found")
    
    return None

def test_model_initialization():
    """Test if the model can be initialized"""
    print("\nTesting model initialization...")
    
    try:
        # Import the ChurnPredictor class
        from main import ChurnPredictor
        
        predictor = ChurnPredictor()
        print("✓ ChurnPredictor class instantiated successfully")
        
        # Test dataset loading
        dataset_file = test_dataset_files()
        if dataset_file:
            df = predictor.load_and_preprocess_data(dataset_file)
            if df is not None:
                print(f"✓ Dataset loaded and preprocessed. Shape: {df.shape}")
                
                # Test model training
                accuracy = predictor.train_model(df)
                print(f"✓ Model trained successfully. Accuracy: {accuracy:.2f}")
                
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
                    print(f"✓ Prediction successful: {result}")
                else:
                    print("✗ Prediction failed")
                    
                return True
            else:
                print("✗ Dataset preprocessing failed")
        else:
            print("✗ No dataset file available")
            
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        traceback.print_exc()
    
    return False

def test_flask_app():
    """Test if Flask app can start"""
    print("\nTesting Flask application...")
    
    try:
        # Test if main.py can be imported
        import main
        print("✓ main.py imported successfully")
        
        # Check if Flask app exists
        if hasattr(main, 'app'):
            print("✓ Flask app object found")
            
            # Check if routes are defined
            routes = [str(rule) for rule in main.app.url_map.iter_rules()]
            print(f"✓ Flask routes defined: {routes}")
            
            return True
        else:
            print("✗ Flask app object not found")
            
    except Exception as e:
        print(f"✗ Flask app test failed: {e}")
        traceback.print_exc()
    
    return False

def main():
    """Run all tests"""
    print("=== Customer Churn Prediction System Test ===")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    # Run tests
    tests_passed = 0
    total_tests = 4
    
    if test_imports():
        tests_passed += 1
    
    if test_dataset_files():
        tests_passed += 1
    
    if test_model_initialization():
        tests_passed += 1
    
    if test_flask_app():
        tests_passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The system should work correctly.")
        print("\nTo run the Flask application:")
        print("python main.py")
        print("Then open: http://localhost:5000")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Check if dataset files are in the correct location")
        print("3. Verify Python environment and dependencies")

if __name__ == "__main__":
    main()