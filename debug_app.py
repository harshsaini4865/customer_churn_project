#!/usr/bin/env python3
"""
Debug script for Customer Churn Prediction System
Provides detailed error analysis and troubleshooting
"""

import sys
import os
import traceback
import json
from pathlib import Path

def test_basic_imports():
    """Test basic package imports"""
    print("=== Testing Basic Imports ===")
    
    required_packages = [
        ('flask', 'Flask web framework'),
        ('pandas', 'Data manipulation'),
        ('numpy', 'Numerical computing'),
        ('sklearn', 'Machine learning')
    ]
    
    results = {}
    
    for package, description in required_packages:
        try:
            if package == 'flask':
                import flask
                results['flask'] = {'success': True, 'version': flask.__version__}
                print(f"✓ {package} ({description}): {flask.__version__}")
            elif package == 'pandas':
                import pandas as pd
                results['pandas'] = {'success': True, 'version': pd.__version__}
                print(f"✓ {package} ({description}): {pd.__version__}")
            elif package == 'numpy':
                import numpy as np
                results['numpy'] = {'success': True, 'version': np.__version__}
                print(f"✓ {package} ({description}): {np.__version__}")
            elif package == 'sklearn':
                import sklearn
                results['sklearn'] = {'success': True, 'version': sklearn.__version__}
                print(f"✓ {package} ({description}): {sklearn.__version__}")
        except ImportError as e:
            results[package] = {'success': False, 'error': str(e)}
            print(f"✗ {package} ({description}): {e}")
    
    print()
    return results

def check_file_structure():
    """Check the project file structure"""
    print("=== Checking File Structure ===")
    
    required_files = {
        'main.py': 'Flask application',
        'templates/index.html': 'HTML template',
        'requirements.txt': 'Dependencies'
    }
    
    results = {}
    
    for file_path, description in required_files.items():
        full_path = Path(file_path)
        if full_path.exists():
            size = full_path.stat().st_size
            results[file_path] = {'success': True, 'size': size}
            print(f"✓ {file_path} ({description}): {size} bytes")
        else:
            results[file_path] = {'success': False, 'error': 'File not found'}
            print(f"✗ {file_path} ({description}): Not found")
    
    # Check for dataset files
    dataset_files = ['churn_cleaned.csv', 'WA_Fn-UseC_-Telco-Customer-Churn.csv']
    for file in dataset_files:
        if Path(file).exists():
            size = Path(file).stat().st_size
            results[file] = {'success': True, 'size': size}
            print(f"✓ {file}: {size} bytes")
        else:
            results[file] = {'success': False, 'error': 'File not found'}
            print(f"✗ {file}: Not found")
    
    print()
    return results

def test_dataset_loading():
    """Test dataset loading and basic analysis"""
    print("=== Testing Dataset Loading ===")
    
    try:
        import pandas as pd
        
        # Check for dataset files
        dataset_files = ['churn_cleaned.csv', 'WA_Fn-UseC_-Telco-Customer-Churn.csv']
        loaded_df = None
        
        for file in dataset_files:
            if Path(file).exists():
                try:
                    df = pd.read_csv(file)
                    print(f"✓ Loaded {file}: {df.shape[0]} rows, {df.shape[1]} columns")
                    
                    # Basic data info
                    print(f"  Columns: {list(df.columns)}")
                    print(f"  Data types: {df.dtypes.value_counts().to_dict()}")
                    
                    # Check for missing values
                    missing = df.isnull().sum().sum()
                    print(f"  Missing values: {missing}")
                    
                    # Check for 'Churn' column
                    if 'Churn' in df.columns:
                        churn_counts = df['Churn'].value_counts()
                        print(f"  Churn distribution: {churn_counts.to_dict()}")
                    
                    loaded_df = df
                    break
                    
                except Exception as e:
                    print(f"✗ Error loading {file}: {e}")
        
        if loaded_df is None:
            print("✗ No dataset files could be loaded")
            return False
        
        return True
        
    except ImportError:
        print("✗ Pandas not available")
        return False
    except Exception as e:
        print(f"✗ Error in dataset loading: {e}")
        return False

def test_model_initialization():
    """Test model initialization without Flask"""
    print("=== Testing Model Initialization ===")
    
    try:
        # Import required modules
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.model_selection import train_test_split
        import pandas as pd
        import numpy as np
        
        # Check if main.py exists and can be imported
        if not Path('main.py').exists():
            print("✗ main.py not found")
            return False
        
        # Read main.py to extract the ChurnPredictor class
        with open('main.py', 'r') as f:
            main_content = f.read()
        
        # Execute the main.py content in a controlled environment
        exec_globals = {}
        exec(main_content, exec_globals)
        
        # Get the ChurnPredictor class
        ChurnPredictor = exec_globals.get('ChurnPredictor')
        
        if ChurnPredictor is None:
            print("✗ ChurnPredictor class not found in main.py")
            return False
        
        print("✓ ChurnPredictor class found")
        
        # Test dataset loading
        dataset_files = ['churn_cleaned.csv', 'WA_Fn-UseC_-Telco-Customer-Churn.csv']
        dataset_file = None
        
        for file in dataset_files:
            if Path(file).exists():
                dataset_file = file
                break
        
        if dataset_file is None:
            print("✗ No dataset file found")
            return False
        
        # Create predictor instance
        predictor = ChurnPredictor()
        print("✓ ChurnPredictor instance created")
        
        # Load data
        df = predictor.load_and_preprocess_data(dataset_file)
        if df is None:
            print("✗ Failed to load and preprocess data")
            return False
        
        print(f"✓ Data loaded and preprocessed: {df.shape}")
        
        # Train model
        accuracy = predictor.train_model(df)
        print(f"✓ Model trained with accuracy: {accuracy:.2%}")
        
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
        
        return True
        
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        traceback.print_exc()
        return False

def test_flask_app():
    """Test Flask application startup"""
    print("=== Testing Flask Application ===")
    
    try:
        # Import Flask
        from flask import Flask
        
        # Create a simple test app
        app = Flask(__name__)
        
        @app.route('/test')
        def test():
            return jsonify({'status': 'ok', 'message': 'Flask is working'})
        
        # Test if we can create routes
        print("✓ Flask app created successfully")
        print("✓ Test route created")
        
        # Check if main.py Flask app can be imported
        if Path('main.py').exists():
            # Read and check Flask app structure
            with open('main.py', 'r') as f:
                content = f.read()
            
            if 'app = Flask(__name__)' in content:
                print("✓ Flask app found in main.py")
            else:
                print("✗ Flask app not found in main.py")
            
            if '@app.route' in content:
                print("✓ Flask routes found in main.py")
            else:
                print("✗ Flask routes not found in main.py")
        
        return True
        
    except ImportError:
        print("✗ Flask not available")
        return False
    except Exception as e:
        print(f"✗ Flask test failed: {e}")
        return False

def generate_report():
    """Generate a comprehensive report"""
    print("\n" + "="*60)
    print("📊 CUSTOMER CHURN PREDICTION SYSTEM - DEBUG REPORT")
    print("="*60)
    
    # Run all tests
    import_results = test_basic_imports()
    file_results = check_file_structure()
    dataset_results = test_dataset_loading()
    model_results = test_model_initialization()
    flask_results = test_flask_app()
    
    # Summary
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    total_tests = 5
    passed_tests = sum([
        all(result.get('success', False) for result in import_results.values()),
        all(result.get('success', False) for result in file_results.values()),
        dataset_results,
        model_results,
        flask_results
    ])
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("✅ All systems ready! You can start the application with:")
        print("   python main.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("   Common fixes:")
        print("   1. Install missing packages: pip install flask pandas numpy scikit-learn")
        print("   2. Ensure dataset files are in the current directory")
        print("   3. Check main.py for syntax errors")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    generate_report()