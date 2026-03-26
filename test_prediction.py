#!/usr/bin/env python3
"""
Test script for Customer Churn Prediction System
Tests the prediction functionality and API endpoints
"""

try:
    import requests
except ImportError:
    print("Error: 'requests' module not found.")
    print("Install it using: pip install requests")
    sys.exit(1)

import json
import sys
import time

def test_prediction_api():
    """Test the prediction API endpoint"""
    print("🧪 Testing Prediction API")
    print("=" * 50)
    
    # Test data that matches the form fields
    test_data = {
        'gender': 'Female',
        'seniorCitizen': 0,
        'partner': 'Yes',
        'dependents': 'No',
        'tenure': 12,
        'phoneService': 'Yes',
        'multipleLines': 'No',
        'internetService': 'DSL',
        'onlineSecurity': 'No',
        'onlineBackup': 'No',
        'deviceProtection': 'No',
        'techSupport': 'No',
        'streamingTV': 'No',
        'streamingMovies': 'No',
        'contractType': 'Month-to-month',
        'paperlessBilling': 'Yes',
        'paymentMethod': 'Electronic check',
        'monthlyCharges': 70.0,
        'totalCharges': 840.0
    }
    
    try:
        print("Sending prediction request...")
        print(f"Data: {json.dumps(test_data, indent=2)}")
        
        response = requests.post('http://localhost:5000/predict', 
                               json=test_data, 
                               timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            
            if result.get('success'):
                print("✅ Prediction successful!")
                prediction = result.get('prediction', {})
                print(f"Churn Prediction: {prediction.get('churn_prediction')}")
                print(f"Churn Probability: {prediction.get('churn_probability', 0) * 100:.1f}%")
                print(f"Risk Level: {result.get('risk_level')}")
                print(f"Recommendation: {result.get('recommendation')}")
                return True
            else:
                print(f"❌ Prediction failed: {result.get('error')}")
                return False
        else:
            print(f"❌ API request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask server")
        print("Make sure the Flask app is running on http://localhost:5000")
        return False
    except Exception as e:
        print(f"❌ Error testing prediction: {e}")
        return False

def test_stats_api():
    """Test the stats API endpoint"""
    print("\n📊 Testing Stats API")
    print("=" * 50)
    
    try:
        response = requests.get('http://localhost:5000/api/stats', timeout=5)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            
            if 'error' not in result:
                print("✅ Stats API working!")
                print(f"Total Customers: {result.get('total_customers')}")
                print(f"Churned Customers: {result.get('churned_customers')}")
                print(f"Retention Rate: {result.get('retention_rate')}%")
                print(f"Model Accuracy: {result.get('model_accuracy')}%")
                return True
            else:
                print(f"❌ Stats error: {result.get('error')}")
                return False
        else:
            print(f"❌ Stats API failed with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask server")
        return False
    except Exception as e:
        print(f"❌ Error testing stats: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Customer Churn Prediction System - Test Suite")
    print("=" * 60)
    
    # Wait a moment for server to be ready
    print("Waiting for server to be ready...")
    time.sleep(2)
    
    # Test both APIs
    prediction_success = test_prediction_api()
    stats_success = test_stats_api()
    
    print("\n" + "=" * 60)
    print("📋 Test Results Summary")
    print("=" * 60)
    
    if prediction_success and stats_success:
        print("✅ All tests passed! The system is working correctly.")
        print("\n🌐 You can now use the web interface at:")
        print("   http://localhost:5000")
    else:
        print("❌ Some tests failed. Check the errors above.")
        if not prediction_success:
            print("   - Prediction API failed")
        if not stats_success:
            print("   - Stats API failed")
        
        print("\n💡 Troubleshooting tips:")
        print("   1. Make sure Flask is running: python main.py")
        print("   2. Check if dataset files are present")
        print("   3. Look for error messages in the Flask console")

if __name__ == "__main__":
    main()