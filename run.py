#!/usr/bin/env python3
"""
Simple run script for Customer Churn Prediction System
"""

import sys
import os

def run_app():
    """Run the Flask application"""
    try:
        # Add current directory to Python path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Import and run the main application
        import main
        
        print("🚀 Starting Customer Churn Prediction System...")
        print("📊 Loading dataset and training model...")
        
        # Initialize the model
        if hasattr(main, 'initialize_model'):
            success = main.initialize_model()
            if success:
                print("✅ Model initialized successfully!")
            else:
                print("⚠️  Model initialization failed. Running in demo mode.")
        
        print("🌐 Starting Flask development server...")
        print("📱 Open your browser and go to: http://localhost:5000")
        print("📝 Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Run the Flask app
        main.app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except ImportError as e:
        print(f"❌ Error importing main.py: {e}")
        print("💡 Make sure all required packages are installed:")
        print("   pip install flask pandas numpy scikit-learn")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_app()