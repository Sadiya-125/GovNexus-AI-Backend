from flask import Flask, jsonify, Response, stream_with_context, request
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
import time
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import ML extensions
try:
    from ml_models import get_model, compute_metrics
    from app_ml_extensions import register_ml_routes
    ML_EXTENSIONS_AVAILABLE = True
except ImportError:
    ML_EXTENSIONS_AVAILABLE = False
    print("⚠️  ML extensions not available. Install required packages or check imports.")

# Import enhanced risk analysis
try:
    from enhanced_risk_analyzer import EnhancedRiskAnalyzer
    ENHANCED_RISK_AVAILABLE = True
except ImportError:
    ENHANCED_RISK_AVAILABLE = False
    print("⚠️  Enhanced risk analysis not available. Install required packages.")

# Import report generator
try:
    from report_generator import ReportGenerator
    REPORT_GENERATOR_AVAILABLE = True
except ImportError:
    REPORT_GENERATOR_AVAILABLE = False
    print("⚠️  Report generator not available. Install matplotlib and seaborn.")

# Import Twilio notifications
try:
    from twilio_notifications import (
        send_whatsapp_notification,
        format_supplier_notification,
        format_retailer_notification,
        format_bulk_notification,
        TWILIO_AVAILABLE
    )
except ImportError:
    TWILIO_AVAILABLE = False
    print("⚠️  Twilio notifications not available.")

# Import TensorFlow (optional for Keras model support)
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️  TensorFlow not available. Keras models will be skipped.")

load_dotenv()

app = Flask(__name__)
CORS(app)

# Ensure models directory exists
os.makedirs('models', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

def get_best_model():
    """
    Find and return the best performing model across all trained models
    Returns: tuple (model_data, model_path, model_type)
    """
    models_to_check = [
        ('models/profit_prediction_model.pkl', 'sklearn'),
        ('models/ensemble_model.pkl', 'sklearn'),
        ('models/transfer_learning_model_metadata.pkl', 'keras'),
        ('models/mlp_model_metadata.pkl', 'keras'),
        ('models/lstm_model_metadata.pkl', 'keras'),
        ('models/gru_model_metadata.pkl', 'keras'),
    ]

    best_model = None
    best_r2 = -float('inf')
    best_path = None
    best_type = None

    for model_path, model_format in models_to_check:
        if os.path.exists(model_path):
            try:
                model_data = joblib.load(model_path)
                metrics = model_data.get('metrics', {})
                r2_score = metrics.get('r2_score', -float('inf'))

                if r2_score > best_r2:
                    best_r2 = r2_score
                    best_model = model_data
                    best_path = model_path
                    best_type = model_format
            except:
                pass

    return best_model, best_path, best_type

# Database connection
def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        port=os.getenv('DB_PORT', 5432)
    )
    return conn

def send_log(message):
    """Helper function to send log messages"""
    return f"data: {message}\n\n"

def fetch_orders_data():
    """Fetch order data from database"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT
            o.id,
            o."orderDate",
            o."orderQty",
            o."costOfSales",
            o.sales,
            o.profit,
            o.year,
            p.name as product_name,
            p.category as product_category,
            p."unitCost" as "unitCost",
            p.price as price
        FROM "Order" o
        JOIN "Product" p ON o."productId" = p.id
        ORDER BY o."orderDate" DESC
    """)

    orders = cursor.fetchall()
    cursor.close()
    conn.close()

    # Convert to DataFrame
    df = pd.DataFrame(orders)

    # Convert Decimal columns to float
    decimal_columns = ['unitCost', 'price', 'costOfSales', 'sales', 'profit']
    for col in decimal_columns:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df

def preprocess_data(df):
    """Preprocess data for ML training"""
    # Extract datetime features
    df['orderDate'] = pd.to_datetime(df['orderDate'])
    df['month'] = df['orderDate'].dt.month
    df['day_of_week'] = df['orderDate'].dt.dayofweek
    df['quarter'] = df['orderDate'].dt.quarter

    # Encode categorical variables
    le_product = LabelEncoder()
    le_category = LabelEncoder()

    df['product_encoded'] = le_product.fit_transform(df['product_name'])
    df['category_encoded'] = le_category.fit_transform(df['product_category'].fillna('Unknown'))

    # Feature engineering
    # Note: unitCost and price now come from the Product table (joined in fetch_orders_data)
    df['price_per_unit'] = df['price']
    df['profit_margin'] = (df['profit'] / df['sales']) * 100
    df['profit_margin'] = df['profit_margin'].replace([np.inf, -np.inf], 0).fillna(0)

    # Save encoders
    joblib.dump(le_product, 'models/product_encoder.pkl')
    joblib.dump(le_category, 'models/category_encoder.pkl')

    return df

@app.route('/train', methods=['POST'])
def train_model():
    """Train ML model using order data with configurable options"""

    def generate_logs():
        try:
            # Get training configuration
            data = request.get_json() or {}
            use_checkpoint = data.get('useCheckpoint', False)
            epochs = data.get('epochs', 100)
            model_type = data.get('modelType', 'random_forest')

            # Validate epochs
            epochs = max(50, min(epochs, 500))

            yield send_log("🚀 Starting ML training pipeline...")
            time.sleep(0.5)

            yield send_log(f"Configuration: Model={model_type}, Epochs={epochs}, Checkpoint={use_checkpoint}")
            time.sleep(0.5)

            yield send_log("📊 Connecting to database...")
            time.sleep(0.5)

            yield send_log("📥 Fetching order data from database...")
            df = fetch_orders_data()
            yield send_log(f"✓ Found {len(df)} orders in database")
            time.sleep(0.5)

            if len(df) < 10:
                yield send_log("❌ Insufficient data. Please add at least 10 orders.")
                return

            yield send_log("🔧 Preprocessing data...")
            df = preprocess_data(df)
            time.sleep(0.5)

            # Select features for training
            feature_columns = ['unitCost', 'price', 'orderQty', 'year', 'month',
                             'day_of_week', 'quarter', 'product_encoded',
                             'category_encoded', 'price_per_unit']

            X = df[feature_columns]
            y = df['profit']  # Predicting profit

            yield send_log("- Handling missing values...")
            X = X.fillna(X.mean())
            time.sleep(0.3)

            yield send_log("- Scaling features...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            joblib.dump(scaler, 'models/scaler.pkl')
            time.sleep(0.3)

            yield send_log("- Splitting data into train/test sets (80/20)...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            yield send_log(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
            time.sleep(0.5)

            # Check for checkpoint (separate checkpoint for each model type)
            model = None
            start_epoch = 0
            checkpoint_path = f'checkpoints/{model_type}_checkpoint.pkl'

            if use_checkpoint and os.path.exists(checkpoint_path):
                yield send_log(f"📂 Loading {model_type} checkpoint...")
                checkpoint_data = joblib.load(checkpoint_path)

                # Verify checkpoint is for the correct model type
                checkpoint_model_type = checkpoint_data.get('model_type', '')
                if checkpoint_model_type == model_type:
                    model = checkpoint_data['model']
                    start_epoch = checkpoint_data.get('epoch', 0)
                    scaler = checkpoint_data.get('scaler', scaler)
                    yield send_log(f"✓ Resumed from epoch {start_epoch}")
                    yield send_log(f"   Previous Train R²: {checkpoint_data.get('train_score', 'N/A'):.4f}")
                    yield send_log(f"   Previous Test R²: {checkpoint_data.get('test_score', 'N/A'):.4f}")
                    time.sleep(0.5)
                else:
                    yield send_log(f"⚠️  Checkpoint is for {checkpoint_model_type}, starting fresh for {model_type}")
                    time.sleep(0.5)

            if model is None:
                yield send_log("🎯 Initializing new ML model...")

                # Select model based on type
                if model_type == 'random_forest':
                    model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1,
                        warm_start=True  # Enable incremental training
                    )
                    yield send_log("Model: Random Forest Regressor (warm_start enabled)")
                elif model_type == 'gradient_boosting':
                    model = GradientBoostingRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5,
                        random_state=42,
                        warm_start=True  # Enable incremental training
                    )
                    yield send_log("Model: Gradient Boosting Regressor (warm_start enabled)")
                else:
                    model = LinearRegression()
                    yield send_log("Model: Linear Regression")

                time.sleep(0.5)

            yield send_log("=" * 50)
            yield send_log("🔄 Training model...")

            # Training with actual incremental improvement
            if model_type in ['random_forest', 'gradient_boosting']:
                # Incremental training with increasing estimators
                total_epochs = epochs if not use_checkpoint else (epochs - start_epoch)

                # Calculate estimators per epoch
                estimators_per_epoch = max(1, 100 // epochs)

                for epoch in range(start_epoch, epochs):
                    progress = ((epoch - start_epoch + 1) / total_epochs) * 100

                    # Incrementally increase n_estimators for better training
                    current_estimators = min((epoch + 1) * estimators_per_epoch, 200)
                    model.n_estimators = current_estimators

                    # Train model
                    model.fit(X_train, y_train)

                    # Calculate metrics every 10 epochs or at start/end
                    if epoch % 10 == 0 or epoch == start_epoch or epoch == epochs - 1:
                        # Calculate metrics
                        y_pred_train = model.predict(X_train)
                        y_pred_test = model.predict(X_test)

                        train_score = r2_score(y_train, y_pred_train)
                        test_score = r2_score(y_test, y_pred_test)
                        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

                        yield send_log(f"Epoch {epoch+1}/{epochs} | Progress: {progress:.1f}% | Estimators: {current_estimators} | Train R²: {train_score:.4f} | Test R²: {test_score:.4f} | RMSE: {test_rmse:.2f}")

                        # Save checkpoint every 50 epochs
                        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
                            os.makedirs('checkpoints', exist_ok=True)
                            checkpoint_data = {
                                'model': model,
                                'scaler': scaler,
                                'epoch': epoch + 1,
                                'train_score': train_score,
                                'test_score': test_score,
                                'model_type': model_type,
                                'feature_columns': feature_columns,
                                'saved_at': datetime.now().isoformat()
                            }
                            joblib.dump(checkpoint_data, checkpoint_path)
                            yield send_log(f"💾 Checkpoint saved at epoch {epoch+1} → {checkpoint_path}")

                    time.sleep(0.05)
            else:
                # Simple linear regression - single fit
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                train_score = r2_score(y_train, y_pred_train)
                test_score = r2_score(y_test, y_pred_test)
                yield send_log(f"Model trained | Train R²: {train_score:.4f} | Test R²: {test_score:.4f}")
                time.sleep(1)

            yield send_log("=" * 50)
            yield send_log("📈 Evaluating model on test data...")
            time.sleep(0.5)

            # Final predictions
            y_pred = model.predict(X_test)

            # Calculate comprehensive metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100

            yield send_log(f"✓ R² Score: {r2:.4f}")
            yield send_log(f"✓ RMSE: {rmse:.2f}")
            yield send_log(f"✓ MAE: {mae:.2f}")
            yield send_log(f"✓ MAPE: {mape:.2f}%")
            time.sleep(0.5)

            yield send_log("💾 Saving model to disk...")
            model_data = {
                'model': model,
                'scaler': scaler,
                'feature_columns': feature_columns,
                'metrics': {
                    'r2_score': float(r2),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'mape': float(mape)
                },
                'trained_at': datetime.now().isoformat(),
                'training_samples': len(X_train),
                'model_type': model_type
            }
            joblib.dump(model_data, 'models/profit_prediction_model.pkl')
            time.sleep(0.5)

            yield send_log("✓ Model saved successfully at: ./models/profit_prediction_model.pkl")
            time.sleep(0.3)

            yield send_log("=" * 50)
            yield send_log("📊 Training Summary:")
            yield send_log(f"Total Orders Processed: {len(df)}")
            yield send_log(f"Training Samples: {len(X_train)}")
            yield send_log(f"Test Samples: {len(X_test)}")
            yield send_log(f"Model Type: {model_type}")
            yield send_log(f"Final R² Score: {r2:.4f} ({r2*100:.2f}% accuracy)")
            yield send_log(f"RMSE: ${rmse:.2f}")
            yield send_log(f"MAE: ${mae:.2f}")
            yield send_log("=" * 50)
            time.sleep(0.5)

            yield send_log("✅ Training completed successfully!")

        except Exception as e:
            yield send_log(f"❌ Error: {str(e)}")
            import traceback
            yield send_log(f"Traceback: {traceback.format_exc()}")

    return Response(
        stream_with_context(generate_logs()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using trained model"""
    try:
        # Load model
        if not os.path.exists('models/profit_prediction_model.pkl'):
            return jsonify({"error": "Model not trained yet"}), 400

        model_data = joblib.load('models/profit_prediction_model.pkl')
        model = model_data['model']
        scaler = model_data['scaler']
        feature_columns = model_data['feature_columns']

        # Get input data
        data = request.get_json()

        # Prepare features
        features = pd.DataFrame([data])[feature_columns]
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]

        return jsonify({
            "predicted_profit": float(prediction),
            "model_type": model_data.get('model_type', 'unknown'),
            "metrics": model_data.get('metrics', {})
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/eda', methods=['GET', 'POST'])
def get_eda_data():
    """Get EDA insights from order data"""
    try:
        df = fetch_orders_data()

        if len(df) == 0:
            return jsonify({"error": "No data available"}), 404

        # Convert orderDate to datetime
        df['orderDate'] = pd.to_datetime(df['orderDate'])

        # Basic statistics
        total_sales = float(df['sales'].sum())
        total_profit = float(df['profit'].sum())
        avg_profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0

        summary = {
            "total_orders": len(df),
            "total_sales": total_sales,
            "total_profit": total_profit,
            "avg_order_value": float(df['sales'].mean()),
            "avg_profit_margin": avg_profit_margin,
            "date_range": {
                "start": df['orderDate'].min().isoformat(),
                "end": df['orderDate'].max().isoformat()
            }
        }

        # Sales by category
        category_agg = df.groupby('product_category').agg({
            'sales': 'sum',
            'profit': 'sum',
            'id': 'count'
        }).reset_index()

        sales_by_category = []
        for _, row in category_agg.iterrows():
            sales_by_category.append({
                'category': row['product_category'] if pd.notna(row['product_category']) else 'Uncategorized',
                'total_sales': float(row['sales']),
                'total_profit': float(row['profit']),
                'order_count': int(row['id'])
            })

        # Monthly trends
        df['month_year'] = df['orderDate'].dt.to_period('M')
        monthly_agg = df.groupby('month_year').agg({
            'sales': 'sum',
            'profit': 'sum',
            'id': 'count'
        }).reset_index()

        monthly_trends = []
        for _, row in monthly_agg.iterrows():
            monthly_trends.append({
                'month': str(row['month_year']),
                'sales': float(row['sales']),
                'profit': float(row['profit']),
                'orders': int(row['id'])
            })

        # Top products
        product_agg = df.groupby(['product_name', 'product_category']).agg({
            'sales': 'sum',
            'profit': 'sum',
            'id': 'count'
        }).reset_index()
        product_agg = product_agg.nlargest(10, 'sales')

        top_products = []
        for _, row in product_agg.iterrows():
            total_sales_product = float(row['sales'])
            total_profit_product = float(row['profit'])
            profit_margin = (total_profit_product / total_sales_product * 100) if total_sales_product > 0 else 0

            top_products.append({
                'product_name': row['product_name'],
                'category': row['product_category'] if pd.notna(row['product_category']) else 'Uncategorized',
                'total_sales': total_sales_product,
                'total_profit': total_profit_product,
                'order_count': int(row['id']),
                'avg_profit_margin': profit_margin
            })

        return jsonify({
            "summary": summary,
            "sales_by_category": sales_by_category,
            "monthly_trends": monthly_trends,
            "top_products": top_products
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/optimize', methods=['GET', 'POST'])
def optimize_inventory():
    """Provide ML-powered inventory optimization recommendations with enhanced risk analysis"""
    try:
        user_state = None
        user_location = None
        use_external_data = True 

        if request.method == 'POST':
            data = request.get_json() or {}
            user_state = data.get('user_state')
            user_location = data.get('user_location')
            use_external_data = data.get('use_external_data', True)  # Opt-in
        else:
            user_state = request.args.get('user_state')
            user_location = request.args.get('user_location')
            use_external_data = request.args.get('use_external_data', 'true').lower() == 'true'

        df = fetch_orders_data()

        if len(df) < 5:
            return jsonify({"error": "Insufficient data for optimization"}), 400

        # Convert date to datetime
        df['orderDate'] = pd.to_datetime(df['orderDate'])

        # Set default location if not provided
        if not user_location and not user_state:
            user_location = "Hyderabad"
            user_state = "Telangana"

        # Initialize enhanced risk analyzer if available
        risk_analyzer = None
        
        if ENHANCED_RISK_AVAILABLE and use_external_data:
            risk_analyzer = EnhancedRiskAnalyzer()
            print(f"✓ Enhanced risk analysis enabled (Location: {user_location}, State: {user_state})")
        else:
            print("✓ Using basic risk analysis (fast)")

        # Get the best available model for predictions
        best_model_data, best_model_path, best_model_type = get_best_model()
        model_available = best_model_data is not None

        if model_available:
            model_name = os.path.basename(best_model_path).replace('_metadata.pkl', '').replace('.pkl', '')
            model_r2 = best_model_data.get('metrics', {}).get('r2_score', 0)

        # Aggregate by product with category
        product_stats = df.groupby(['product_name', 'product_category']).agg({
            'orderQty': ['mean', 'std', 'sum', 'count'],
            'sales': 'sum',
            'profit': ['sum', 'mean'],
            'unitCost': 'mean',
            'price': 'mean',
            'orderDate': ['min', 'max']
        }).reset_index()

        product_stats.columns = ['_'.join(col).strip('_') for col in product_stats.columns.values]

        product_stats = product_stats.sort_values('sales_sum', ascending=False)

        recommendations = []

        # Use fast basic analysis for all products (enhanced analysis disabled)
        max_enhanced_products = 0
        enhanced_count = 0

        total_products = len(product_stats)
        print(f"📊 Processing {total_products} products...")
        # if use_external_data and max_enhanced_products > 0:
        #     print(f"🔍 Enhanced analysis will be applied to top {max_enhanced_products} products by sales")
        #     print("⏱️  Remaining products will use fast basic analysis")

        for _, row in product_stats.iterrows():
            product_name = row['product_name']
            category = row['product_category'] if pd.notna(row['product_category']) else 'Uncategorized'

            # Current metrics
            avg_demand = float(row['orderQty_mean'])
            std_demand = float(row['orderQty_std']) if pd.notna(row['orderQty_std']) else 0
            total_orders = int(row['orderQty_count'])

            # Calculate safety stock (95% service level)
            safety_stock = std_demand * 1.65

            # Calculate reorder point
            reorder_point = avg_demand + safety_stock

            # Calculate optimal order quantity
            if total_orders > 0:
                demand_rate = float(row['orderQty_sum']) / total_orders
                optimal_qty = max(int(demand_rate * 1.5), 1)
            else:
                optimal_qty = 1

            # Predict future demand using best ML model if available
            predicted_demand = avg_demand  # Default to current avg
            ml_model_used = None
            if model_available:
                try:
                    scaler = best_model_data['scaler']

                    # Create features for prediction
                    features = pd.DataFrame([{
                        'unitCost': float(row['unitCost_mean']),
                        'price': float(row['price_mean']),
                        'orderQty': optimal_qty,
                        'year': datetime.now().year,
                        'month': datetime.now().month,
                        'day_of_week': datetime.now().weekday(),
                        'quarter': (datetime.now().month - 1) // 3 + 1,
                        'product_encoded': 0,
                        'category_encoded': 0,
                        'price_per_unit': float(row['price_mean'])
                    }])

                    # Scale features
                    features_scaled = scaler.transform(features)

                    # Predict based on model type
                    predicted_profit = None
                    if best_model_type == 'sklearn':
                        if 'model' in best_model_data:
                            model = best_model_data['model']
                            predicted_profit = float(model.predict(features_scaled)[0])
                        elif 'ensemble' in best_model_data:
                            # Handle ensemble models
                            model = best_model_data['ensemble']
                            predicted_profit = float(model.predict(features_scaled)[0])
                    else:  # keras model
                        # Load the keras model only if TensorFlow is available
                        if TENSORFLOW_AVAILABLE:
                            keras_model_path = best_model_path.replace('_metadata.pkl', '.h5')
                            if os.path.exists(keras_model_path):
                                from tensorflow.keras.losses import MeanSquaredError
                                from tensorflow.keras.metrics import MeanAbsoluteError
                                custom_objects = {
                                    'mse': MeanSquaredError(),
                                    'mae': MeanAbsoluteError(),
                                    'MeanSquaredError': MeanSquaredError,
                                    'MeanAbsoluteError': MeanAbsoluteError
                                }
                                model = tf.keras.models.load_model(keras_model_path, custom_objects=custom_objects, compile=False)
                                predicted_profit = float(model.predict(features_scaled, verbose=0)[0][0])
                            else:
                                predicted_profit = None
                        else:
                            predicted_profit = None

                    if predicted_profit is not None:
                        # Estimate predicted demand based on predicted profit
                        avg_profit_per_unit = float(row['profit_mean'])
                        if avg_profit_per_unit > 0:
                            predicted_demand = max((predicted_profit / avg_profit_per_unit) * avg_demand, avg_demand * 0.5)
                        else:
                            predicted_demand = avg_demand

                        ml_model_used = model_name

                except Exception as e:
                    import traceback
                    error_type = type(e).__name__
                    error_details = str(e)
                    stack_info = traceback.format_exc()

                    print("⚠️ Prediction Error Details:")
                    print(f"Error Type: {error_type}")
                    print(f"Error Message: {error_details}")
                    print("Stack Trace:")
                    print(stack_info)
                    print(f"Fallback applied: Using average demand ({avg_demand}) instead of predicted value.\n")

                    predicted_demand = avg_demand

            # Use basic analysis for all products (enhanced analysis disabled for performance)
            should_use_external_data = False
            # if risk_analyzer and enhanced_count < max_enhanced_products:
            #     should_use_external_data = True
            #     enhanced_count += 1
            #     print(f"✓ Enhanced analysis [{enhanced_count}/{max_enhanced_products}]: {product_name}")

            if risk_analyzer and should_use_external_data:
                risk_analysis = risk_analyzer.analyze_product_risk(
                    product_name=product_name,
                    category=category,
                    avg_demand=avg_demand,
                    std_demand=std_demand,
                    predicted_demand=predicted_demand,
                    safety_stock=safety_stock,
                    user_state=user_state,
                    user_location=user_location,
                    use_external_data=True  # Enable external APIs for this product
                )

                risk_level = risk_analysis["risk_level"]
                trend = risk_analysis["trend"]
                demand_change_pct = risk_analysis["demand_change_pct"]
                risk_score = risk_analysis["risk_score"]
                external_insights = risk_analysis.get("external_insights", {})
                risk_explanations = risk_analyzer.get_risk_explanation(risk_analysis)
            elif risk_analyzer and not should_use_external_data:
                # Use risk analyzer but WITHOUT external API calls for remaining products
                risk_analysis = risk_analyzer.analyze_product_risk(
                    product_name=product_name,
                    category=category,
                    avg_demand=avg_demand,
                    std_demand=std_demand,
                    predicted_demand=predicted_demand,
                    safety_stock=safety_stock,
                    user_state=user_state,
                    user_location=user_location,
                    use_external_data=False  # Disable external APIs for speed
                )

                risk_level = risk_analysis["risk_level"]
                trend = risk_analysis["trend"]
                demand_change_pct = risk_analysis["demand_change_pct"]
                risk_score = risk_analysis["risk_score"]
                external_insights = {}
                risk_explanations = []
            else:
                # Basic risk calculation (fallback)
                demand_change_pct = ((predicted_demand - avg_demand) / avg_demand * 100) if avg_demand > 0 else 0

                if predicted_demand > avg_demand * 1.15:  # 15% increase
                    trend = "increasing"
                elif predicted_demand < avg_demand * 0.85:  # 15% decrease
                    trend = "decreasing"
                else:
                    trend = "stable"

                # Improved risk level calculation with multiple factors
                cv = (std_demand / avg_demand) if avg_demand > 0 else 0  # Coefficient of variation

                # Calculate risk score based on multiple factors
                risk_score = 0

                # Factor 1: Demand variability (CV)
                if cv > 0.6:
                    risk_score += 3
                elif cv > 0.4:
                    risk_score += 2
                elif cv > 0.2:
                    risk_score += 1

                # Factor 2: Demand trend (increasing demand = higher risk of stockout)
                if trend == "increasing" and demand_change_pct > 25:
                    risk_score += 3
                elif trend == "increasing" and demand_change_pct > 15:
                    risk_score += 2
                elif trend == "decreasing":
                    risk_score -= 1  # Decreasing demand = lower stockout risk

                # Factor 3: Safety stock adequacy
                safety_ratio = safety_stock / avg_demand if avg_demand > 0 else 0
                if safety_ratio < 0.3:  # Less than 30% safety stock
                    risk_score += 2
                elif safety_ratio < 0.5:
                    risk_score += 1

                # Determine risk level based on score
                if risk_score >= 5:
                    risk_level = "high"
                elif risk_score >= 2:
                    risk_level = "medium"
                else:
                    risk_level = "low"

                external_insights = {}
                risk_explanations = []

            rec = {
                "product_name": product_name,
                "category": category,
                "current_avg_demand": avg_demand,
                "predicted_demand": predicted_demand,
                "demand_trend": trend,
                "reorder_point": reorder_point,
                "optimal_order_qty": optimal_qty,
                "safety_stock": safety_stock,
                "risk_level": risk_level,
                "risk_score": risk_score,
                "total_sales": float(row['sales_sum']),
                "total_profit": float(row['profit_sum'])
            }

            # Add enhanced insights if available
            if risk_analyzer:
                trends_chart = external_insights.get("trends_chart")
                rec["external_insights"] = {
                    "seasonality_index": external_insights.get("seasonality_index", 1.0),
                    "state_interest": external_insights.get("state_interest", 0),
                    "sentiment": external_insights.get("sentiment_analysis", {}).get("overall_sentiment", "neutral"),
                    "demand_signal": external_insights.get("sentiment_analysis", {}).get("demand_signal", "stable"),
                    "news_count": external_insights.get("news_count", 0),
                    "upcoming_holidays": external_insights.get("upcoming_holidays_count", 0),
                    "pricing_competitive": external_insights.get("pricing", {}).get("success", False)
                }
                rec["risk_factors"] = risk_explanations

                # Add Google Trends visualization data
                if trends_chart and trends_chart.get("success"):
                    rec["trends_visualization"] = {
                        "chart_data": trends_chart.get("chart_data", []),
                        "labels": trends_chart.get("labels", []),
                        "values": trends_chart.get("values", []),
                        "average": trends_chart.get("average", 0)
                    }
                else:
                    rec["trends_visualization"] = None
            else:
                rec["external_insights"] = None
                rec["risk_factors"] = []
                rec["trends_visualization"] = None

            recommendations.append(rec)

        # Sort by risk level (critical/high first) then by total sales
        risk_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda x: (risk_order.get(x['risk_level'], 3), -x['total_sales']))

        # Calculate summary
        summary = {
            "total_products": len(recommendations),
            "critical_risk_products": sum(1 for r in recommendations if r['risk_level'] == "critical"),
            "high_risk_products": sum(1 for r in recommendations if r['risk_level'] == "high"),
            "medium_risk_products": sum(1 for r in recommendations if r['risk_level'] == "medium"),
            "low_risk_products": sum(1 for r in recommendations if r['risk_level'] == "low"),
            "total_recommended_stock": sum(int(r['optimal_order_qty']) for r in recommendations),
            "ml_model_used": model_name if model_available else None,
            "ml_model_r2_score": model_r2 if model_available else None,
            "enhanced_analysis_enabled": risk_analyzer is not None,
            "user_location": user_location,
            "user_state": user_state
        }

        return jsonify({
            "recommendations": recommendations,  # Return all recommendations, pagination handled by frontend
            "summary": summary
        })

    except Exception as e:
        import traceback
        print(f"Optimization error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_exists = os.path.exists('models/profit_prediction_model.pkl')
    checkpoint_exists = os.path.exists('checkpoints/model_checkpoint.pkl')

    return jsonify({
        "status": "healthy",
        "message": "Flask backend is running",
        "model_trained": model_exists,
        "checkpoint_available": checkpoint_exists,
        "twilio_available": TWILIO_AVAILABLE
    })

@app.route('/supplier/notify', methods=['POST'])
def notify_supplier():
    """Send WhatsApp notification to supplier"""
    try:
        if not TWILIO_AVAILABLE:
            return jsonify({
                "error": "Twilio not configured. Install twilio package and set environment variables."
            }), 503

        data = request.get_json()
        to_number = data.get('to_number')
        product_name = data.get('product_name')
        category = data.get('category')
        predicted_demand = data.get('predicted_demand', 0)
        optimal_qty = data.get('optimal_qty', 0)

        if not to_number:
            return jsonify({"error": "Missing 'to_number' field"}), 400

        # Format the message (removed action_required parameter)
        message = format_supplier_notification(
            product_name=product_name,
            category=category,
            predicted_demand=predicted_demand,
            optimal_qty=optimal_qty
        )

        # Send notification
        result = send_whatsapp_notification(to_number, message)

        if result['success']:
            return jsonify({
                "success": True,
                "message": "Notification sent successfully",
                "message_sid": result['message_sid'],
                "to": result['to']
            })
        else:
            return jsonify({
                "success": False,
                "error": result['error']
            }), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/supplier/notify-bulk', methods=['POST'])
def notify_supplier_bulk():
    """Send bulk WhatsApp notification for a category (splits into multiple messages if needed)"""
    try:
        if not TWILIO_AVAILABLE:
            return jsonify({
                "error": "Twilio not configured"
            }), 503

        data = request.get_json()
        to_number = data.get('to_number')
        category = data.get('category')
        products_list = data.get('products_list', [])

        if not to_number or not category:
            return jsonify({"error": "Missing required fields"}), 400

        if not products_list:
            return jsonify({"error": "No products found for this category"}), 400

        # Format the messages with detailed product list (returns list of messages)
        messages = format_bulk_notification(
            category=category,
            products_list=products_list
        )

        # Send all messages
        results = []
        all_successful = True
        for message in messages:
            result = send_whatsapp_notification(to_number, message)
            results.append(result)
            if not result['success']:
                all_successful = False

        if all_successful:
            return jsonify({
                "success": True,
                "message": f"Bulk notification sent successfully ({len(messages)} message(s))",
                "messages_sent": len(messages),
                "message_sids": [r['message_sid'] for r in results],
                "to": to_number
            })
        else:
            failed_count = sum(1 for r in results if not r['success'])
            return jsonify({
                "success": False,
                "message": f"{failed_count} out of {len(messages)} message(s) failed",
                "results": results
            }), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/retailer/notify', methods=['POST'])
def notify_retailer():
    """Send WhatsApp notification to retailer about supplier's decision"""
    try:
        print("\n" + "="*60)
        print("🔔 RETAILER NOTIFICATION REQUEST RECEIVED")
        print("="*60)

        if not TWILIO_AVAILABLE:
            print("❌ Twilio not available")
            return jsonify({
                "error": "Twilio not configured. Install twilio package and set environment variables."
            }), 503

        data = request.get_json()
        print(f"📥 Request data: {data}")

        to_number = data.get('to_number')
        product_name = data.get('product_name')
        status = data.get('status')
        requested_qty = data.get('requested_qty')
        supplier_name = data.get('supplier_name')

        print(f"📱 Sending to: {to_number}")
        print(f"📦 Product: {product_name}")
        print(f"✅/❌ Status: {status}")
        print(f"📊 Quantity: {requested_qty}")
        print(f"🏢 Supplier: {supplier_name}")

        if not to_number or not product_name or not status:
            print("❌ Missing required fields")
            return jsonify({"error": "Missing required fields: to_number, product_name, status"}), 400

        if status not in ['accepted', 'rejected']:
            print(f"❌ Invalid status: {status}")
            return jsonify({"error": "Invalid status. Must be 'accepted' or 'rejected'"}), 400

        # Format the message
        print("📝 Formatting message...")
        message = format_retailer_notification(
            product_name=product_name,
            status=status,
            requested_qty=requested_qty,
            supplier_name=supplier_name
        )
        print(f"📄 Message:\n{message}\n")

        # Send notification
        print("📤 Sending WhatsApp notification...")
        result = send_whatsapp_notification(to_number, message)
        print(f"📥 Twilio result: {result}")

        if result['success']:
            print(f"✅ SUCCESS! Message SID: {result['message_sid']}")
            print("="*60 + "\n")
            return jsonify({
                "success": True,
                "message": "Retailer notification sent successfully",
                "message_sid": result['message_sid'],
                "to": result['to']
            })
        else:
            print(f"❌ FAILED! Error: {result['error']}")
            print("="*60 + "\n")
            return jsonify({
                "success": False,
                "error": result['error']
            }), 500

    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        print("="*60 + "\n")
        return jsonify({"error": str(e)}), 500

@app.route('/orders/count', methods=['GET'])
def get_order_count():
    """Get total number of orders"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM "Order"')
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return jsonify({"count": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==================== REPORT GENERATION ENDPOINTS ====================

@app.route('/reports/eda', methods=['GET', 'POST'])
def generate_eda_report():
    """Generate EDA report"""
    try:
        if not REPORT_GENERATOR_AVAILABLE:
            return jsonify({
                "error": "Report generator not available. Install matplotlib and seaborn."
            }), 503

        df = fetch_orders_data()

        if len(df) < 1:
            return jsonify({"error": "Insufficient data for report generation"}), 400

        generator = ReportGenerator()
        report = generator.generate_eda_report(df, output_format='html')

        return jsonify({
            "success": True,
            "filename": report['filename'],
            "content": report['content'],
            "format": report['format']
        })

    except Exception as e:
        import traceback
        print(f"EDA report error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/reports/optimization', methods=['GET', 'POST'])
def generate_optimization_report():
    """Generate optimization report"""
    try:
        print("Starting...")
        if not REPORT_GENERATOR_AVAILABLE:
            return jsonify({
                "error": "Report generator not available. Install matplotlib and seaborn."
            }), 503

        # Get optimization data
        df = fetch_orders_data()

        if len(df) < 5:
            return jsonify({"error": "Insufficient data for optimization report"}), 400

        # Get user location if provided
        user_state = None
        user_location = None

        if request.method == 'POST':
            data = request.get_json() or {}
            user_state = data.get('user_state')
            user_location = data.get('user_location')
        else:
            user_state = request.args.get('user_state')
            user_location = request.args.get('user_location')

        # Initialize enhanced risk analyzer if available
        risk_analyzer = None
        if ENHANCED_RISK_AVAILABLE:
            print("Enhanced risk analyzer available")
            risk_analyzer = EnhancedRiskAnalyzer()

        # Convert date to datetime
        df['orderDate'] = pd.to_datetime(df['orderDate'])

        # Get the best available model for predictions
        best_model_data, best_model_path, best_model_type = get_best_model()
        model_available = best_model_data is not None

        if model_available:
            model_name = os.path.basename(best_model_path).replace('_metadata.pkl', '').replace('.pkl', '')
            model_r2 = best_model_data.get('metrics', {}).get('r2_score', 0)

        # Aggregate by product with category
        product_stats = df.groupby(['product_name', 'product_category']).agg({
            'orderQty': ['mean', 'std', 'sum', 'count'],
            'sales': 'sum',
            'profit': ['sum', 'mean'],
            'unitCost': 'mean',
            'price': 'mean',
            'orderDate': ['min', 'max']
        }).reset_index()

        product_stats.columns = ['_'.join(col).strip('_') for col in product_stats.columns.values]

        recommendations = []

        for _, row in product_stats.iterrows():
            product_name = row['product_name']
            category = row['product_category'] if pd.notna(row['product_category']) else 'Uncategorized'

            # Current metrics
            avg_demand = float(row['orderQty_mean'])
            std_demand = float(row['orderQty_std']) if pd.notna(row['orderQty_std']) else 0
            total_orders = int(row['orderQty_count'])

            # Calculate safety stock (95% service level)
            safety_stock = std_demand * 1.65

            # Calculate reorder point
            reorder_point = avg_demand + safety_stock

            # Calculate optimal order quantity
            if total_orders > 0:
                demand_rate = float(row['orderQty_sum']) / total_orders
                optimal_qty = max(int(demand_rate * 1.5), 1)
            else:
                optimal_qty = 1

            # Predict future demand
            predicted_demand = avg_demand

            # Use enhanced risk analyzer if available
            if risk_analyzer:
                print("Using enhanced risk analyzer")
                risk_analysis = risk_analyzer.analyze_product_risk(
                    product_name=product_name,
                    category=category,
                    avg_demand=avg_demand,
                    std_demand=std_demand,
                    predicted_demand=predicted_demand,
                    safety_stock=safety_stock,
                    user_state=user_state,
                    user_location=user_location,
                    use_external_data=True
                )

                risk_level = risk_analysis["risk_level"]
                trend = risk_analysis["trend"]
                risk_score = risk_analysis["risk_score"]
            else:
                # Basic risk calculation
                cv = (std_demand / avg_demand) if avg_demand > 0 else 0
                risk_score = 0

                if cv > 0.6:
                    risk_score += 3
                elif cv > 0.4:
                    risk_score += 2
                elif cv > 0.2:
                    risk_score += 1

                if risk_score >= 5:
                    risk_level = "high"
                elif risk_score >= 2:
                    risk_level = "medium"
                else:
                    risk_level = "low"

                trend = "stable"

            rec = {
                "product_name": product_name,
                "category": category,
                "current_avg_demand": avg_demand,
                "predicted_demand": predicted_demand,
                "demand_trend": trend,
                "reorder_point": reorder_point,
                "optimal_order_qty": optimal_qty,
                "safety_stock": safety_stock,
                "risk_level": risk_level,
                "risk_score": risk_score,
                "total_sales": float(row['sales_sum']),
                "total_profit": float(row['profit_sum'])
            }
            recommendations.append(rec)

        # Sort by risk level
        risk_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda x: (risk_order.get(x['risk_level'], 3), -x['total_sales']))

        # Calculate summary
        summary = {
            "total_products": len(recommendations),
            "critical_risk_products": sum(1 for r in recommendations if r['risk_level'] == "critical"),
            "high_risk_products": sum(1 for r in recommendations if r['risk_level'] == "high"),
            "medium_risk_products": sum(1 for r in recommendations if r['risk_level'] == "medium"),
            "low_risk_products": sum(1 for r in recommendations if r['risk_level'] == "low"),
            "total_recommended_stock": sum(int(r['optimal_order_qty']) for r in recommendations),
            "ml_model_used": model_name if model_available else None,
            "ml_model_r2_score": model_r2 if model_available else None,
            "enhanced_analysis_enabled": risk_analyzer is not None,
            "user_location": user_location,
            "user_state": user_state
        }
        print("Generating optimization report...")

        # Generate report
        generator = ReportGenerator()
        report = generator.generate_optimization_report(recommendations, summary, output_format='html')
        print(report)
        return jsonify({
            "success": True,
            "filename": report['filename'],
            "content": report['content'],
            "format": report['format']
        })

    except Exception as e:
        import traceback
        print(f"Optimization report error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/reports/send-whatsapp', methods=['POST'])
def send_report_whatsapp():
    """Send detailed report via WhatsApp"""
    try:
        if not TWILIO_AVAILABLE:
            return jsonify({
                "error": "Twilio not configured"
            }), 503

        data = request.get_json()
        to_number = data.get('to_number')
        report_type = data.get('report_type', 'optimization')  # 'eda' or 'optimization'
        report_data = data.get('report_data')  # Optional: pre-generated report data
        user_state = data.get('user_state')
        user_location = data.get('user_location')

        if not to_number:
            return jsonify({"error": "Missing 'to_number' field"}), 400

        # Generate detailed report message
        message = f"📊 *Inventory {report_type.upper()} Report*\n\n"
        message += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

        if user_location and user_state:
            message += f"Location: {user_location}, {user_state}\n\n"

        # Fetch optimization data if not provided
        if not report_data:
            try:
                conn = get_db_connection()
                cursor = conn.cursor(cursor_factory=RealDictCursor)

                # Get summary statistics
                cursor.execute("""
                    SELECT
                        p.category,
                        p.name as product_name,
                        COALESCE(AVG(o."orderQty"), 0) as avg_demand,
                        COALESCE(SUM(o.sales), 0) as total_sales,
                        COALESCE(SUM(o.profit), 0) as total_profit
                    FROM "Product" p
                    LEFT JOIN "Order" o ON p.id = o."productId"
                    GROUP BY p.id, p.name, p.category
                    ORDER BY total_sales DESC
                    LIMIT 5
                """)

                top_products = cursor.fetchall()
                cursor.close()
                conn.close()

                # Build report from database data
                message += "📦 *Top 5 Products by Sales:*\n\n"
                for idx, product in enumerate(top_products, 1):
                    message += f"{idx}. *{product['product_name']}*\n"
                    message += f"   Category: {product['category'] or 'N/A'}\n"
                    message += f"   Avg Demand: {int(product['avg_demand'])} units\n"
                    message += f"   Total Sales: ${float(product['total_sales']):.2f}\n"
                    message += f"   Total Profit: ${float(product['total_profit']):.2f}\n\n"

            except Exception as e:
                print(f"Error fetching report data: {e}")
                message += "Unable to fetch detailed data. Please check the dashboard.\n\n"
        else:
            # Use provided report data to create summary
            message += "📋 *Report Summary:*\n\n"
            message += f"For detailed analysis, please check the dashboard or email report.\n\n"

        message += "🤖 Generated with Inventory Optimization System"

        # Send report
        result = send_whatsapp_notification(to_number, message)

        if result['success']:
            return jsonify({
                "success": True,
                "message": "Report sent successfully via WhatsApp",
                "message_sid": result['message_sid'],
                "to": result['to']
            })
        else:
            return jsonify({
                "success": False,
                "error": result['error']
            }), 500

    except Exception as e:
        import traceback
        print(f"WhatsApp report error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/reports/send-email', methods=['POST'])
def send_report_email():
    """Send report via email using Node.js email service"""
    try:
        data = request.get_json()
        to_email = data.get('to_email')
        report_type = data.get('report_type', 'optimization')  # 'eda' or 'optimization'
        report_html = data.get('report_html')
        user_state = data.get('user_state')
        user_location = data.get('user_location')

        if not to_email:
            return jsonify({"error": "Missing 'to_email' field"}), 400

        if not report_html:
            return jsonify({"error": "Missing 'report_html' field. Generate report first."}), 400

        # Forward request to Node.js email service
        import requests as http_requests

        email_service_url = os.getenv('EMAIL_SERVICE_URL', 'http://localhost:5001/send-report')

        email_payload = {
            'to_email': to_email,
            'report_type': report_type,
            'report_html': report_html,
            'user_state': user_state,
            'user_location': user_location
        }

        response = http_requests.post(email_service_url, json=email_payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            return jsonify({
                "success": True,
                "message": "Report sent via email successfully",
                "to": to_email,
                "filename": result.get('filename')
            })
        else:
            error_data = response.json()
            return jsonify({
                "success": False,
                "error": error_data.get('error', 'Email service error')
            }), 500

    except http_requests.exceptions.ConnectionError:
        return jsonify({
            "error": "Email service not available. Please ensure Node.js email service is running on port 5001."
        }), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Register ML extension routes
if ML_EXTENSIONS_AVAILABLE:
    try:
        register_ml_routes(app, fetch_orders_data, preprocess_data)
        print("✅ ML Extensions loaded successfully")
    except Exception as e:
        print(f"⚠️  Could not register ML extensions: {e}")

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Flask ML Backend Server Starting...")
    print("=" * 60)
    print("📍 Server URL: http://localhost:5000")
    print("\n📚 Available Endpoints:")
    print("  POST /train - Train ML model with options")
    print("  POST /predict - Make profit predictions")
    print("  GET  /eda - Get exploratory data analysis insights")
    print("  GET/POST /optimize - Get inventory optimization recommendations")
    print("  GET  /health - Health check")
    print("  GET  /orders/count - Get order count")

    if TWILIO_AVAILABLE:
        print("\n📱 Supplier Notification Endpoints:")
        print("  POST /supplier/notify - Send WhatsApp to supplier")
        print("  POST /supplier/notify-bulk - Send bulk category notification")

    if ML_EXTENSIONS_AVAILABLE:
        print("\n🔬 Advanced ML Endpoints:")
        print("  POST /models/compare - Compare multiple models")
        print("  POST /models/ensemble - Train ensemble model")
        print("  GET  /models/list - List all trained models")
        print("  GET  /models/best - Get best performing model")
        print("  POST /models/predict/advanced - Advanced predictions")
        print("  GET  /inventory/recommendations - ML-powered recommendations")
        print("  POST /models/transfer-learning - Train model with transfer learning")
        print("  POST /models/deep-learning - Train deep learning models (LSTM, GRU, MLP)")

    print("=" * 60)
    print("\n✨ Server ready to accept connections...")
    app.run(debug=True, port=5000, threaded=True)
