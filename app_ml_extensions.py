"""
ML Extensions for Flask App
Advanced model training, comparison, and transfer learning features
"""

from flask import jsonify, request, Response, stream_with_context
import time
import numpy as np
import pandas as pd
from ml_models import get_model, compute_metrics, ModelEnsemble, create_profit_features
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime

# Import deep learning modules
try:
    from deep_learning_models import (
        TF_AVAILABLE,
        build_lstm_model,
        build_gru_model,
        build_mlp_model,
        transfer_learning_model,
        get_callbacks,
        train_deep_model
    )
except ImportError:
    TF_AVAILABLE = False


def send_log(message):
    """Helper function to send log messages for streaming"""
    return f"data: {message}\n\n"


def register_ml_routes(app, fetch_orders_data, preprocess_data, prisma=None):
    """Register ML routes to Flask app"""

    @app.route('/models/compare', methods=['POST'])
    def compare_models():
        """Compare multiple ML models and return performance metrics"""
        def generate_logs():
            try:
                data = request.get_json() or {}
                model_types = data.get('models', ['random_forest', 'gradient_boosting', 'linear_regression'])

                yield send_log("🔬 Starting Model Comparison...")
                time.sleep(0.5)

                yield send_log(f"📊 Models to compare: {', '.join(model_types)}")
                time.sleep(0.5)

                yield send_log("📥 Fetching data...")
                df = fetch_orders_data()

                if len(df) < 20:
                    yield send_log("❌ Insufficient data for comparison (need at least 20 orders)")
                    return

                yield send_log(f"✓ Loaded {len(df)} orders")
                time.sleep(0.3)

                yield send_log("🔧 Preprocessing data...")
                df = preprocess_data(df)

                # Feature selection
                feature_columns = ['unitCost', 'price', 'orderQty', 'year', 'month',
                                 'day_of_week', 'quarter', 'product_encoded',
                                 'category_encoded', 'price_per_unit']

                X = df[feature_columns]
                y = df['profit']

                X = X.fillna(X.mean())
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )

                yield send_log(f"✓ Data split: {len(X_train)} train, {len(X_test)} test")
                time.sleep(0.3)

                results = []
                for model_type in model_types:
                    try:
                        yield send_log(f"\n🎯 Training {model_type}...")
                        time.sleep(0.3)

                        model = get_model(model_type)
                        model.fit(X_train, y_train)

                        # Predictions
                        y_pred_train = model.predict(X_train)
                        y_pred_test = model.predict(X_test)

                        # Metrics
                        train_metrics = compute_metrics(y_train, y_pred_train)
                        test_metrics = compute_metrics(y_test, y_pred_test)

                        # Cross-validation score
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

                        result = {
                            'model_type': model_type,
                            'train_metrics': train_metrics,
                            'test_metrics': test_metrics,
                            'cv_score_mean': float(np.mean(cv_scores)),
                            'cv_score_std': float(np.std(cv_scores)),
                            'training_time': time.time()
                        }
                        results.append(result)

                        yield send_log(f"✓ {model_type} - R²: {test_metrics['r2_score']:.4f}, RMSE: {test_metrics['rmse']:.2f}")
                        time.sleep(0.3)

                    except Exception as e:
                        yield send_log(f"❌ Error training {model_type}: {str(e)}")

                # Find best model
                if results:
                    best_model = max(results, key=lambda x: x['test_metrics']['r2_score'])
                    yield send_log(f"\n🏆 Best Model: {best_model['model_type']}")
                    yield send_log(f"   R² Score: {best_model['test_metrics']['r2_score']:.4f}")
                    yield send_log(f"   RMSE: {best_model['test_metrics']['rmse']:.2f}")

                # Save comparison results
                comparison_result = {
                    'timestamp': datetime.now().isoformat(),
                    'models_compared': len(results),
                    'results': results,
                    'best_model': best_model['model_type'] if results else None
                }

                joblib.dump(comparison_result, 'models/comparison_results.pkl')
                yield send_log("\n✅ Comparison complete! Results saved.")

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

    @app.route('/models/ensemble', methods=['POST'])
    def train_ensemble():
        """Train an ensemble of models"""
        def generate_logs():
            try:
                data = request.get_json() or {}
                model_types = data.get('models', ['random_forest', 'gradient_boosting', 'ridge'])

                yield send_log("🎭 Training Ensemble Model...")
                time.sleep(0.5)

                yield send_log("📥 Fetching data...")
                df = fetch_orders_data()
                df = preprocess_data(df)

                feature_columns = ['unitCost', 'price', 'orderQty', 'year', 'month',
                                 'day_of_week', 'quarter', 'product_encoded',
                                 'category_encoded', 'price_per_unit']

                X = df[feature_columns].fillna(df[feature_columns].mean())
                y = df['profit']

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )

                # Create ensemble
                yield send_log(f"Creating ensemble with: {', '.join(model_types)}")
                models = [(name, get_model(name)) for name in model_types]
                ensemble = ModelEnsemble(models)

                yield send_log("🔄 Training ensemble models...")
                ensemble.fit(X_train, y_train)

                # Evaluate
                y_pred = ensemble.predict(X_test)
                metrics = compute_metrics(y_test, y_pred)

                yield send_log(f"✓ Ensemble R² Score: {metrics['r2_score']:.4f}")
                yield send_log(f"✓ Ensemble RMSE: {metrics['rmse']:.2f}")

                # Save ensemble (include both 'ensemble' and 'model' keys for compatibility)
                model_data = {
                    'model': ensemble,  # For compatibility with optimization endpoint
                    'ensemble': ensemble,
                    'scaler': scaler,
                    'feature_columns': feature_columns,
                    'metrics': metrics,
                    'model_types': model_types,
                    'model_type': 'ensemble',
                    'trained_at': datetime.now().isoformat()
                }
                joblib.dump(model_data, 'models/ensemble_model.pkl')

                yield send_log("✅ Ensemble model trained and saved!")

            except Exception as e:
                yield send_log(f"❌ Error: {str(e)}")

        return Response(
            stream_with_context(generate_logs()),
            mimetype='text/event-stream'
        )

    @app.route('/models/list', methods=['GET'])
    def list_models():
        """List all available trained models"""
        try:
            models_dir = 'models'
            models = []

            if os.path.exists(models_dir):
                for filename in os.listdir(models_dir):
                    if filename.endswith('.pkl') and 'model' in filename:
                        filepath = os.path.join(models_dir, filename)
                        try:
                            model_data = joblib.load(filepath)
                            info = {
                                'name': filename,
                                'type': model_data.get('model_type', 'unknown'),
                                'trained_at': model_data.get('trained_at', 'unknown'),
                                'metrics': model_data.get('metrics', {}),
                                'file_size': os.path.getsize(filepath)
                            }
                            models.append(info)
                        except:
                            pass

            return jsonify({
                'models': models,
                'count': len(models)
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/models/best', methods=['GET'])
    def get_best_model():
        """Get the best performing model based on comparison"""
        try:
            comparison_file = 'models/comparison_results.pkl'

            if not os.path.exists(comparison_file):
                return jsonify({"error": "No model comparison available. Run comparison first."}), 404

            comparison_data = joblib.load(comparison_file)

            return jsonify({
                'best_model': comparison_data.get('best_model'),
                'timestamp': comparison_data.get('timestamp'),
                'models_compared': comparison_data.get('models_compared'),
                'results': comparison_data.get('results', [])
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/models/predict/advanced', methods=['POST'])
    def predict_advanced():
        """
        Advanced prediction with multiple models and confidence intervals
        """
        try:
            data = request.get_json()
            use_ensemble = data.get('use_ensemble', False)

            # Load appropriate model
            if use_ensemble and os.path.exists('models/ensemble_model.pkl'):
                model_data = joblib.load('models/ensemble_model.pkl')
                model = model_data['ensemble']
            elif os.path.exists('models/profit_prediction_model.pkl'):
                model_data = joblib.load('models/profit_prediction_model.pkl')
                model = model_data['model']
            else:
                return jsonify({"error": "No trained model available"}), 400

            scaler = model_data['scaler']
            feature_columns = model_data['feature_columns']

            # Prepare features
            features = pd.DataFrame([data])[feature_columns]
            features_scaled = scaler.transform(features)

            # Make prediction
            prediction = model.predict(features_scaled)[0]

            # Get individual model predictions if ensemble
            individual_predictions = {}
            if use_ensemble and hasattr(model, 'get_model_predictions'):
                individual_predictions = model.get_model_predictions(features_scaled)
                individual_predictions = {k: float(v[0]) for k, v in individual_predictions.items()}

            return jsonify({
                "predicted_profit": float(prediction),
                "model_type": model_data.get('model_type', 'ensemble' if use_ensemble else 'unknown'),
                "metrics": model_data.get('metrics', {}),
                "individual_predictions": individual_predictions,
                "confidence": "high" if model_data.get('metrics', {}).get('r2_score', 0) > 0.8 else "medium"
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/inventory/recommendations', methods=['GET'])
    def get_inventory_recommendations():
        """
        Get ML-powered inventory recommendations
        """
        try:
            df = fetch_orders_data()

            if len(df) < 5:
                return jsonify({"error": "Insufficient data"}), 400

            # Load the best model if available
            model_path = 'models/profit_prediction_model.pkl'
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                has_model = True
            else:
                has_model = False

            # Aggregate by product
            product_stats = df.groupby(['product_name', 'product_category']).agg({
                'orderQty': ['mean', 'std', 'sum', 'count'],
                'sales': 'sum',
                'profit': ['sum', 'mean'],
                'unitCost': 'mean',
                'price': 'mean'
            }).reset_index()

            product_stats.columns = ['_'.join(col).strip('_') for col in product_stats.columns.values]

            recommendations = []
            for _, row in product_stats.iterrows():
                avg_demand = float(row['orderQty_mean'])
                std_demand = float(row['orderQty_std']) if pd.notna(row['orderQty_std']) else 0
                total_orders = int(row['orderQty_count'])
                total_profit = float(row['profit_sum'])
                avg_profit = float(row['profit_mean'])

                # Calculate reorder point (using safety stock formula)
                safety_stock = std_demand * 1.65  # 95% service level
                reorder_point = avg_demand + safety_stock

                # Calculate optimal order quantity (simple EOQ approximation)
                if total_orders > 0:
                    demand_rate = float(row['orderQty_sum']) / total_orders
                    optimal_qty = max(int(demand_rate * 1.5), 1)
                else:
                    optimal_qty = 1

                # Determine status
                profit_margin = (avg_profit / float(row['sales_sum'])) * 100 if row['sales_sum'] > 0 else 0

                if avg_demand > product_stats['orderQty_mean'].median():
                    status = "High Demand"
                    action = f"Stock {int(optimal_qty * 1.5)} units - Popular item"
                elif avg_demand < product_stats['orderQty_mean'].quantile(0.25):
                    status = "Low Demand"
                    action = f"Reduce stock to {int(optimal_qty * 0.7)} units"
                else:
                    status = "Normal"
                    action = f"Maintain {int(optimal_qty)} units"

                # Add ML prediction if available
                ml_recommendation = None
                if has_model:
                    try:
                        # Predict future profit
                        features = pd.DataFrame([{
                            'unitCost': float(row['unitCost_mean']),
                            'price': float(row['price_mean']),
                            'orderQty': optimal_qty,
                            'year': datetime.now().year,
                            'month': datetime.now().month,
                            'day_of_week': datetime.now().weekday(),
                            'quarter': (datetime.now().month - 1) // 3 + 1,
                            'product_encoded': 0,  # Placeholder
                            'category_encoded': 0,  # Placeholder
                            'price_per_unit': float(row['price_mean'])
                        }])

                        scaler = model_data['scaler']
                        model = model_data['model']
                        features_scaled = scaler.transform(features)
                        predicted_profit = float(model.predict(features_scaled)[0])

                        ml_recommendation = {
                            'predicted_profit': predicted_profit,
                            'recommended_qty': optimal_qty,
                            'confidence': 'high' if model_data['metrics']['r2_score'] > 0.8 else 'medium'
                        }
                    except:
                        pass

                rec = {
                    "product_name": row['product_name'],
                    "category": row['product_category'] if pd.notna(row['product_category']) else 'Uncategorized',
                    "avg_demand": avg_demand,
                    "std_demand": std_demand,
                    "safety_stock": safety_stock,
                    "reorder_point": reorder_point,
                    "optimal_order_qty": optimal_qty,
                    "total_orders": total_orders,
                    "total_sales": float(row['sales_sum']),
                    "total_profit": total_profit,
                    "avg_profit": avg_profit,
                    "profit_margin": profit_margin,
                    "status": status,
                    "action": action,
                    "ml_recommendation": ml_recommendation
                }
                recommendations.append(rec)

            # Sort by total profit
            recommendations.sort(key=lambda x: x['total_profit'], reverse=True)

            return jsonify({
                "recommendations": recommendations[:20],
                "summary": {
                    "total_products": len(recommendations),
                    "high_demand": sum(1 for r in recommendations if r['status'] == "High Demand"),
                    "low_demand": sum(1 for r in recommendations if r['status'] == "Low Demand"),
                    "ml_enabled": has_model
                }
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/models/transfer-learning', methods=['POST'])
    def train_transfer_learning():
        """
        Train a model using transfer learning from a base model
        """
        def generate_logs():
            try:
                if not TF_AVAILABLE:
                    yield send_log("❌ TensorFlow not available. Install with: pip install tensorflow")
                    return

                data = request.get_json() or {}
                use_best_model = data.get('use_best_model', True)
                base_model_name = data.get('base_model', 'profit_prediction_model.pkl')
                epochs = data.get('epochs', 100)
                freeze_layers = data.get('freeze_layers', 2)

                yield send_log("🔬 Starting Transfer Learning Pipeline...")
                time.sleep(0.5)

                # If use_best_model is True, find the best model
                if use_best_model:
                    yield send_log("🔍 Finding best available model...")

                    # Import get_best_model from app
                    import sys
                    import os
                    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

                    # Manually find best model
                    models_to_check = [
                        ('models/profit_prediction_model.pkl', 'sklearn'),
                        ('models/ensemble_model.pkl', 'sklearn'),
                        ('models/transfer_learning_model_metadata.pkl', 'keras'),
                        ('models/mlp_model_metadata.pkl', 'keras'),
                        ('models/lstm_model_metadata.pkl', 'keras'),
                        ('models/gru_model_metadata.pkl', 'keras'),
                    ]

                    best_r2 = -float('inf')
                    best_found = None

                    for model_path, _ in models_to_check:
                        if os.path.exists(model_path):
                            try:
                                import joblib
                                model_data = joblib.load(model_path)
                                metrics = model_data.get('metrics', {})
                                r2_score = metrics.get('r2_score', -float('inf'))

                                if r2_score > best_r2:
                                    best_r2 = r2_score
                                    best_found = model_path
                            except:
                                pass

                    if best_found:
                        base_model_name = os.path.basename(best_found)
                        yield send_log(f"✓ Best model found: {base_model_name} (R² = {best_r2:.4f})")
                    else:
                        yield send_log("⚠️  No trained models found, using default")

                    time.sleep(0.5)

                yield send_log(f"Configuration: Base Model={base_model_name}, Epochs={epochs}, Freeze Layers={freeze_layers}")
                time.sleep(0.5)

                yield send_log("📥 Fetching data...")
                df = fetch_orders_data()

                if len(df) < 50:
                    yield send_log("❌ Insufficient data for deep learning (need at least 50 orders)")
                    return

                yield send_log(f"✓ Loaded {len(df)} orders")
                time.sleep(0.3)

                yield send_log("🔧 Preprocessing data...")
                df = preprocess_data(df)

                # Feature selection
                feature_columns = ['unitCost', 'price', 'orderQty', 'year', 'month',
                                 'day_of_week', 'quarter', 'product_encoded',
                                 'category_encoded', 'price_per_unit']

                X = df[feature_columns].fillna(df[feature_columns].mean())
                y = df['profit'].values

                # Scale data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )

                yield send_log(f"✓ Data split: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
                time.sleep(0.3)

                # Check for base model
                base_model_path = f'models/{base_model_name.replace(".pkl", ".h5")}'

                # If sklearn model exists, create a base neural network first
                sklearn_model_path = f'models/{base_model_name}'
                if not os.path.exists(base_model_path) and os.path.exists(sklearn_model_path):
                    yield send_log("📦 Converting sklearn model to neural network base...")

                    # Create a simple base model
                    base_model = build_mlp_model(
                        input_shape=(X_train.shape[1],),
                        hidden_layers=[128, 64, 32],
                        dropout=0.2
                    )

                    # Train base model briefly
                    base_model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50,
                        batch_size=32,
                        verbose=0,
                        callbacks=get_callbacks('checkpoints/base_model.h5')
                    )

                    # Save base model
                    os.makedirs('models', exist_ok=True)
                    base_model.save(base_model_path)
                    yield send_log(f"✓ Base model created and saved to {base_model_path}")
                    time.sleep(0.5)

                # Build transfer learning model
                yield send_log("🧬 Building transfer learning model...")

                if os.path.exists(base_model_path):
                    yield send_log(f"📂 Loading base model from {base_model_path}")

                    # Import keras to load model
                    import tensorflow as tf
                    from tensorflow.keras.losses import MeanSquaredError
                    from tensorflow.keras.metrics import MeanAbsoluteError

                    # Custom objects to handle metric deserialization
                    custom_objects = {
                        'mse': MeanSquaredError(),
                        'mae': MeanAbsoluteError(),
                        'MeanSquaredError': MeanSquaredError,
                        'MeanAbsoluteError': MeanAbsoluteError
                    }

                    base_model = tf.keras.models.load_model(base_model_path, custom_objects=custom_objects, compile=False)

                    # Create new model with frozen layers
                    new_model = tf.keras.Sequential()

                    # Add base model layers (freeze first N layers)
                    for i, layer in enumerate(base_model.layers[:-1]):  # Exclude output layer
                        if i < freeze_layers:
                            layer.trainable = False
                            yield send_log(f"  🔒 Frozen layer {i}: {layer.name}")
                        else:
                            yield send_log(f"  🔓 Trainable layer {i}: {layer.name}")
                        new_model.add(layer)

                    # Add new output layer
                    new_model.add(tf.keras.layers.Dense(1, name='transfer_output'))

                    # Compile with lower learning rate for transfer learning
                    new_model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                        loss='mse',
                        metrics=['mae']
                    )

                    model = new_model
                else:
                    yield send_log("⚠️  No base model found, training from scratch...")
                    model = build_mlp_model(
                        input_shape=(X_train.shape[1],),
                        hidden_layers=[128, 64, 32]
                    )

                time.sleep(0.3)

                # Build the model by calling it on sample input to initialize weights
                sample_input = tf.zeros((1, X_train.shape[1]))
                _ = model(sample_input)

                yield send_log("=" * 50)
                yield send_log("🔄 Training transfer learning model...")
                yield send_log(f"Total parameters: {model.count_params():,}")
                trainable_params = sum([np.prod(var.shape) for var in model.trainable_weights])
                yield send_log(f"Trainable parameters: {trainable_params:,}")
                time.sleep(0.5)

                # Create custom callback for progress logging
                class LoggingCallback(tf.keras.callbacks.Callback):
                    def __init__(self, log_func, total_epochs):
                        super().__init__()
                        self.log_func = log_func
                        self.total_epochs = total_epochs

                    def on_epoch_end(self, epoch, logs=None):
                        if epoch % 10 == 0 or epoch == self.total_epochs - 1:
                            progress = ((epoch + 1) / self.total_epochs) * 100
                            train_loss = logs.get('loss', 0)
                            val_loss = logs.get('val_loss', 0)
                            mae = logs.get('val_mae', 0)

                            # Calculate R² score
                            val_pred = self.model.predict(X_val, verbose=0)
                            from sklearn.metrics import r2_score
                            r2 = r2_score(y_val, val_pred)

                            self.log_func(f"Epoch {epoch+1}/{self.total_epochs} | Progress: {progress:.1f}% | Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {mae:.2f} | Val R²: {r2:.4f}")

                # Prepare callbacks
                logging_cb = LoggingCallback(lambda msg: None, epochs)  # We'll handle logging separately
                checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                    'checkpoints/transfer_learning_checkpoint.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=0
                )
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True,
                    verbose=0
                )

                # Train model with epoch-by-epoch logging
                for epoch in range(epochs):
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=1,
                        batch_size=32,
                        verbose=0
                    )

                    # Log progress every 10 epochs
                    if epoch % 10 == 0 or epoch == epochs - 1:
                        progress = ((epoch + 1) / epochs) * 100
                        train_loss = history.history['loss'][0]
                        val_loss = history.history['val_loss'][0]

                        # Get MAE from history - check both possible metric names
                        mae_key = 'val_mean_absolute_error' if 'val_mean_absolute_error' in history.history else 'val_mae'
                        mae = history.history.get(mae_key, [0])[0] if mae_key in history.history else 0

                        # Calculate R² score
                        val_pred = model.predict(X_val, verbose=0)
                        from sklearn.metrics import r2_score
                        r2 = r2_score(y_val, val_pred)

                        yield send_log(f"Epoch {epoch+1}/{epochs} | Progress: {progress:.1f}% | Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {mae:.2f} | Val R²: {r2:.4f}")

                        # Save checkpoint every 50 epochs
                        if (epoch + 1) % 50 == 0:
                            os.makedirs('checkpoints', exist_ok=True)
                            model.save('checkpoints/transfer_learning_checkpoint.h5')
                            yield send_log(f"💾 Checkpoint saved at epoch {epoch+1}")

                    time.sleep(0.01)

                yield send_log("=" * 50)
                yield send_log("📈 Evaluating transfer learning model...")
                time.sleep(0.3)

                # Final evaluation
                test_pred = model.predict(X_test, verbose=0)
                train_pred = model.predict(X_train, verbose=0)

                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

                test_r2 = r2_score(y_test, test_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                test_mae = mean_absolute_error(y_test, test_pred)

                train_r2 = r2_score(y_train, train_pred)

                yield send_log(f"✓ Test R² Score: {test_r2:.4f}")
                yield send_log(f"✓ Test RMSE: {test_rmse:.2f}")
                yield send_log(f"✓ Test MAE: {test_mae:.2f}")
                yield send_log(f"✓ Train R² Score: {train_r2:.4f}")
                time.sleep(0.3)

                # Save model
                yield send_log("💾 Saving transfer learning model...")
                os.makedirs('models', exist_ok=True)

                # Save Keras model
                model.save('models/transfer_learning_model.h5')

                # Save metadata
                model_data = {
                    'scaler': scaler,
                    'feature_columns': feature_columns,
                    'metrics': {
                        'r2_score': float(test_r2),
                        'rmse': float(test_rmse),
                        'mae': float(test_mae),
                        'train_r2': float(train_r2)
                    },
                    'model_type': 'transfer_learning',
                    'base_model': base_model_name,
                    'frozen_layers': freeze_layers,
                    'trained_at': datetime.now().isoformat()
                }
                joblib.dump(model_data, 'models/transfer_learning_model_metadata.pkl')

                yield send_log("✓ Model saved: models/transfer_learning_model.h5")
                yield send_log("✓ Metadata saved: models/transfer_learning_model_metadata.pkl")
                time.sleep(0.3)

                yield send_log("=" * 50)
                yield send_log("📊 Transfer Learning Summary:")
                yield send_log(f"Base Model: {base_model_name}")
                yield send_log(f"Frozen Layers: {freeze_layers}")
                yield send_log(f"Total Parameters: {model.count_params():,}")
                yield send_log(f"Trainable Parameters: {trainable_params:,}")
                yield send_log(f"Test R² Score: {test_r2:.4f} ({test_r2*100:.2f}% accuracy)")
                yield send_log(f"Test RMSE: ${test_rmse:.2f}")
                yield send_log(f"Test MAE: ${test_mae:.2f}")
                yield send_log("=" * 50)

                yield send_log("✅ Transfer learning completed successfully!")

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

    @app.route('/models/deep-learning', methods=['POST'])
    def train_deep_learning():
        """
        Train deep learning models (LSTM, GRU, MLP)
        """
        def generate_logs():
            try:
                if not TF_AVAILABLE:
                    yield send_log("❌ TensorFlow not available. Install with: pip install tensorflow")
                    return

                data = request.get_json() or {}
                model_type = data.get('model_type', 'mlp')
                epochs = data.get('epochs', 100)

                yield send_log(f"🧠 Starting Deep Learning Training ({model_type.upper()})...")
                time.sleep(0.5)

                yield send_log("📥 Fetching data...")
                df = fetch_orders_data()

                if len(df) < 50:
                    yield send_log("❌ Insufficient data for deep learning (need at least 50 orders)")
                    return

                yield send_log(f"✓ Loaded {len(df)} orders")
                df = preprocess_data(df)

                # Prepare features
                feature_columns = ['unitCost', 'price', 'orderQty', 'year', 'month',
                                 'day_of_week', 'quarter', 'product_encoded',
                                 'category_encoded', 'price_per_unit']

                X = df[feature_columns].fillna(df[feature_columns].mean()).values
                y = df['profit'].values

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )

                yield send_log(f"✓ Data prepared: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")

                # Build model
                yield send_log(f"🏗️  Building {model_type.upper()} model...")

                if model_type == 'lstm':
                    # Reshape for LSTM (samples, timesteps, features)
                    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                    X_val_reshaped = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
                    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
                    model = build_lstm_model(input_shape=(1, X_train.shape[1]))
                elif model_type == 'gru':
                    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                    X_val_reshaped = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
                    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
                    model = build_gru_model(input_shape=(1, X_train.shape[1]))
                else:  # mlp
                    X_train_reshaped = X_train
                    X_val_reshaped = X_val
                    X_test_reshaped = X_test
                    model = build_mlp_model(input_shape=(X_train.shape[1],))

                yield send_log(f"✓ Model created with {model.count_params():,} parameters")
                time.sleep(0.3)

                # Train
                yield send_log("=" * 50)
                yield send_log("🔄 Training model...")

                import tensorflow as tf
                for epoch in range(epochs):
                    history = model.fit(
                        X_train_reshaped, y_train,
                        validation_data=(X_val_reshaped, y_val),
                        epochs=1,
                        batch_size=32,
                        verbose=0
                    )

                    if epoch % 10 == 0 or epoch == epochs - 1:
                        progress = ((epoch + 1) / epochs) * 100
                        train_loss = history.history['loss'][0]
                        val_loss = history.history['val_loss'][0]

                        # Get MAE from history - check both possible metric names
                        mae_key = 'val_mean_absolute_error' if 'val_mean_absolute_error' in history.history else 'val_mae'
                        mae = history.history.get(mae_key, [0])[0] if mae_key in history.history else 0

                        val_pred = model.predict(X_val_reshaped, verbose=0)
                        from sklearn.metrics import r2_score
                        r2 = r2_score(y_val, val_pred)

                        yield send_log(f"Epoch {epoch+1}/{epochs} | Progress: {progress:.1f}% | Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {mae:.2f} | Val R²: {r2:.4f}")

                    time.sleep(0.01)

                # Evaluate
                test_pred = model.predict(X_test_reshaped, verbose=0)
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

                test_r2 = r2_score(y_test, test_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                test_mae = mean_absolute_error(y_test, test_pred)

                yield send_log("=" * 50)
                yield send_log(f"✓ Test R² Score: {test_r2:.4f}")
                yield send_log(f"✓ Test RMSE: {test_rmse:.2f}")
                yield send_log(f"✓ Test MAE: {test_mae:.2f}")

                # Save
                model.save(f'models/{model_type}_model.h5')
                metadata = {
                    'scaler': scaler,
                    'feature_columns': feature_columns,
                    'metrics': {'r2_score': float(test_r2), 'rmse': float(test_rmse), 'mae': float(test_mae)},
                    'model_type': model_type,
                    'trained_at': datetime.now().isoformat()
                }
                joblib.dump(metadata, f'models/{model_type}_model_metadata.pkl')

                yield send_log(f"✅ {model_type.upper()} model trained and saved!")

            except Exception as e:
                yield send_log(f"❌ Error: {str(e)}")
                import traceback
                yield send_log(f"Traceback: {traceback.format_exc()}")

        return Response(
            stream_with_context(generate_logs()),
            mimetype='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
        )

    return app
