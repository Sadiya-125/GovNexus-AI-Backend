# Flask Backend for Inventory Management ML Training

This is a comprehensive Flask backend that handles machine learning model training, optimization, and real-time predictions for an inventory management system. It includes advanced ML capabilities, deep learning support, transfer learning, ensemble methods, WhatsApp notifications, and enhanced risk analysis with external data integration.

## 🚀 Features

### Core Features

- **ML Model Training**: Train multiple ML models for profit prediction
- **Real-time Streaming**: Server-Sent Events (SSE) for live training logs
- **Database Integration**: PostgreSQL connection with automatic data fetching
- **Model Persistence**: Save/load trained models and checkpoints
- **WhatsApp Notifications**: Supplier notifications via Twilio
- **Inventory Optimization**: ML-powered recommendations and risk assessment

### Advanced ML Features

- **Multiple Model Types**: Random Forest, Gradient Boosting, Linear Regression, Ridge, Lasso, Decision Tree, SVM, KNN, MLP
- **Deep Learning**: LSTM, GRU, CNN-LSTM, Attention mechanisms
- **Transfer Learning**: Adapt pre-trained models for new tasks
- **Ensemble Models**: Combine multiple models for better predictions
- **Model Comparison**: Automatic comparison of different algorithms
- **Advanced Predictions**: Confidence intervals and uncertainty estimation

### Enhanced Risk Analysis

#### Features

- **Google Trends Integration**: Identifies seasonal demand patterns, tracks product interest by state/region, and provides historical search interest data
- **News Sentiment Analysis**: Fetches recent news articles about products, analyzes positive/negative sentiment, and detects market demand indicators
- **Market Pricing Intelligence**: Monitors market prices, tracks pricing consistency across sellers, and provides market availability data
- **Holiday Calendar**: Fetches public holidays and adjusts predictions for holiday seasons
- **Advanced Risk Classification**: Classifies products into 4 risk levels (Critical, High, Medium, Low)

#### Architecture

```
┌─────────────────────────────────────────────────────┐
│              /optimize Endpoint                      │
│         (with user location data)                    │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌──────────────────┐     ┌──────────────────┐
│ Basic Analysis   │     │ Enhanced Risk    │
│  (Fallback)      │     │    Analyzer      │
└──────────────────┘     └────────┬─────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
            ┌───────────────┐          ┌──────────────┐
            │ API           │          │ Sentiment    │
            │ Integrations  │          │ Analyzer     │
            └───────┬───────┘          └──────────────┘
                    │
        ┌───────────┼───────────┬──────────┐
        ▼           ▼           ▼          ▼
    ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
    │Trends│   │ News │   │Price │   │Holid.│
    └──────┘   └──────┘   └──────┘   └──────┘
```

#### Module Structure

- **api_integrations.py**: Handles external API communications

  - `get_trends_timeseries(product_name, region)` - Google Trends data
  - `get_trends_by_region(product_name, region)` - Regional interest
  - `get_state_interest(product_name, state_name)` - State-specific interest
  - `get_product_news(product_name, location)` - News articles
  - `get_product_pricing(product_name, location)` - Market pricing
  - `get_upcoming_holidays(country, months_ahead)` - Holiday calendar
  - `compute_eoq(annual_demand, ordering_cost, holding_cost)` - Economic Order Quantity

- **sentiment_analyzer.py**: Analyzes text sentiment from news articles

  - `analyze_text(text)` - Single text sentiment
  - `analyze_news_articles(news_articles)` - Bulk sentiment analysis
  - `get_sentiment_risk_modifier(sentiment_analysis)` - Risk score adjustment
  - `get_trend_from_sentiment(sentiment_analysis)` - Trend prediction

- **enhanced_risk_analyzer.py**: Comprehensive risk analysis engine
  - `analyze_product_risk(...)` - Main risk analysis function
  - `calculate_optimal_reorder_point(...)` - Seasonality-adjusted reorder point
  - `get_risk_explanation(analysis_result)` - Human-readable explanations

#### Supported Indian States

The system supports all 28 Indian states and 8 union territories:

- Andhra Pradesh, Arunachal Pradesh, Assam, Bihar, Chhattisgarh
- Goa, Gujarat, Haryana, Himachal Pradesh, Jharkhand
- Karnataka, Kerala, Madhya Pradesh, Maharashtra, Manipur
- Meghalaya, Mizoram, Nagaland, Odisha, Punjab
- Rajasthan, Sikkim, Tamil Nadu, Telangana, Tripura
- Uttar Pradesh, Uttarakhand, West Bengal

### Business Intelligence

- **Risk Assessment**: Enhanced risk categorization with external data signals
- **Demand Forecasting**: ML-powered demand predictions with seasonality
- **Optimization**: Reorder points, safety stock, optimal order quantities
- **Performance Metrics**: Comprehensive evaluation (R², RMSE, MAE, MAPE, etc.)

---

## 📦 Setup & Installation

### Prerequisites

- Python 3.8+
- PostgreSQL database
- Node.js (for frontend)

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**

- Flask==3.0.0
- flask-cors==4.0.0
- psycopg2-binary==2.9.9
- python-dotenv==1.0.0
- numpy==1.26.2
- pandas==2.1.4
- scikit-learn==1.3.2
- joblib==1.3.2
- scipy==1.11.4

**Optional (for deep learning):**

```bash
pip install tensorflow keras
```

**Optional (for notifications):**

```bash
pip install twilio
```

**Optional (for enhanced risk analysis):**

```bash
pip install textblob serpapi calendarific
python -m textblob.download_corpora
```

### 2. Configure Database Connection

1. Copy `.env.example` to `.env`
2. Update database credentials:

```env
DB_HOST=localhost
DB_NAME=inventory_db
DB_USER=your_username
DB_PASSWORD=your_password
DB_PORT=5432
```

### 3. Configure Twilio (Optional)

For WhatsApp notifications:

```env
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=+1234567890
```

### 4. Configure External APIs (Optional)

For enhanced risk analysis:

```env
SERPAPI_KEY=your_serpapi_key_here
CALENDARIFIC_API_KEY=your_calendarific_api_key_here
```

### 5. Email Service Setup (Optional)

For sending reports via email:

1. **Install Node.js dependencies** (in the Backend directory):

```bash
npm install nodemailer cors body-parser dotenv
```

2. **Configure email settings** in your `.env` file:

```env
# Email Service Configuration
EMAIL_SERVICE=gmail  # or 'outlook', 'yahoo', etc.
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password  # Use app password for Gmail
EMAIL_SERVICE_PORT=5001
```

3. **Run the email service**:

```bash
node email_service.js
```

The email service runs on `http://localhost:5001`

### 6. Run the Flask Server

```bash
python app.py
```

The server starts on `http://localhost:5000`

---

## 🔌 API Endpoints

### Core Endpoints

| Endpoint        | Method   | Description                       |
| --------------- | -------- | --------------------------------- |
| `/train`        | POST     | Train ML model with configuration |
| `/predict`      | POST     | Make profit predictions           |
| `/eda`          | GET      | Get exploratory data analysis     |
| `/optimize`     | GET/POST | ML-powered inventory optimization |
| `/health`       | GET      | Health check                      |
| `/orders/count` | GET      | Get total order count             |

### ML Advanced Endpoints

| Endpoint                     | Method | Description                                 |
| ---------------------------- | ------ | ------------------------------------------- |
| `/models/compare`            | POST   | Compare multiple models                     |
| `/models/ensemble`           | POST   | Train ensemble model                        |
| `/models/list`               | GET    | List all trained models                     |
| `/models/best`               | GET    | Get best performing model                   |
| `/models/predict/advanced`   | POST   | Advanced predictions with confidence        |
| `/inventory/recommendations` | GET    | ML-powered inventory recommendations        |
| `/models/transfer-learning`  | POST   | Train model with transfer learning          |
| `/models/deep-learning`      | POST   | Train deep learning models (LSTM, GRU, MLP) |

### Notification Endpoints

| Endpoint                | Method | Description                     |
| ----------------------- | ------ | ------------------------------- |
| `/supplier/notify`      | POST   | Send WhatsApp to supplier       |
| `/supplier/notify-bulk` | POST   | Send bulk category notification |

---

## 🎯 ML Models & Training

### Supported Model Types

#### Traditional ML Models

- **Random Forest**: Best for complex patterns, handles non-linear relationships
- **Gradient Boosting**: High accuracy, sequential model building
- **Linear Regression**: Fast, simple baseline model
- **Ridge/Lasso**: Regularized regression to prevent overfitting
- **Decision Tree**: Interpretable, good for feature importance
- **SVM**: Effective for smaller datasets with clear margins
- **K-Nearest Neighbors**: Instance-based learning
- **MLP**: Neural network for complex patterns

#### Deep Learning Models

- **LSTM**: Long Short-Term Memory for time series
- **GRU**: Gated Recurrent Units (simpler than LSTM)
- **CNN-LSTM**: Hybrid convolutional-recurrent networks
- **Attention**: Enhanced predictions with attention mechanisms
- **MLP**: Multi-Layer Perceptron neural network

### Training Process

1. **Data Fetching**: Load orders from PostgreSQL database
2. **Preprocessing**:
   - Convert decimals to floats
   - Extract time features (month, day_of_week, quarter)
   - Encode categorical variables (products, categories)
   - Calculate profit margins and interaction features
3. **Feature Engineering**:
   - Price per unit calculations
   - Temporal features (seasonal patterns)
   - Interaction terms
4. **Model Training**:
   - 80/20 train/test split
   - Feature scaling with StandardScaler
   - Cross-validation for model selection
5. **Evaluation**: Comprehensive metrics calculation
6. **Persistence**: Save models, scalers, and metadata

### Model Configuration

```javascript
// Training request
{
  "modelType": "random_forest",  // or "gradient_boosting", "linear_regression", etc.
  "epochs": 100,                 // training iterations
  "useCheckpoint": false         // resume from checkpoint
}
```

### Model Comparison

Compare multiple models automatically:

```javascript
POST /models/compare
{
  "models": ["random_forest", "gradient_boosting", "linear_regression"]
}
```

Returns performance metrics for each model and recommends the best one.

### Ensemble Training

```javascript
POST /models/ensemble
{
  "models": ["random_forest", "gradient_boosting", "ridge"]
}
```

### Transfer Learning

Adapt pre-trained models:

```javascript
POST /models/transfer-learning
{
  "use_best_model": true,
  "epochs": 100,
  "freeze_layers": 2
}
```

---

## 📊 Enhanced Risk Analysis

The enhanced `/optimize` endpoint integrates external data sources for comprehensive risk assessment:

### External Data Sources

#### Google Trends Integration

- **Seasonality Analysis**: Identifies seasonal demand patterns
- **Regional Interest**: Tracks product interest by state/region
- **Trend Timeseries**: Historical search interest data

#### News Sentiment Analysis

- **Real-time News**: Fetches recent news articles about products
- **Sentiment Scoring**: Analyzes positive/negative sentiment
- **Demand Signals**: Detects market demand indicators

#### Market Pricing Intelligence

- **Competitive Pricing**: Monitors market prices
- **Price Variance**: Tracks pricing consistency across sellers
- **Market Availability**: Number of active sellers

#### Holiday Calendar

- **Upcoming Events**: Fetches public holidays
- **Demand Forecasting**: Adjusts predictions for holiday seasons

### Risk Classification

Classifies products into 4 risk levels:

- **Critical** (Risk Score ≥ 7): Immediate action required
- **High** (Risk Score ≥ 5): High priority monitoring
- **Medium** (Risk Score ≥ 2): Regular monitoring
- **Low** (Risk Score < 2): Stable products

### Enhanced Optimization Response

```json
{
  "recommendations": [
    {
      "product_name": "Laptop",
      "category": "Electronics",
      "current_avg_demand": 50,
      "predicted_demand": 65,
      "demand_trend": "increasing",
      "reorder_point": 74.75,
      "optimal_order_qty": 75,
      "safety_stock": 24.75,
      "risk_level": "high",
      "risk_score": 8,
      "total_sales": 125000,
      "total_profit": 25000,
      "external_insights": {
        "seasonality_index": 1.45,
        "state_interest": 78,
        "sentiment": "positive",
        "demand_signal": "increasing",
        "news_count": 8,
        "upcoming_holidays": 3,
        "pricing_competitive": true
      },
      "risk_factors": [
        "Demand increasing by 30.0%",
        "High seasonality detected in search trends",
        "Very high interest in your region",
        "News sentiment indicates rising demand",
        "Multiple holidays approaching (3 events)"
      ]
    }
  ],
  "summary": {
    "total_products": 45,
    "critical_risk_products": 3,
    "high_risk_products": 12,
    "medium_risk_products": 20,
    "low_risk_products": 10,
    "total_recommended_stock": 2500,
    "ml_model_used": "ensemble_model",
    "ml_model_r2_score": 0.87,
    "enhanced_analysis_enabled": true,
    "user_location": "Mumbai",
    "user_state": "Maharashtra"
  }
}
```

---

## 📊 Optimization Features

The `/optimize` endpoint provides ML-powered inventory insights:

### Risk Assessment

- **High Risk**: Immediate action required (CV > 0.5 or increasing demand)
- **Medium Risk**: Monitor closely (CV 0.25-0.5 or decreasing demand)
- **Low Risk**: Maintain current levels

### Key Calculations

```python
# Safety Stock (95% service level)
safety_stock = std_demand * 1.65

# Reorder Point
reorder_point = avg_demand + safety_stock

# Optimal Order Quantity (EOQ approximation)
optimal_qty = demand_rate * 1.5
```

### Optimization Response

```json
{
  "recommendations": [
    {
      "product_name": "Product A",
      "category": "Electronics",
      "risk_level": "high",
      "current_avg_demand": 25.5,
      "predicted_demand": 32.1,
      "demand_trend": "increasing",
      "reorder_point": 45.2,
      "optimal_order_qty": 38,
      "safety_stock": 12.3,
      "action_required": "Increase stock by 50% - High demand trend"
    }
  ],
  "summary": {
    "total_products": 150,
    "high_risk_products": 12,
    "medium_risk_products": 28,
    "low_risk_products": 110
  }
}
```

---

## 🔧 Usage Examples

### Train a Model

```javascript
const response = await fetch("http://localhost:5000/train", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    modelType: "random_forest",
    epochs: 100,
    useCheckpoint: false,
  }),
});

// Stream training logs
const reader = response.body.getReader();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  console.log(new TextDecoder().decode(value));
}
```

### Make Predictions

```javascript
const response = await fetch("http://localhost:5000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    unitCost: 10.5,
    price: 25.0,
    orderQty: 100,
    year: 2024,
    month: 1,
    day_of_week: 1,
    quarter: 1,
    product_encoded: 5,
    category_encoded: 2,
    price_per_unit: 25.0,
  }),
});

const result = await response.json();
// result.predicted_profit
```

### Get Optimization

```javascript
const response = await fetch("http://localhost:5000/optimize");
const data = await response.json();
// data.recommendations contains ML insights
```

### Get Enhanced Optimization with Location

```javascript
const response = await fetch("http://localhost:5000/optimize", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    user_state: "Maharashtra",
    user_location: "Mumbai",
    use_external_data: true,
  }),
});

const data = await response.json();
// data.recommendations contains enhanced insights with external data
```

### Send Supplier Notification

```javascript
const response = await fetch("http://localhost:5000/supplier/notify", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    to_number: "+1234567890",
    product_name: "Wireless Headphones",
    category: "Electronics",
    predicted_demand: 150,
    optimal_qty: 200,
  }),
});
```

---

## 💾 Model Persistence

### Models Directory Structure

```
models/
├── profit_prediction_model.pkl          # Main trained model
├── ensemble_model.pkl                   # Ensemble model
├── comparison_results.pkl               # Model comparison data
├── product_encoder.pkl                  # Product label encoder
├── category_encoder.pkl                 # Category label encoder
├── scaler.pkl                           # Feature scaler
├── transfer_learning_model.h5           # Keras transfer learning model
├── transfer_learning_model_metadata.pkl # Transfer learning metadata
├── lstm_model.h5                        # LSTM model
├── lstm_model_metadata.pkl             # LSTM metadata
└── ...
```

### Checkpoints Directory

```
checkpoints/
├── model_checkpoint.pkl                 # Training checkpoints
├── random_forest_checkpoint.pkl         # Model-specific checkpoints
├── gradient_boosting_checkpoint.pkl
├── dl_checkpoint.h5                     # Deep learning checkpoints
└── ...
```

---

## 📈 Performance Metrics

All models are evaluated using:

- **R² Score**: Coefficient of determination (0-1, higher is better)
- **RMSE**: Root Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **MAPE**: Mean Absolute Percentage Error (lower is better)
- **Correlation**: Pearson correlation coefficient
- **RAE**: Relative Absolute Error
- **RRSE**: Root Relative Squared Error

### Interpreting Results

- **R² > 0.8**: Excellent model performance
- **R² 0.6-0.8**: Good performance
- **R² 0.4-0.6**: Acceptable performance
- **R² < 0.4**: Poor performance, consider different model

---

## 🎓 Business Impact

### Before ML Implementation

- Manual inventory decisions
- Reactive restocking based on gut feeling
- Frequent stockouts or overstocking
- No demand forecasting capabilities

### After ML Implementation

- **Automated Risk Assessment**: Data-driven inventory decisions
- **Predictive Demand Forecasting**: Reduce stockouts by 30%
- **Optimized Order Quantities**: Minimize holding costs
- **Real-time Insights**: Instant optimization recommendations
- **Supplier Integration**: Automated notifications and coordination

### Key Benefits

- **Reduced Stockouts**: Better demand prediction
- **Lower Holding Costs**: Optimized inventory levels
- **Improved Cash Flow**: Better inventory turnover
- **Data-Driven Decisions**: Confidence scores for all recommendations

---

## 🔮 Future Enhancements

Potential additions:

- [ ] **Demand Seasonality Detection**: Identify seasonal patterns
- [ ] **Multi-step Forecasting**: Predict multiple future periods
- [ ] **Anomaly Detection**: Identify unusual order patterns
- [ ] **Price Optimization**: ML-powered pricing recommendations
- [ ] **Supplier Lead Time Prediction**: Estimate delivery times
- [ ] **Customer Segmentation**: Group customers by behavior
- [ ] **Market Basket Analysis**: Product recommendation engine
- [ ] **Automated Reordering**: Trigger orders based on ML predictions

---

## 📝 Notes

- **Minimum Data Requirements**: 20+ orders for basic ML, 50+ for deep learning
- **Model Retraining**: Retrain weekly/monthly for best performance
- **Database Schema**: Ensure Order and Product tables exist with required fields
- **Real-time Updates**: Training logs update every few epochs
- **Model Selection**: Start with Random Forest for most use cases
- **Checkpointing**: Use checkpoints for long training sessions

---

**Built with ❤️ using Flask, TensorFlow, Scikit-learn, PostgreSQL, and Twilio**
