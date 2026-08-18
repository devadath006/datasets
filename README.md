# Price Benchmarking System for Government Procurement

A web-based application for price benchmarking of products and services in government procurement, powered by AI/ML models.

## Features

- **AI-Powered Price Prediction**: Uses machine learning models trained on historical procurement data
- **Product Price Benchmarking**: Predict prices for networking equipment, computing hardware, storage systems, and more
- **Service Price Benchmarking**: Predict prices for IT services, facility management, and professional services
- **Market Trends Analysis**: View category-wise, location-wise, and vendor-wise price trends
- **Query History**: Track and review past price predictions
- **Price Reasonability Check**: Determine if quoted prices are reasonable based on market data

## Dataset

The system uses a synthetic dataset generated based on government procurement patterns:

- **5,000 product entries** across 10 categories
- **2,000 service entries** across 3 categories
- **15 locations** across India
- **9 pricing sources** including GeM Portal, CPP Portal, etc.

### Product Categories
- Networking Equipment
- Computing Hardware
- Storage Systems
- Display Systems
- Communication Systems
- Power Systems
- Security Systems
- Office Furniture
- Electrical Equipment
- Audio/Video Equipment

### Service Categories
- IT Services
- Facility Management
- Professional Services

## AI Model

The enhanced model achieves:
- **Product Price Prediction**: 95.4% accuracy (R² score)
- **Service Price Prediction**: 87.7% accuracy (R² score)

### Features Used
- Category and item specifications
- Vendor tier (Tier 1/2/3)
- Location (metro vs non-metro)
- Source reliability
- Quantity and bulk discounts
- Warranty period
- Delivery timeline
- Seasonal demand factors

## Installation

```bash
# Install dependencies
pip install flask pandas numpy scikit-learn

# Generate dataset
cd data
python generate_dataset.py

# Train model (optional, pre-trained model included)
cd ../src
python enhanced_model.py

# Run application
cd ..
python app.py
```

## Usage

1. **Login**: Use credentials `admin/admin123` or `user1/user123`
2. **Predict Price**: Select product/service category, item, location, and other parameters
3. **View Trends**: Analyze market trends across categories and locations
4. **Check History**: Review past predictions

## API Endpoint

```bash
# POST /api/predict
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "product",
    "category": "Networking Equipment",
    "item_name": "Enterprise Router",
    "vendor_tier": "Tier 1",
    "location": "Delhi",
    "source": "GeM Portal",
    "quantity": 10,
    "warranty_years": 3,
    "delivery_days": 14
  }'
```

## Project Structure

```
price-benchmarking/
├── app.py                 # Flask web application
├── requirements.txt       # Python dependencies
├── data/
│   ├── generate_dataset.py
│   ├── products_prices.csv
│   └── services_prices.csv
├── src/
│   ├── enhanced_model.py  # AI model with feature engineering
│   └── price_model.py     # Base AI model
├── templates/
│   ├── login.html
│   ├── dashboard.html
│   ├── predict.html
│   ├── result.html
│   ├── trends.html
│   └── history.html
└── enhanced_price_model.pkl  # Pre-trained model
```

## License

This project is for educational and government procurement use.
