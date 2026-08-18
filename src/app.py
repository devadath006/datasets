"""
Price Benchmarking Web Application
Flask-based web interface for government procurement price benchmarking.
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import json
import os
from datetime import datetime
from price_model import PriceBenchmarkingModel, analyze_market_trends

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize model
model = PriceBenchmarkingModel()

# Load datasets
PRODUCTS_DF = None
SERVICES_DF = None

def load_data():
    global PRODUCTS_DF, SERVICES_DF
    try:
        PRODUCTS_DF = pd.read_csv('products_prices.csv')
        SERVICES_DF = pd.read_csv('services_prices.csv')
        return True
    except FileNotFoundError:
        return False

def load_model():
    try:
        model.load_model('price_benchmarking_model.pkl')
        return True
    except FileNotFoundError:
        return False

# Sample user database (in production, use a proper database)
users_db = {
    'admin': {'password': 'admin123', 'role': 'admin'},
    'user1': {'password': 'user123', 'role': 'user'}
}

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    
    if username in users_db and users_db[username]['password'] == password:
        session['username'] = username
        session['role'] = users_db[username]['role']
        return redirect(url_for('dashboard'))
    
    return render_template('login.html', error='Invalid credentials')

@app.route('/register', methods=['POST'])
def register():
    username = request.form.get('username')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')
    
    if username in users_db:
        return render_template('login.html', error='Username already exists')
    
    if password != confirm_password:
        return render_template('login.html', error='Passwords do not match')
    
    users_db[username] = {'password': password, 'role': 'user'}
    session['username'] = username
    session['role'] = 'user'
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('index'))
    
    # Get summary statistics
    stats = {
        'total_products': len(PRODUCTS_DF) if PRODUCTS_DF is not None else 0,
        'total_services': len(SERVICES_DF) if SERVICES_DF is not None else 0,
        'categories': PRODUCTS_DF['category'].nunique() if PRODUCTS_DF is not None else 0,
        'locations': PRODUCTS_DF['location'].nunique() if PRODUCTS_DF is not None else 0
    }
    
    return render_template('dashboard.html', 
                         username=session['username'],
                         stats=stats)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        query_type = request.form.get('query_type')
        
        if query_type == 'product':
            result = predict_product()
        else:
            result = predict_service()
        
        return render_template('result.html', result=result, query_type=query_type)
    
    return render_template('predict.html')

def predict_product():
    """Predict product price based on inputs."""
    category = request.form.get('category')
    item_name = request.form.get('item_name')
    vendor_tier = request.form.get('vendor_tier', 'Tier 2')
    location = request.form.get('location')
    source = request.form.get('source')
    quantity = int(request.form.get('quantity', 1))
    warranty_years = int(request.form.get('warranty_years', 1))
    delivery_days = int(request.form.get('delivery_days', 14))
    
    # Get prediction
    prediction = model.predict_product_price(
        category=category,
        item_name=item_name,
        vendor_tier=vendor_tier,
        location=location,
        source=source,
        quantity=quantity,
        warranty_years=warranty_years,
        delivery_days=delivery_days
    )
    
    # Get historical prices for comparison
    historical_prices = []
    if PRODUCTS_DF is not None:
        similar_products = PRODUCTS_DF[
            (PRODUCTS_DF['category'] == category) &
            (PRODUCTS_DF['item_name'] == item_name)
        ]
        if len(similar_products) > 0:
            historical_prices = similar_products['unit_price'].tolist()
    
    # Get recommendation
    recommendation = model.get_price_recommendation(
        prediction, 
        budget=None
    )
    
    return {
        'type': 'product',
        'prediction': prediction,
        'recommendation': recommendation,
        'historical_prices': historical_prices[:10],  # Last 10 prices
        'input': {
            'category': category,
            'item_name': item_name,
            'vendor_tier': vendor_tier,
            'location': location,
            'source': source,
            'quantity': quantity,
            'warranty_years': warranty_years,
            'delivery_days': delivery_days
        }
    }

def predict_service():
    """Predict service price based on inputs."""
    category = request.form.get('category')
    service_name = request.form.get('service_name')
    location = request.form.get('location')
    source = request.form.get('source')
    duration_months = int(request.form.get('duration_months', 12))
    
    # Get prediction
    prediction = model.predict_service_price(
        category=category,
        service_name=service_name,
        location=location,
        source=source,
        duration_months=duration_months
    )
    
    # Get historical prices
    historical_prices = []
    if SERVICES_DF is not None:
        similar_services = SERVICES_DF[
            (SERVICES_DF['category'] == category) &
            (SERVICES_DF['service_name'] == service_name)
        ]
        if len(similar_services) > 0:
            historical_prices = similar_services['monthly_price'].tolist()
    
    # Get recommendation
    recommendation = model.get_price_recommendation(
        prediction,
        budget=None
    )
    
    return {
        'type': 'service',
        'prediction': prediction,
        'recommendation': recommendation,
        'historical_prices': historical_prices[:10],
        'input': {
            'category': category,
            'service_name': service_name,
            'location': location,
            'source': source,
            'duration_months': duration_months
        }
    }

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for price prediction."""
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json()
    query_type = data.get('query_type')
    
    try:
        if query_type == 'product':
            prediction = model.predict_product_price(
                category=data['category'],
                item_name=data['item_name'],
                vendor_tier=data.get('vendor_tier', 'Tier 2'),
                location=data['location'],
                source=data['source'],
                quantity=data.get('quantity', 1),
                warranty_years=data.get('warranty_years', 1),
                delivery_days=data.get('delivery_days', 14)
            )
        else:
            prediction = model.predict_service_price(
                category=data['category'],
                service_name=data['service_name'],
                location=data['location'],
                source=data['source'],
                duration_months=data.get('duration_months', 12)
            )
        
        return jsonify({'success': True, 'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/trends')
def trends():
    """Display market trends and analytics."""
    if 'username' not in session:
        return redirect(url_for('index'))
    
    if PRODUCTS_DF is not None:
        trends_data = analyze_market_trends(PRODUCTS_DF)
    else:
        trends_data = {}
    
    return render_template('trends.html', trends=trends_data)

@app.route('/history')
def history():
    """Display procurement history."""
    if 'username' not in session:
        return redirect(url_for('index'))
    
    # Get recent queries (in production, store in database)
    history_data = session.get('query_history', [])
    
    return render_template('history.html', history=history_data)

@app.context_processor
def utility_processor():
    """Add utility functions to templates."""
    def format_currency(amount):
        return f"₹{amount:,.2f}"
    
    return dict(format_currency=format_currency)


if __name__ == '__main__':
    # Load data and model
    if not load_data():
        print("Warning: Could not load datasets. Please run generate_dataset.py first.")
    
    if not load_model():
        print("Warning: Could not load pre-trained model. Training new model...")
        if PRODUCTS_DF is not None and SERVICES_DF is not None:
            model.train(PRODUCTS_DF, SERVICES_DF)
            model.save_model('price_benchmarking_model.pkl')
    
    print("Starting Price Benchmarking Application...")
    print("Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
