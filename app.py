"""
Price Benchmarking Web Application
Flask-based web interface for government procurement price benchmarking.
Uses Enhanced AI Model for accurate price predictions. test
"""

import sys
sys.path.insert(0, 'src')

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import json
import os
from datetime import datetime
from enhanced_model import EnhancedPriceModel

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize model
model = EnhancedPriceModel()

# Load datasets
PRODUCTS_DF = None
SERVICES_DF = None

# Price lookup for common items
PRODUCT_PRICES = {}
SERVICE_PRICES = {}

def load_data():
    global PRODUCTS_DF, SERVICES_DF, PRODUCT_PRICES, SERVICE_PRICES
    try:
        PRODUCTS_DF = pd.read_csv('products_prices.csv')
        SERVICES_DF = pd.read_csv('services_prices.csv')
        
        # Build lookup tables
        for _, row in PRODUCTS_DF.iterrows():
            key = (row['category'], row['item_name'])
            if key not in PRODUCT_PRICES:
                PRODUCT_PRICES[key] = {
                    'base_price': row['base_price'],
                    'vendors': set(),
                    'locations': set()
                }
            PRODUCT_PRICES[key]['vendors'].add(row['vendor'])
            PRODUCT_PRICES[key]['locations'].add(row['location'])
        
        for _, row in SERVICES_DF.iterrows():
            key = (row['category'], row['service_name'])
            if key not in SERVICE_PRICES:
                SERVICE_PRICES[key] = {
                    'base_price': row['base_price'],
                    'locations': set()
                }
            SERVICE_PRICES[key]['locations'].add(row['location'])
        
        return True
    except FileNotFoundError:
        return False

def load_model():
    try:
        model.load_model('enhanced_price_model.pkl')
        return True
    except FileNotFoundError:
        return False

# Sample user database
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
        session['query_history'] = []
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
    session['query_history'] = []
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('index'))
    
    stats = {
        'total_products': len(PRODUCTS_DF) if PRODUCTS_DF is not None else 0,
        'total_services': len(SERVICES_DF) if SERVICES_DF is not None else 0,
        'categories': PRODUCTS_DF['category'].nunique() if PRODUCTS_DF is not None else 0,
        'locations': PRODUCTS_DF['location'].nunique() if PRODUCTS_DF is not None else 0,
        'vendors': PRODUCTS_DF['vendor'].nunique() if PRODUCTS_DF is not None else 0
    }
    
    return render_template('dashboard.html', 
                         username=session['username'],
                         stats=stats)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('index'))
    
    # Get available items for dropdowns
    product_items = {}
    if PRODUCTS_DF is not None:
        for cat in PRODUCTS_DF['category'].unique():
            items = PRODUCTS_DF[PRODUCTS_DF['category'] == cat]['item_name'].unique().tolist()
            product_items[cat] = items
    
    service_items = {}
    if SERVICES_DF is not None:
        for cat in SERVICES_DF['category'].unique():
            items = SERVICES_DF[SERVICES_DF['category'] == cat]['service_name'].unique().tolist()
            service_items[cat] = items
    
    if request.method == 'POST':
        query_type = request.form.get('query_type')
        
        if query_type == 'product':
            result = predict_product()
        else:
            result = predict_service()
        
        # Store in history
        if 'query_history' not in session:
            session['query_history'] = []
        
        history_entry = {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'type': query_type,
            'item_name': result['input'].get('item_name') or result['input'].get('service_name'),
            'location': result['input']['location'],
            'price': result['prediction'].get('predicted_price') or result['prediction'].get('predicted_monthly_price'),
            'confidence': result['prediction'].get('confidence', 0)
        }
        session['query_history'].append(history_entry)
        session.modified = True
        
        return render_template('result.html', result=result, query_type=query_type)
    
    return render_template('predict.html', 
                         product_items=product_items,
                         service_items=service_items)

def predict_product():
    category = request.form.get('category')
    item_name = request.form.get('item_name')
    vendor_tier = request.form.get('vendor_tier', 'Tier 2')
    location = request.form.get('location')
    source = request.form.get('source')
    quantity = int(request.form.get('quantity', 1))
    warranty_years = int(request.form.get('warranty_years', 1))
    delivery_days = int(request.form.get('delivery_days', 14))
    
    # Get base price
    key = (category, item_name)
    base_price = PRODUCT_PRICES.get(key, {}).get('base_price', 50000)
    
    prediction = model.predict_product(
        category=category,
        item_name=item_name,
        vendor_tier=vendor_tier,
        location=location,
        source=source,
        quantity=quantity,
        warranty_years=warranty_years,
        delivery_days=delivery_days,
        base_price=base_price
    )
    
    # Get historical prices
    historical_prices = []
    if PRODUCTS_DF is not None:
        similar = PRODUCTS_DF[
            (PRODUCTS_DF['category'] == category) &
            (PRODUCTS_DF['item_name'] == item_name)
        ]['unit_price'].tolist()
        historical_prices = similar[:10]
    
    recommendation = model.get_recommendation(prediction)
    
    return {
        'type': 'product',
        'prediction': prediction,
        'recommendation': recommendation,
        'historical_prices': historical_prices,
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
    category = request.form.get('category')
    service_name = request.form.get('service_name')
    location = request.form.get('location')
    source = request.form.get('source')
    duration_months = int(request.form.get('duration_months', 12))
    
    # Get base price
    key = (category, service_name)
    base_price = SERVICE_PRICES.get(key, {}).get('base_price', 50000)
    
    prediction = model.predict_service(
        category=category,
        service_name=service_name,
        location=location,
        source=source,
        duration_months=duration_months,
        base_price=base_price
    )
    
    # Get historical prices
    historical_prices = []
    if SERVICES_DF is not None:
        similar = SERVICES_DF[
            (SERVICES_DF['category'] == category) &
            (SERVICES_DF['service_name'] == service_name)
        ]['monthly_price'].tolist()
        historical_prices = similar[:10]
    
    recommendation = model.get_recommendation(prediction)
    
    return {
        'type': 'service',
        'prediction': prediction,
        'recommendation': recommendation,
        'historical_prices': historical_prices,
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
            key = (data['category'], data['item_name'])
            base_price = PRODUCT_PRICES.get(key, {}).get('base_price', 50000)
            
            prediction = model.predict_product(
                category=data['category'],
                item_name=data['item_name'],
                vendor_tier=data.get('vendor_tier', 'Tier 2'),
                location=data['location'],
                source=data['source'],
                quantity=data.get('quantity', 1),
                warranty_years=data.get('warranty_years', 1),
                delivery_days=data.get('delivery_days', 14),
                base_price=base_price
            )
        else:
            key = (data['category'], data['service_name'])
            base_price = SERVICE_PRICES.get(key, {}).get('base_price', 50000)
            
            prediction = model.predict_service(
                category=data['category'],
                service_name=data['service_name'],
                location=data['location'],
                source=data['source'],
                duration_months=data.get('duration_months', 12),
                base_price=base_price
            )
        
        recommendation = model.get_recommendation(prediction)
        
        return jsonify({
            'success': True, 
            'prediction': prediction,
            'recommendation': recommendation
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/trends')
def trends():
    """Display market trends."""
    if 'username' not in session:
        return redirect(url_for('index'))
    
    trends_data = {}
    if PRODUCTS_DF is not None:
        # Category trends
        trends_data['categories'] = {}
        for cat in PRODUCTS_DF['category'].unique():
            cat_data = PRODUCTS_DF[PRODUCTS_DF['category'] == cat]
            trends_data['categories'][cat] = {
                'avg_price': round(cat_data['unit_price'].mean(), 2),
                'count': len(cat_data),
                'top_vendors': cat_data['vendor'].value_counts().head(3).to_dict()
            }
        
        # Location trends
        trends_data['locations'] = {}
        for loc in PRODUCTS_DF['location'].unique():
            loc_data = PRODUCTS_DF[PRODUCTS_DF['location'] == loc]
            trends_data['locations'][loc] = {
                'avg_price': round(loc_data['unit_price'].mean(), 2),
                'count': len(loc_data)
            }
        
        # Vendor trends
        trends_data['vendors'] = {}
        for vendor in PRODUCTS_DF['vendor'].unique():
            v_data = PRODUCTS_DF[PRODUCTS_DF['vendor'] == vendor]
            trends_data['vendors'][vendor] = {
                'avg_price': round(v_data['unit_price'].mean(), 2),
                'count': len(v_data),
                'avg_quality': round(v_data['quality_rating'].mean(), 2)
            }
    
    return render_template('trends.html', trends=trends_data)

@app.route('/history')
def history():
    if 'username' not in session:
        return redirect(url_for('index'))
    
    history_data = session.get('query_history', [])
    return render_template('history.html', history=history_data)

@app.context_processor
def utility_processor():
    def format_currency(amount):
        return f"{amount:,.2f}"
    return dict(format_currency=format_currency)


if __name__ == '__main__':
    print("Initializing Price Benchmarking Application...")
    
    if not load_data():
        print("Warning: Could not load datasets. Please run generate_dataset.py first.")
    
    if not load_model():
        print("Training new model...")
        if PRODUCTS_DF is not None and SERVICES_DF is not None:
            model.train(PRODUCTS_DF, SERVICES_DF)
            model.save_model('enhanced_price_model.pkl')
    
    print("Application ready!")
    print("Access at: http://localhost:5000")
    print("Default credentials: admin/admin123 or user1/user123")
    app.run(debug=True, host='0.0.0.0', port=5000)
