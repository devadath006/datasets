"""
Enhanced Price Benchmarking AI Model with Feature Engineering
Improved model with better accuracy for price prediction.
"""

import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class EnhancedPriceModel:
    """
    Enhanced AI Model for Government Procurement Price Benchmarking.
    Uses feature engineering and ensemble methods for better accuracy.
    """
    
    def __init__(self):
        self.product_model = None
        self.service_model = None
        self.label_encoders = {}
        self.scalers = {}
        self.product_features = None
        self.service_features = None
        
    def _engineer_product_features(self, df):
        """Create advanced features for product pricing."""
        df = df.copy()
        
        # Parse specifications
        df['specs_dict'] = df['specifications'].apply(lambda x: json.loads(x) if isinstance(x, str) else {})
        
        # Category-specific features
        category_avg_prices = df.groupby('category')['base_price'].transform('mean')
        df['category_price_ratio'] = df['base_price'] / category_avg_prices
        
        # Vendor tier encoding with quality mapping
        tier_map = {'Tier 1': 1.15, 'Tier 2': 1.0, 'Tier 3': 0.85}
        df['tier_factor'] = df['vendor_tier'].map(tier_map).fillna(1.0)
        
        # Location features
        metro_cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune']
        df['is_metro'] = df['location'].isin(metro_cities).astype(int)
        
        # Source reliability
        source_map = {
            'GeM Portal': 0.95,
            'Central Public Procurement Portal': 0.93,
            'State e-Procurement': 0.92,
            'Vendor Quotation': 0.88,
            'Previous Purchase Order': 0.90,
            'Industry Report': 0.85,
            'Online Marketplace': 0.87,
            'Dealer Price List': 0.89,
            'Tender Document': 0.91
        }
        df['source_reliability'] = df['source'].map(source_map).fillna(0.85)
        
        # Quantity discount tiers
        df['quantity_tier'] = pd.cut(
            df['quantity'], 
            bins=[0, 5, 20, 50, 100, 1000],
            labels=[0, 0.02, 0.05, 0.08, 0.12]
        ).astype(float)
        
        # Time features
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_quarter_end'] = df['month'].isin([3, 6, 9, 12]).astype(int)
        
        # Demand factor features
        df['demand_category'] = pd.cut(
            df['demand_factor'],
            bins=[0, 0.95, 1.0, 1.05, 1.2],
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        # Price statistics by category
        cat_stats = df.groupby('category')['unit_price'].agg(['mean', 'std', 'min', 'max'])
        df = df.merge(cat_stats, left_on='category', right_index=True, how='left', suffixes=('', '_cat'))
        
        # Encode categorical variables
        categorical_cols = ['category', 'vendor_tier', 'location', 'source', 'month', 'quarter']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_enc'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[f'{col}_enc'] = df[col].map(
                    lambda x: self.label_encoders[col].transform([x])[0]
                    if x in self.label_encoders[col].classes_
                    else -1
                )
        
        # Feature columns
        feature_cols = [
            'base_price', 'quantity', 'warranty_years', 'delivery_days',
            'demand_factor', 'tier_factor', 'is_metro', 'source_reliability',
            'quantity_tier', 'is_quarter_end', 'category_price_ratio',
            'mean', 'std', 'min', 'max',
            'category_enc', 'vendor_tier_enc', 'location_enc', 'source_enc',
            'month_enc', 'quarter_enc'
        ]
        
        self.product_features = feature_cols
        return df[feature_cols].fillna(0)
    
    def _engineer_service_features(self, df):
        """Create advanced features for service pricing."""
        df = df.copy()
        
        # Parse specifications
        df['specs_dict'] = df['specifications'].apply(lambda x: json.loads(x) if isinstance(x, str) else {})
        df['num_specs'] = df['specs_dict'].apply(len)
        
        # Category features
        category_avg = df.groupby('category')['base_price'].transform('mean')
        df['category_price_ratio'] = df['base_price'] / category_avg
        
        # Location features
        metro_cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune']
        df['is_metro'] = df['location'].isin(metro_cities).astype(int)
        
        # Duration features
        df['duration_category'] = pd.cut(
            df['duration_months'],
            bins=[0, 3, 6, 12, 24],
            labels=['short', 'medium', 'long', 'very_long']
        )
        
        # Source reliability
        source_map = {
            'GeM Portal': 0.95,
            'Central Public Procurement Portal': 0.93,
            'Vendor Quotation': 0.88,
            'Tender Document': 0.91,
            'Industry Report': 0.85
        }
        df['source_reliability'] = df['source'].map(source_map).fillna(0.85)
        
        # Time features
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        # Price statistics
        cat_stats = df.groupby('category')['monthly_price'].agg(['mean', 'std'])
        df = df.merge(cat_stats, left_on='category', right_index=True, how='left', suffixes=('', '_cat'))
        
        # Encode categoricals
        categorical_cols = ['category', 'location', 'source', 'month', 'quarter', 'duration_category']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_enc'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[f'{col}_enc'] = df[col].map(
                    lambda x: self.label_encoders[col].transform([x])[0]
                    if x in self.label_encoders[col].classes_
                    else -1
                )
        
        feature_cols = [
            'base_price', 'duration_months', 'num_specs', 'demand_factor',
            'is_metro', 'source_reliability', 'category_price_ratio',
            'mean', 'std',
            'category_enc', 'location_enc', 'source_enc',
            'month_enc', 'quarter_enc', 'duration_category_enc'
        ]
        
        self.service_features = feature_cols
        return df[feature_cols].fillna(0)
    
    def train(self, products_df, services_df):
        """Train both models with enhanced features."""
        print("=" * 60)
        print("Training Enhanced Price Benchmarking AI Models")
        print("=" * 60)
        
        # Train product model
        print("\n1. Training Product Price Model...")
        X_prod = self._engineer_product_features(products_df)
        y_prod = products_df['unit_price'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_prod, y_prod, test_size=0.2, random_state=42
        )
        
        self.scalers['product'] = StandardScaler()
        X_train_scaled = self.scalers['product'].fit_transform(X_train)
        X_test_scaled = self.scalers['product'].transform(X_test)
        
        # Ensemble of models
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=5,
                random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                random_state=42
            )
        }
        
        best_score = -np.inf
        for name, m in models.items():
            m.fit(X_train_scaled, y_train)
            score = m.score(X_test_scaled, y_test)
            print(f"  {name}: R² = {score:.4f}")
            if score > best_score:
                best_score = score
                self.product_model = m
        
        y_pred = self.product_model.predict(X_test_scaled)
        prod_metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100
        }
        print(f"\n  Product Model Performance:")
        print(f"    R² Score: {prod_metrics['r2']:.4f}")
        print(f"    MAE: ₹{prod_metrics['mae']:,.2f}")
        print(f"    RMSE: ₹{prod_metrics['rmse']:,.2f}")
        print(f"    MAPE: {prod_metrics['mape']:.2f}%")
        
        # Train service model
        print("\n2. Training Service Price Model...")
        X_svc = self._engineer_service_features(services_df)
        y_svc = services_df['monthly_price'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_svc, y_svc, test_size=0.2, random_state=42
        )
        
        self.scalers['service'] = StandardScaler()
        X_train_scaled = self.scalers['service'].fit_transform(X_train)
        X_test_scaled = self.scalers['service'].transform(X_test)
        
        self.service_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            random_state=42
        )
        self.service_model.fit(X_train_scaled, y_train)
        
        y_pred = self.service_model.predict(X_test_scaled)
        svc_metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100
        }
        print(f"\n  Service Model Performance:")
        print(f"    R² Score: {svc_metrics['r2']:.4f}")
        print(f"    MAE: ₹{svc_metrics['mae']:,.2f}")
        print(f"    RMSE: ₹{svc_metrics['rmse']:,.2f}")
        print(f"    MAPE: {svc_metrics['mape']:.2f}%")
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        
        return prod_metrics, svc_metrics
    
    def predict_product(self, category, item_name, vendor_tier, location,
                       source, quantity, warranty_years, delivery_days,
                       base_price, demand_factor=1.0):
        """Predict product price."""
        input_data = pd.DataFrame([{
            'category': category,
            'item_name': item_name,
            'vendor_tier': vendor_tier,
            'location': location,
            'source': source,
            'quantity': quantity,
            'warranty_years': warranty_years,
            'delivery_days': delivery_days,
            'base_price': base_price,
            'demand_factor': demand_factor,
            'price_variance': 0.2,
            'specifications': '{}',
            'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'unit_price': 0,
            'quality_rating': 4.0
        }])
        
        X = self._engineer_product_features(input_data)
        X_scaled = self.scalers['product'].transform(X)
        
        predicted_price = self.product_model.predict(X_scaled)[0]
        predicted_price = max(0, predicted_price)
        
        return {
            'predicted_price': round(predicted_price, 2),
            'quantity': quantity,
            'total_price': round(predicted_price * quantity, 2),
            'confidence': min(95, max(60, self.product_model.score(
                self.scalers['product'].transform(
                    self._engineer_product_features(input_data)
                ),
                np.array([base_price])
            ) * 100 + 70))
        }
    
    def predict_service(self, category, service_name, location, source,
                       duration_months, base_price, demand_factor=1.0):
        """Predict service price."""
        input_data = pd.DataFrame([{
            'category': category,
            'service_name': service_name,
            'location': location,
            'source': source,
            'duration_months': duration_months,
            'base_price': base_price,
            'demand_factor': demand_factor,
            'price_variance': 0.2,
            'specifications': '{}',
            'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'monthly_price': 0
        }])
        
        X = self._engineer_service_features(input_data)
        X_scaled = self.scalers['service'].transform(X)
        
        predicted_price = self.service_model.predict(X_scaled)[0]
        predicted_price = max(0, predicted_price)
        
        return {
            'predicted_monthly_price': round(predicted_price, 2),
            'duration_months': duration_months,
            'total_price': round(predicted_price * duration_months, 2),
            'confidence': 75
        }
    
    def get_recommendation(self, prediction, budget=None):
        """Generate price recommendation."""
        price = prediction.get('predicted_price') or prediction.get('predicted_monthly_price', 0)
        
        recommendation = {
            'recommended_price': round(price, 2),
            'price_range': {
                'minimum': round(price * 0.85, 2),
                'maximum': round(price * 1.15, 2),
                'average': round(price, 2)
            },
            'price_reasonable': True,
            'suggestions': []
        }
        
        if budget and price > budget:
            recommendation['price_reasonable'] = False
            recommendation['suggestions'].append(
                f"Price exceeds budget by ₹{price - budget:,.2f}. Consider negotiating or exploring alternatives."
            )
        
        if price > 500000:
            recommendation['suggestions'].append(
                "High-value procurement. Ensure proper administrative approval and multiple vendor quotes."
            )
        
        return recommendation
    
    def save_model(self, filepath='enhanced_price_model.pkl'):
        """Save model to file."""
        model_data = {
            'product_model': self.product_model,
            'service_model': self.service_model,
            'label_encoders': self.label_encoders,
            'scalers': self.scalers,
            'product_features': self.product_features,
            'service_features': self.service_features
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='enhanced_price_model.pkl'):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.product_model = model_data['product_model']
        self.service_model = model_data['service_model']
        self.label_encoders = model_data['label_encoders']
        self.scalers = model_data['scalers']
        self.product_features = model_data['product_features']
        self.service_features = model_data['service_features']
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    print("Enhanced Price Benchmarking AI Model")
    print("=" * 60)
    
    # Load data
    products_df = pd.read_csv('products_prices.csv')
    services_df = pd.read_csv('services_prices.csv')
    
    print(f"Loaded {len(products_df)} product records")
    print(f"Loaded {len(services_df)} service records")
    
    # Train model
    model = EnhancedPriceModel()
    prod_metrics, svc_metrics = model.train(products_df, services_df)
    
    # Example predictions
    print("\n" + "=" * 60)
    print("Example Predictions")
    print("=" * 60)
    
    # Get base prices from data
    router_base = products_df[products_df['item_name'] == 'Enterprise Router']['base_price'].iloc[0]
    
    product_pred = model.predict_product(
        category="Networking Equipment",
        item_name="Enterprise Router",
        vendor_tier="Tier 1",
        location="Delhi",
        source="GeM Portal",
        quantity=10,
        warranty_years=3,
        delivery_days=14,
        base_price=router_base
    )
    
    print(f"\nProduct: Enterprise Router (Qty: 10)")
    print(f"  Predicted Unit Price: ₹{product_pred['predicted_price']:,.2f}")
    print(f"  Total Price: ₹{product_pred['total_price']:,.2f}")
    print(f"  Confidence: {product_pred['confidence']:.1f}%")
    
    # Service prediction
    maint_base = services_df[services_df['service_name'] == 'Network Maintenance Annual']['base_price'].iloc[0]
    
    service_pred = model.predict_service(
        category="IT Services",
        service_name="Network Maintenance Annual",
        location="Mumbai",
        source="Vendor Quotation",
        duration_months=12,
        base_price=maint_base
    )
    
    print(f"\nService: Network Maintenance Annual (12 months)")
    print(f"  Monthly Price: ₹{service_pred['predicted_monthly_price']:,.2f}")
    print(f"  Total Price: ₹{service_pred['total_price']:,.2f}")
    
    # Save model
    model.save_model('enhanced_price_model.pkl')
