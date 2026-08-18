"""
Price Benchmarking AI Model
Machine learning model for predicting and recommending procurement prices.
"""

import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

class PriceBenchmarkingModel:
    """
    AI Model for Government Procurement Price Benchmarking.
    Predicts reasonable prices for products and services based on specifications.
    """
    
    def __init__(self):
        self.product_model = None
        self.service_model = None
        self.product_preprocessor = None
        self.service_preprocessor = None
        self.label_encoders = {}
        self.scalers = {}
        self.feature_columns = {}
        
    def _prepare_product_features(self, df):
        """Prepare features for product price prediction."""
        df = df.copy()
        
        # Parse specifications JSON
        df['specs_dict'] = df['specifications'].apply(json.loads)
        
        # Extract common spec features
        df['num_specs'] = df['specs_dict'].apply(len)
        
        # Encode categorical variables
        categorical_cols = ['category', 'vendor_tier', 'location', 'source']
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen labels
                    df[f'{col}_encoded'] = df[col].map(
                        lambda x: self.label_encoders[col].transform([x])[0] 
                        if x in self.label_encoders[col].classes_ 
                        else -1
                    )
        
        # Numerical features
        numerical_cols = ['quantity', 'warranty_years', 'delivery_days', 'demand_factor']
        
        # Add encoded and numerical features
        feature_cols = [f'{col}_encoded' for col in categorical_cols] + numerical_cols
        
        return df[feature_cols], feature_cols
    
    def _prepare_service_features(self, df):
        """Prepare features for service price prediction."""
        df = df.copy()
        
        # Parse specifications JSON
        df['specs_dict'] = df['specifications'].apply(json.loads)
        df['num_specs'] = df['specs_dict'].apply(len)
        
        # Encode categorical variables
        categorical_cols = ['category', 'location', 'source']
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[f'{col}_encoded'] = df[col].map(
                        lambda x: self.label_encoders[col].transform([x])[0] 
                        if x in self.label_encoders[col].classes_ 
                        else -1
                    )
        
        # Numerical features
        numerical_cols = ['duration_months', 'demand_factor']
        
        feature_cols = [f'{col}_encoded' for col in categorical_cols] + numerical_cols
        
        return df[feature_cols], feature_cols
    
    def train_product_model(self, products_df):
        """Train the product price prediction model."""
        print("Training Product Price Model...")
        
        X, feature_cols = self._prepare_product_features(products_df)
        y = products_df['unit_price'].values
        
        self.feature_columns['product'] = feature_cols
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scalers['product'] = StandardScaler()
        X_train_scaled = self.scalers['product'].fit_transform(X_train)
        X_test_scaled = self.scalers['product'].transform(X_test)
        
        # Train multiple models and select best
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Ridge Regression': Ridge(alpha=1.0)
        }
        
        best_score = -np.inf
        best_model_name = None
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            print(f"  {name}: R² = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model_name = name
                self.product_model = model
        
        # Evaluate best model
        y_pred = self.product_model.predict(X_test_scaled)
        
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        print(f"\n  Best Model: {best_model_name}")
        print(f"  R² Score: {metrics['r2_score']:.4f}")
        print(f"  MAE: ₹{metrics['mae']:,.2f}")
        print(f"  RMSE: ₹{metrics['rmse']:,.2f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        
        return metrics
    
    def train_service_model(self, services_df):
        """Train the service price prediction model."""
        print("\nTraining Service Price Model...")
        
        X, feature_cols = self._prepare_service_features(services_df)
        y = services_df['monthly_price'].values
        
        self.feature_columns['service'] = feature_cols
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scalers['service'] = StandardScaler()
        X_train_scaled = self.scalers['service'].fit_transform(X_train)
        X_test_scaled = self.scalers['service'].transform(X_test)
        
        # Train model
        self.service_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.service_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.service_model.predict(X_test_scaled)
        
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        print(f"  R² Score: {metrics['r2_score']:.4f}")
        print(f"  MAE: ₹{metrics['mae']:,.2f}")
        print(f"  RMSE: ₹{metrics['rmse']:,.2f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        
        return metrics
    
    def train(self, products_df, services_df):
        """Train both product and service models."""
        print("=" * 60)
        print("Training Price Benchmarking AI Models")
        print("=" * 60)
        
        product_metrics = self.train_product_model(products_df)
        service_metrics = self.train_service_model(services_df)
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        
        return product_metrics, service_metrics
    
    def predict_product_price(self, category, item_name, vendor_tier, location, 
                             source, quantity, warranty_years, delivery_days, 
                             demand_factor=1.0):
        """Predict price for a product."""
        if self.product_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create input DataFrame
        input_data = pd.DataFrame([{
            'category': category,
            'vendor_tier': vendor_tier,
            'location': location,
            'source': source,
            'quantity': quantity,
            'warranty_years': warranty_years,
            'delivery_days': delivery_days,
            'demand_factor': demand_factor,
            'specifications': '{}'
        }])
        
        X, _ = self._prepare_product_features(input_data)
        X_scaled = self.scalers['product'].transform(X)
        
        predicted_price = self.product_model.predict(X_scaled)[0]
        
        return {
            'predicted_price': round(predicted_price, 2),
            'quantity': quantity,
            'total_price': round(predicted_price * quantity, 2),
            'confidence': self._get_prediction_confidence(X_scaled, 'product')
        }
    
    def predict_service_price(self, category, service_name, location, source,
                             duration_months, demand_factor=1.0):
        """Predict price for a service."""
        if self.service_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        input_data = pd.DataFrame([{
            'category': category,
            'location': location,
            'source': source,
            'duration_months': duration_months,
            'demand_factor': demand_factor,
            'specifications': '{}'
        }])
        
        X, _ = self._prepare_service_features(input_data)
        X_scaled = self.scalers['service'].transform(X)
        
        predicted_price = self.service_model.predict(X_scaled)[0]
        
        return {
            'predicted_monthly_price': round(predicted_price, 2),
            'duration_months': duration_months,
            'total_price': round(predicted_price * duration_months, 2),
            'confidence': self._get_prediction_confidence(X_scaled, 'service')
        }
    
    def _get_prediction_confidence(self, X_scaled, model_type):
        """Estimate prediction confidence based on training data distribution."""
        # Simple confidence estimation based on distance from training mean
        if model_type == 'product':
            scaler = self.scalers['product']
        else:
            scaler = self.scalers['service']
        
        # Calculate Mahalanobis-like distance
        mean = scaler.mean_
        std = scaler.scale_
        distance = np.mean(np.abs((X_scaled - mean) / (std + 1e-8)))
        
        # Convert distance to confidence score (0-1)
        confidence = max(0, min(1, 1 - (distance / 3)))
        
        return round(confidence, 2)
    
    def get_price_recommendation(self, predictions, budget=None):
        """Generate price recommendation based on predictions."""
        if isinstance(predictions, dict):
            prices = [predictions['predicted_price']]
        else:
            prices = [p['predicted_price'] for p in predictions]
        
        avg_price = np.mean(prices)
        min_price = np.min(prices)
        max_price = np.max(prices)
        std_price = np.std(prices) if len(prices) > 1 else 0
        
        recommendation = {
            'recommended_price': round(avg_price, 2),
            'price_range': {
                'minimum': round(min_price, 2),
                'maximum': round(max_price, 2),
                'average': round(avg_price, 2),
                'std_dev': round(std_price, 2)
            },
            'price_reasonable': True,
            'suggestions': []
        }
        
        # Check if within budget
        if budget:
            if avg_price > budget:
                recommendation['price_reasonable'] = False
                recommendation['suggestions'].append(
                    f"Price exceeds budget by ₹{avg_price - budget:,.2f}"
                )
        
        # Add suggestions based on price variance
        if std_price > avg_price * 0.2:
            recommendation['suggestions'].append(
                "High price variance detected. Consider negotiating with vendors."
            )
        
        return recommendation
    
    def save_model(self, filepath='price_benchmarking_model.pkl'):
        """Save trained model to file."""
        model_data = {
            'product_model': self.product_model,
            'service_model': self.service_model,
            'label_encoders': self.label_encoders,
            'scalers': self.scalers,
            'feature_columns': self.feature_columns
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='price_benchmarking_model.pkl'):
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.product_model = model_data['product_model']
        self.service_model = model_data['service_model']
        self.label_encoders = model_data['label_encoders']
        self.scalers = model_data['scalers']
        self.feature_columns = model_data['feature_columns']
        
        print(f"Model loaded from {filepath}")


def analyze_market_trends(df):
    """Analyze market trends from procurement data."""
    trends = {
        'category_trends': {},
        'location_analysis': {},
        'vendor_analysis': {},
        'temporal_trends': {}
    }
    
    # Category-wise analysis
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        trends['category_trends'][category] = {
            'avg_price': round(cat_data['unit_price'].mean(), 2),
            'price_range': {
                'min': round(cat_data['unit_price'].min(), 2),
                'max': round(cat_data['unit_price'].max(), 2)
            },
            'total_entries': len(cat_data),
            'top_vendors': cat_data['vendor'].value_counts().head(3).to_dict()
        }
    
    # Location-wise analysis
    for location in df['location'].unique():
        loc_data = df[df['location'] == location]
        trends['location_analysis'][location] = {
            'avg_price': round(loc_data['unit_price'].mean(), 2),
            'total_procurements': len(loc_data)
        }
    
    # Vendor analysis
    for vendor in df['vendor'].unique():
        vendor_data = df[df['vendor'] == vendor]
        trends['vendor_analysis'][vendor] = {
            'avg_price': round(vendor_data['unit_price'].mean(), 2),
            'total_entries': len(vendor_data),
            'avg_quality': round(vendor_data['quality_rating'].mean(), 2)
        }
    
    return trends


if __name__ == "__main__":
    # Example usage
    print("Price Benchmarking AI Model")
    print("=" * 60)
    
    # Load data (assuming generate_dataset.py has been run)
    try:
        products_df = pd.read_csv('products_prices.csv')
        services_df = pd.read_csv('services_prices.csv')
    except FileNotFoundError:
        print("Please run generate_dataset.py first to create the datasets.")
        exit(1)
    
    # Initialize and train model
    model = PriceBenchmarkingModel()
    product_metrics, service_metrics = model.train(products_df, services_df)
    
    # Example predictions
    print("\n" + "=" * 60)
    print("Example Predictions")
    print("=" * 60)
    
    # Product prediction
    product_prediction = model.predict_product_price(
        category="Networking Equipment",
        item_name="Enterprise Router",
        vendor_tier="Tier 1",
        location="Delhi",
        source="GeM Portal",
        quantity=10,
        warranty_years=3,
        delivery_days=14,
        demand_factor=1.0
    )
    
    print(f"\nProduct Prediction:")
    print(f"  Predicted Unit Price: ₹{product_prediction['predicted_price']:,.2f}")
    print(f"  Quantity: {product_prediction['quantity']}")
    print(f"  Total Price: ₹{product_prediction['total_price']:,.2f}")
    print(f"  Confidence: {product_prediction['confidence'] * 100}%")
    
    # Service prediction
    service_prediction = model.predict_service_price(
        category="IT Services",
        service_name="Network Maintenance Annual",
        location="Mumbai",
        source="Vendor Quotation",
        duration_months=12,
        demand_factor=1.0
    )
    
    print(f"\nService Prediction:")
    print(f"  Predicted Monthly Price: ₹{service_prediction['predicted_monthly_price']:,.2f}")
    print(f"  Duration: {service_prediction['duration_months']} months")
    print(f"  Total Price: ₹{service_prediction['total_price']:,.2f}")
    print(f"  Confidence: {service_prediction['confidence'] * 100}%")
    
    # Get recommendation
    recommendation = model.get_price_recommendation(
        product_prediction, 
        budget=50000
    )
    
    print(f"\nPrice Recommendation:")
    print(f"  Recommended Price: ₹{recommendation['recommended_price']:,.2f}")
    print(f"  Price Range: ₹{recommendation['price_range']['minimum']:,.2f} - ₹{recommendation['price_range']['maximum']:,.2f}")
    print(f"  Within Budget: {'Yes' if recommendation['price_reasonable'] else 'No'}")
    
    # Save model
    model.save_model('price_benchmarking_model.pkl')
