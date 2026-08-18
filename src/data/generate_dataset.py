"""
Synthetic Dataset Generator for Government Procurement Price Benchmarking
Generates realistic product and service pricing data for training ML models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

np.random.seed(42)
random.seed(42)

# Product categories with specifications and price ranges
PRODUCT_CATEGORIES = {
    "Networking Equipment": {
        "items": [
            {"name": "Enterprise Router", "specs": {"speed": "1Gbps", "ports": 8, "type": "wired"}, "base_price": 45000, "variance": 0.25},
            {"name": "Managed Switch 24-Port", "specs": {"speed": "1Gbps", "ports": 24, "type": "managed"}, "base_price": 35000, "variance": 0.20},
            {"name": "Managed Switch 48-Port", "specs": {"speed": "1Gbps", "ports": 48, "type": "managed"}, "base_price": 75000, "variance": 0.20},
            {"name": "Wireless Access Point", "specs": {"standard": "WiFi 6", "range": "50m", "bands": "dual"}, "base_price": 12000, "variance": 0.30},
            {"name": "Network Firewall", "specs": {"throughput": "10Gbps", "vpn": True, "type": "hardware"}, "base_price": 150000, "variance": 0.35},
            {"name": "Network Switch Layer 3", "specs": {"speed": "10Gbps", "ports": 16, "routing": True}, "base_price": 120000, "variance": 0.25},
            {"name": "Load Balancer", "specs": {"throughput": "5Gbps", "ssl": True, "HA": True}, "base_price": 200000, "variance": 0.30},
        ],
        "vendors": ["Cisco", "HP Enterprise", "Juniper", "Fortinet", "D-Link", "TP-Link", "ZTE"]
    },
    "Computing Hardware": {
        "items": [
            {"name": "Desktop Workstation", "specs": {"ram": "32GB", "cpu": "Intel i7", "storage": "1TB SSD"}, "base_price": 85000, "variance": 0.20},
            {"name": "Server Rack Mount", "specs": {"ram": "64GB", "cpu": "Intel Xeon", "storage": "2TB SSD", "rack_units": 2}, "base_price": 350000, "variance": 0.25},
            {"name": "Laptop Standard", "specs": {"ram": "16GB", "cpu": "Intel i5", "storage": "512GB SSD", "screen": "14inch"}, "base_price": 55000, "variance": 0.15},
            {"name": "Laptop High-End", "specs": {"ram": "32GB", "cpu": "Intel i7", "storage": "1TB SSD", "screen": "15.6inch", "gpu": "dedicated"}, "base_price": 95000, "variance": 0.20},
            {"name": "Graphics Workstation", "specs": {"ram": "64GB", "cpu": "Intel i9", "gpu": "NVIDIA RTX 4000", "storage": "2TB SSD"}, "base_price": 250000, "variance": 0.30},
            {"name": "Thin Client", "specs": {"ram": "4GB", "storage": "32GB SSD", "type": "zero client"}, "base_price": 18000, "variance": 0.15},
            {"name": "Mini PC", "specs": {"ram": "8GB", "cpu": "Intel i3", "storage": "256GB SSD"}, "base_price": 25000, "variance": 0.20},
        ],
        "vendors": ["Dell", "HP", "Lenovo", "Acer", "ASUS", "Apple", "Intel NUC"]
    },
    "Storage Systems": {
        "items": [
            {"name": "Hard Disk Drive 1TB", "specs": {"capacity": "1TB", "type": "HDD", "rpm": 7200}, "base_price": 4500, "variance": 0.15},
            {"name": "Hard Disk Drive 4TB", "specs": {"capacity": "4TB", "type": "HDD", "rpm": 7200}, "base_price": 12000, "variance": 0.20},
            {"name": "SSD 512GB", "specs": {"capacity": "512GB", "type": "SSD", "interface": "NVMe"}, "base_price": 5500, "variance": 0.25},
            {"name": "SSD 2TB", "specs": {"capacity": "2TB", "type": "SSD", "interface": "NVMe"}, "base_price": 18000, "variance": 0.25},
            {"name": "NAS Storage 4-Bay", "specs": {"bays": 4, "type": "NAS", "max_capacity": "64TB"}, "base_price": 45000, "variance": 0.20},
            {"name": "SAN Storage Array", "specs": {"capacity": "100TB", "type": "SAN", "redundancy": True}, "base_price": 500000, "variance": 0.30},
            {"name": "Tape Library", "specs": {"slots": 8, "type": "LTO-8", "capacity": "12TB"}, "base_price": 300000, "variance": 0.25},
        ],
        "vendors": ["Seagate", "Western Digital", "Synology", "QNAP", "Dell EMC", "NetApp", "HPE"]
    },
    "Display Systems": {
        "items": [
            {"name": "LED Monitor 24-inch", "specs": {"size": 24, "resolution": "1080p", "panel": "IPS"}, "base_price": 14000, "variance": 0.15},
            {"name": "LED Monitor 27-inch", "specs": {"size": 27, "resolution": "1440p", "panel": "IPS"}, "base_price": 22000, "variance": 0.20},
            {"name": "4K Monitor 32-inch", "specs": {"size": 32, "resolution": "4K", "panel": "IPS"}, "base_price": 38000, "variance": 0.25},
            {"name": "Interactive Display 65-inch", "specs": {"size": 65, "resolution": "4K", "touch": True}, "base_price": 250000, "variance": 0.30},
            {"name": "Video Wall Display 55-inch", "specs": {"size": 55, "resolution": "4K", "bezel": "narrow"}, "base_price": 180000, "variance": 0.25},
            {"name": "Projector HD", "specs": {"brightness": "4000 lumens", "resolution": "1080p", "type": "DLP"}, "base_price": 65000, "variance": 0.25},
            {"name": "Projector 4K", "specs": {"brightness": "5000 lumens", "resolution": "4K", "type": "LCD"}, "base_price": 180000, "variance": 0.30},
        ],
        "vendors": ["Samsung", "LG", "Dell", "BenQ", "Epson", "Sony", "ViewSonic"]
    },
    "Communication Systems": {
        "items": [
            {"name": "IP Phone System", "specs": {"lines": 8, "type": "VoIP", "display": True}, "base_price": 25000, "variance": 0.20},
            {"name": "PBX System", "specs": {"extensions": 50, "type": "IP-PBX", "recording": True}, "base_price": 350000, "variance": 0.30},
            {"name": "Video Conferencing System", "specs": {"resolution": "1080p", "participants": 12, "camera": True}, "base_price": 150000, "variance": 0.25},
            {"name": "HF Radio Equipment", "specs": {"frequency": "HF", "power": "100W", "mode": "SSB"}, "base_price": 85000, "variance": 0.35},
            {"name": "VHF Radio Equipment", "specs": {"frequency": "VHF", "power": "25W", "mode": "FM"}, "base_price": 45000, "variance": 0.30},
            {"name": "Software Defined Radio", "specs": {"frequency": "wideband", "type": "SDR", "bandwidth": "40MHz"}, "base_price": 120000, "variance": 0.40},
            {"name": "Satellite Communication Terminal", "specs": {"type": "VSAT", "band": "Ku", "speed": "2Mbps"}, "base_price": 450000, "variance": 0.35},
        ],
        "vendors": ["Cisco", "Avaya", "Polycom", "Yealink", "Motorola", "Icom", "Kenwood"]
    },
    "Power Systems": {
        "items": [
            {"name": "UPS 1KVA", "specs": {"capacity": "1KVA", "type": "online", "battery": "30min"}, "base_price": 12000, "variance": 0.15},
            {"name": "UPS 3KVA", "specs": {"capacity": "3KVA", "type": "online", "battery": "30min"}, "base_price": 28000, "variance": 0.20},
            {"name": "UPS 10KVA", "specs": {"capacity": "10KVA", "type": "online", "battery": "30min"}, "base_price": 85000, "variance": 0.25},
            {"name": "Diesel Generator 50KVA", "specs": {"capacity": "50KVA", "fuel": "diesel", "auto": True}, "base_price": 450000, "variance": 0.20},
            {"name": "Diesel Generator 100KVA", "specs": {"capacity": "100KVA", "fuel": "diesel", "auto": True}, "base_price": 750000, "variance": 0.20},
            {"name": "Solar Panel 300W", "specs": {"wattage": 300, "type": "monocrystalline", "efficiency": "20%"}, "base_price": 12000, "variance": 0.25},
            {"name": "Solar Inverter 5KVA", "specs": {"capacity": "5KVA", "type": "hybrid", "battery_support": True}, "base_price": 35000, "variance": 0.30},
        ],
        "vendors": ["APC", "Eaton", "Vertiv", "Microtek", "Sukam", "Luminous", "V-Guard"]
    },
    "Security Systems": {
        "items": [
            {"name": "CCTV Camera Indoor", "specs": {"resolution": "2MP", "type": "dome", "night_vision": True}, "base_price": 4500, "variance": 0.20},
            {"name": "CCTV Camera Outdoor", "specs": {"resolution": "4MP", "type": "bullet", "IP66": True}, "base_price": 7500, "variance": 0.25},
            {"name": "NVR 16-Channel", "specs": {"channels": 16, "storage": "4TB", "resolution": "4K"}, "base_price": 35000, "variance": 0.20},
            {"name": "Biometric Access Control", "specs": {"type": "fingerprint", "users": 1000, "door": 4}, "base_price": 65000, "variance": 0.30},
            {"name": "Fire Alarm System", "specs": {"zones": 32, "type": "addressable", "panel": True}, "base_price": 120000, "variance": 0.25},
            {"name": "Video Management Software", "specs": {"cameras": 64, "type": "enterprise", "analytics": True}, "base_price": 180000, "variance": 0.35},
            {"name": "Intrusion Detection System", "specs": {"sensors": 16, "type": "wireless", "monitoring": True}, "base_price": 45000, "variance": 0.30},
        ],
        "vendors": ["Hikvision", "Dahua", "Axis", "Bosch", "Honeywell", "CP Plus", "Samsung"]
    },
    "Office Furniture": {
        "items": [
            {"name": "Office Desk Executive", "specs": {"size": "160x80cm", "material": "engineered wood", "drawer": 3}, "base_price": 15000, "variance": 0.25},
            {"name": "Office Chair Ergonomic", "specs": {"type": "ergonomic", "adjustable": True, "lumbar": True}, "base_price": 18000, "variance": 0.30},
            {"name": "Filing Cabinet 4-Drawer", "specs": {"drawers": 4, "material": "metal", "lock": True}, "base_price": 12000, "variance": 0.20},
            {"name": "Conference Table 12-Seater", "specs": {"seats": 12, "material": "wood", "size": "360x120cm"}, "base_price": 65000, "variance": 0.30},
            {"name": "Modular Workstation", "specs": {"seats": 6, "material": "engineered wood", "partitions": True}, "base_price": 85000, "variance": 0.25},
            {"name": "Bookshelf 5-Tier", "specs": {"tiers": 5, "material": "metal", "height": "180cm"}, "base_price": 8000, "variance": 0.15},
            {"name": "Office Sofa 3-Seater", "specs": {"seats": 3, "material": "fabric", "frame": "metal"}, "base_price": 22000, "variance": 0.25},
        ],
        "vendors": ["Godrej", "Featherlite", "Ergo", "Fantasy", "Style Spa", "HATIM", "Wipro"]
    },
    "Electrical Equipment": {
        "items": [
            {"name": "Split AC 1.5 Ton", "specs": {"capacity": "1.5 ton", "rating": "5 star", "type": "inverter"}, "base_price": 42000, "variance": 0.15},
            {"name": "Split AC 2 Ton", "specs": {"capacity": "2 ton", "rating": "5 star", "type": "inverter"}, "base_price": 55000, "variance": 0.15},
            {"name": "Ceiling Fan", "specs": {"speed": "380 RPM", "sweep": "1200mm", "type": "BLDC"}, "base_price": 3500, "variance": 0.20},
            {"name": "Tube Light LED 4ft", "specs": {"wattage": 20, "length": "4ft", "color": "cool white"}, "base_price": 600, "variance": 0.15},
            {"name": "LED Bulb 15W", "specs": {"wattage": 15, "lumens": 1500, "color": "cool white"}, "base_price": 200, "variance": 0.20},
            {"name": "MCB Distribution Board", "specs": {"ways": 12, "type": "MCB", "rating": "63A"}, "base_price": 4500, "variance": 0.15},
            {"name": "Stabilizer 5KVA", "specs": {"capacity": "5KVA", "type": "digital", "range": "150-280V"}, "base_price": 8000, "variance": 0.20},
        ],
        "vendors": ["LG", "Daikin", "Voltas", "Crompton", "Philips", "Havells", "Anchor"]
    },
    "Audio/Video Equipment": {
        "items": [
            {"name": "PA System 100W", "specs": {"power": "100W", "speakers": 2, "mixer": True}, "base_price": 25000, "variance": 0.25},
            {"name": "Sound System 500W", "specs": {"power": "500W", "speakers": 4, "subwoofer": True}, "base_price": 85000, "variance": 0.30},
            {"name": "Digital Signage Display", "specs": {"size": 43, "resolution": "4K", "brightness": "500nits"}, "base_price": 55000, "variance": 0.25},
            {"name": "Recording System", "specs": {"channels": 16, "type": "digital", "storage": "1TB"}, "base_price": 120000, "variance": 0.30},
            {"name": "Media Player Digital", "specs": {"resolution": "4K", "storage": "32GB", "wifi": True}, "base_price": 8000, "variance": 0.20},
            {"name": "Digital Podium", "specs": {"display": True, "mic": "gooseneck", "amplifier": True}, "base_price": 75000, "variance": 0.30},
            {"name": "Lecture Capture System", "specs": {"resolution": "1080p", "recording": True, "streaming": True}, "base_price": 180000, "variance": 0.35},
        ],
        "vendors": ["JBL", "Bose", "Yamaha", "Sennheiser", "Samsung", "LG", "Crestron"]
    }
}

# Service categories
SERVICE_CATEGORIES = {
    "IT Services": {
        "items": [
            {"name": "Network Maintenance Annual", "specs": {"scope": "complete network", "support": "24x7", "response": "4hrs"}, "base_price": 250000, "variance": 0.30},
            {"name": "Server Maintenance Annual", "specs": {"servers": 5, "support": "24x7", "response": "4hrs"}, "base_price": 180000, "variance": 0.25},
            {"name": "Software AMC", "specs": {"licenses": 50, "type": "enterprise", "support": "business hours"}, "base_price": 150000, "variance": 0.30},
            {"name": "Cloud Migration Service", "specs": {"scope": "complete", "data": "10TB", "downtime": "minimal"}, "base_price": 500000, "variance": 0.40},
            {"name": "Cybersecurity Audit", "specs": {"scope": "complete", "report": True, "remediation": True}, "base_price": 300000, "variance": 0.35},
            {"name": "Data Backup Service Monthly", "specs": {"data": "5TB", "frequency": "daily", "retention": "90 days"}, "base_price": 25000, "variance": 0.20},
            {"name": "IT Helpdesk Support Monthly", "specs": {"users": 100, "response": "1hr", "resolution": "4hrs"}, "base_price": 45000, "variance": 0.25},
        ]
    },
    "Facility Management": {
        "items": [
            {"name": "Housekeeping Services Monthly", "specs": {"area": "5000sqft", "staff": 4, "frequency": "daily"}, "base_price": 35000, "variance": 0.20},
            {"name": "Security Guard Services Monthly", "specs": {"guards": 4, "shift": "24x7", "armed": False}, "base_price": 80000, "variance": 0.25},
            {"name": "Pest Control Quarterly", "specs": {"area": "5000sqft", "type": "commercial", "guarantee": "3 months"}, "base_price": 8000, "variance": 0.20},
            {"name": "Garden Maintenance Monthly", "specs": {"area": "2000sqft", "frequency": "weekly", "irrigation": True}, "base_price": 12000, "variance": 0.25},
            {"name": "Electrical Maintenance Annual", "specs": {"scope": "complete", "testing": True, "certification": True}, "base_price": 45000, "variance": 0.20},
            {"name": "Plumbing Services Monthly", "specs": {"scope": "complete building", "response": "24hrs"}, "base_price": 15000, "variance": 0.25},
            {"name": "Building Management System Annual", "specs": {"floors": 5, "monitoring": True, "reporting": True}, "base_price": 180000, "variance": 0.30},
        ]
    },
    "Professional Services": {
        "items": [
            {"name": "Web Development Project", "specs": {"pages": 20, "responsive": True, "CMS": True}, "base_price": 200000, "variance": 0.35},
            {"name": "Mobile App Development", "specs": {"platform": "both", "features": "moderate", "maintenance": "6 months"}, "base_price": 400000, "variance": 0.40},
            {"name": "ERP Implementation", "specs": {"modules": "complete", "users": 100, "customization": "moderate"}, "base_price": 800000, "variance": 0.35},
            {"name": "Training Program 5-Day", "specs": {"participants": 20, "topic": "IT", "certification": True}, "base_price": 150000, "variance": 0.30},
            {"name": "Consulting Service Man-Day", "specs": {"domain": "IT", "level": "senior", "travel": True}, "base_price": 25000, "variance": 0.25},
            {"name": "Document Digitization Project", "specs": {"pages": 10000, "scanning": True, "indexing": True}, "base_price": 50000, "variance": 0.20},
            {"name": "Annual Maintenance Contract IT", "specs": {"scope": "complete IT", "support": "24x7", "spare_parts": True}, "base_price": 400000, "variance": 0.30},
        ]
    }
}

# Locations for regional price variation
LOCATIONS = [
    "Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata",
    "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "Chandigarh", "Bhopal", "Thiruvananthapuram", "Guwahati", "Patna"
]

# Seasons affecting demand
SEASONS = {
    "Q1": {"month_range": (1, 3), "demand_factor": 1.0},
    "Q2": {"month_range": (4, 6), "demand_factor": 0.95},
    "Q3": {"month_range": (7, 9), "demand_factor": 0.90},
    "Q4": {"month_range": (10, 12), "demand_factor": 1.10},  # End of financial year rush
}

# Government procurement sources
SOURCES = [
    "GeM Portal", "Central Public Procurement Portal", "State e-Procurement",
    "Vendor Quotation", "Previous Purchase Order", "Industry Report",
    "Online Marketplace", "Dealer Price List", "Tender Document"
]

# Vendor reliability ratings
VENDOR_RATINGS = {
    "Tier 1": {"price_factor": 1.15, "quality": 0.98, "reliability": 0.95},
    "Tier 2": {"price_factor": 1.00, "quality": 0.95, "reliability": 0.90},
    "Tier 3": {"price_factor": 0.85, "quality": 0.88, "reliability": 0.80},
}

def get_season(month):
    for season, info in SEASONS.items():
        if month >= info["month_range"][0] and month <= info["month_range"][1]:
            return season, info["demand_factor"]
    return "Q1", 1.0

def generate_vendor_price(base_price, variance, vendor_tier, location):
    """Generate realistic vendor price based on multiple factors."""
    # Base price with variance
    price = base_price * (1 + np.random.uniform(-variance, variance))
    
    # Vendor tier adjustment
    price *= VENDOR_RATINGS[vendor_tier]["price_factor"]
    
    # Location factor (metro cities tend to be slightly higher)
    metro_cities = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
    if location in metro_cities:
        price *= np.random.uniform(1.02, 1.08)
    else:
        price *= np.random.uniform(0.95, 1.02)
    
    # Add some randomness for market conditions
    price *= np.random.uniform(0.90, 1.10)
    
    return round(price, 2)

def generate_products_dataset(num_entries=5000):
    """Generate synthetic product pricing dataset."""
    data = []
    
    for i in range(num_entries):
        category = random.choice(list(PRODUCT_CATEGORIES.keys()))
        category_data = PRODUCT_CATEGORIES[category]
        item = random.choice(category_data["items"])
        vendor = random.choice(category_data["vendors"])
        location = random.choice(LOCATIONS)
        source = random.choice(SOURCES)
        vendor_tier = random.choice(list(VENDOR_RATINGS.keys()))
        
        # Generate date within last 2 years
        days_ago = random.randint(0, 730)
        date = datetime.now() - timedelta(days=days_ago)
        month = date.month
        _, demand_factor = get_season(month)
        
        # Calculate price
        base_price = generate_vendor_price(
            item["base_price"], 
            item["variance"],
            vendor_tier,
            location
        )
        
        # Apply demand factor
        final_price = round(base_price * demand_factor, 2)
        
        # Generate quantity (bulk orders get discounts)
        quantity = random.choice([1, 2, 3, 5, 10, 20, 50, 100])
        if quantity >= 10:
            discount = random.uniform(0.02, 0.08)
        elif quantity >= 5:
            discount = random.uniform(0.01, 0.05)
        else:
            discount = 0
        
        unit_price = round(final_price * (1 - discount), 2)
        total_price = round(unit_price * quantity, 2)
        
        # Warranty (in years)
        warranty = random.choice([1, 2, 3, 5])
        
        # Delivery time (in days)
        delivery_days = random.randint(3, 30)
        
        # Quality rating (1-5)
        quality_rating = round(np.random.uniform(
            VENDOR_RATINGS[vendor_tier]["quality"] * 4,
            min(5, VENDOR_RATINGS[vendor_tier]["quality"] * 5)
        ), 1)
        
        row = {
            "id": f"PRD-{i+1:06d}",
            "category": category,
            "item_name": item["name"],
            "specifications": json.dumps(item["specs"]),
            "vendor": vendor,
            "vendor_tier": vendor_tier,
            "location": location,
            "date": date.strftime("%Y-%m-%d"),
            "source": source,
            "quantity": quantity,
            "unit_price": unit_price,
            "total_price": total_price,
            "discount_percent": round(discount * 100, 2),
            "warranty_years": warranty,
            "delivery_days": delivery_days,
            "quality_rating": quality_rating,
            "base_price": item["base_price"],
            "price_variance": item["variance"],
            "demand_factor": demand_factor
        }
        data.append(row)
    
    return pd.DataFrame(data)

def generate_services_dataset(num_entries=2000):
    """Generate synthetic service pricing dataset."""
    data = []
    
    for i in range(num_entries):
        category = random.choice(list(SERVICE_CATEGORIES.keys()))
        category_data = SERVICE_CATEGORIES[category]
        item = random.choice(category_data["items"])
        location = random.choice(LOCATIONS)
        source = random.choice(SOURCES)
        
        # Generate date within last 2 years
        days_ago = random.randint(0, 730)
        date = datetime.now() - timedelta(days=days_ago)
        month = date.month
        _, demand_factor = get_season(month)
        
        # Service provider rating
        provider_rating = random.choice(["A", "B", "C"])
        rating_factor = {"A": 1.10, "B": 1.00, "C": 0.90}[provider_rating]
        
        # Calculate price
        base_price = item["base_price"] * (1 + np.random.uniform(-item["variance"], item["variance"]))
        price = base_price * rating_factor * demand_factor
        
        # Location adjustment
        metro_cities = ["Delmi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
        if location in metro_cities:
            price *= np.random.uniform(1.05, 1.15)
        
        final_price = round(price, 2)
        
        # Duration (in months)
        duration = random.choice([1, 3, 6, 12])
        
        # Service quality metrics
        sla_compliance = round(np.random.uniform(0.85, 0.99), 3)
        customer_rating = round(np.random.uniform(3.0, 5.0), 1)
        
        row = {
            "id": f"SRV-{i+1:06d}",
            "category": category,
            "service_name": item["name"],
            "specifications": json.dumps(item["specs"]),
            "location": location,
            "date": date.strftime("%Y-%m-%d"),
            "source": source,
            "duration_months": duration,
            "monthly_price": final_price,
            "total_price": round(final_price * duration, 2),
            "provider_rating": provider_rating,
            "sla_compliance": sla_compliance,
            "customer_rating": customer_rating,
            "base_price": item["base_price"],
            "price_variance": item["variance"],
            "demand_factor": demand_factor
        }
        data.append(row)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Generating procurement price benchmarking datasets...")
    
    # Generate datasets
    products_df = generate_products_dataset(5000)
    services_df = generate_services_dataset(2000)
    
    # Save to CSV
    products_df.to_csv("products_prices.csv", index=False)
    services_df.to_csv("services_prices.csv", index=False)
    
    # Create combined dataset for ML
    combined_df = pd.concat([products_df, services_df], ignore_index=True)
    combined_df.to_csv("procurement_prices.csv", index=False)
    
    print(f"Products dataset: {len(products_df)} entries saved to products_prices.csv")
    print(f"Services dataset: {len(services_df)} entries saved to services_prices.csv")
    print(f"Combined dataset: {len(combined_df)} entries saved to procurement_prices.csv")
    
    # Print summary statistics
    print("\n=== Dataset Summary ===")
    print(f"\nProducts by Category:")
    print(products_df['category'].value_counts())
    print(f"\nServices by Category:")
    print(services_df['category'].value_counts())
    print(f"\nLocations:")
    print(products_df['location'].value_counts())
    print(f"\nPrice Range (Products): ₹{products_df['unit_price'].min():,.2f} - ₹{products_df['unit_price'].max():,.2f}")
    print(f"Price Range (Services): ₹{services_df['monthly_price'].min():,.2f} - ₹{services_df['monthly_price'].max():,.2f}")
