"""
Python script to create the data directory structure for AQI Predictor
Usage: python setup_data_structure.py
"""

from pathlib import Path

def create_data_structure():
    """Create the data directory structure."""
    
    print("Creating data directory structure...\n")
    
    directories = [
        "data/artifacts",
        "data/plots/daily",
        "data/plots/hourly",
        "data/predictions/daily",
        "data/predictions/hourly",
        "data/processed/daily",
        "data/processed/hourly",
        "data/raw/dynamic",
        "data/raw/static/daily/aqi",
        "data/raw/static/daily/weather",
        "data/raw/static/hourly",
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ {directory}")
    
    print("\n✓ Data directory structure created successfully!")


if __name__ == "__main__":
    create_data_structure()
