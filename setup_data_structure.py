"""Create the data directory structure for AQI Predictor.

Usage:
    python setup_data_structure.py
"""

from pathlib import Path


DATA_DIRECTORIES = [
    "data/artifacts",
    "data/metadata",
    "data/predictions/hourly/us_paro_hourly/prod",
    "data/predictions/hourly/us_paro_hourly/test",
    "data/processed/hourly/us_paro_hourly",
    "data/processed/pred_ingestion/us_paro",
    "data/raw/dynamic",
    "data/raw/pred_ingestion/us_paro",
    "data/raw/static/daily/aqi",
    "data/raw/static/daily/weather",
    "data/raw/static/hourly",
]


def create_data_structure() -> None:
    """Create the data directory structure."""

    print("Creating data directory structure...\n")

    for directory in DATA_DIRECTORIES:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ {directory}")

    print("\n✓ Data directory structure created successfully!")


if __name__ == "__main__":
    create_data_structure()
