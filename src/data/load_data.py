import pandas as pd

def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw Titanic data."""
    return pd.read_csv(path)

if __name__ == "__main__":
    train = load_raw_data("data/raw/train.csv")
    print(f"Train data shape: {train.shape}")
