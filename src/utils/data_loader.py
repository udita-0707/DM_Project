import pandas as pd
import os

def load_processed_data(mock=True):
    """
    Loads the processed arXiv data. Uses mock data if the full dataset is not available.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_path = os.path.join(base_dir, 'data', 'processed', 'arxiv_processed.csv')
    mock_path = os.path.join(base_dir, 'data', 'processed', 'arxiv_processed_mock.csv')
    
    if os.path.exists(processed_path) and not mock:
        print(f"Loading full processed data from: {processed_path}")
        return pd.read_csv(processed_path)
    elif os.path.exists(mock_path):
        print(f"Loading mock data from: {mock_path}")
        return pd.read_csv(mock_path)
    else:
        raise FileNotFoundError("Processed data file not found. Please run the data acquisition script or ensure mock data is present.")

if __name__ == "__main__":
    # Example usage
    try:
        df = load_processed_data(mock=True)
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(df.head())
    except FileNotFoundError as e:
        print(e)

