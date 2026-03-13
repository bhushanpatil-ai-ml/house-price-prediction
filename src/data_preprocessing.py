import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def preprocess_data(df: pd.DataFrame):
    # Drop unnecessary columns if present
    columns_to_drop = ["id", "date"]
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    if existing_cols:
        df = df.drop(columns=existing_cols)

    # Remove missing values
    df = df.dropna()

    # Separate features and target
    X = df.drop("price", axis=1)
    y = df["price"]

    return X, y

    

    