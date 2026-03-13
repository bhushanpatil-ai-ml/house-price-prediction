import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from data_preprocessing import load_data, preprocess_data


def main():
    model = joblib.load("../models/best_house_price_model.pkl")

    df = load_data("../data/housing.csv")
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    print("Final Evaluation")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")


if __name__ == "__main__":
    main()