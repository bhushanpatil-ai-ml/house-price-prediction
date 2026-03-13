import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from data_preprocessing import load_data, preprocess_data


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    print(f"\n{name}")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print("-" * 40)

    return model, r2, mae, rmse


def save_feature_importance(best_model, feature_names):
    if hasattr(best_model, "feature_importances_"):
        os.makedirs("../outputs", exist_ok=True)

        feature_importance = pd.Series(
            best_model.feature_importances_,
            index=feature_names
        ).sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        feature_importance.head(10).plot(kind="barh")
        plt.title("Top 10 Important Features")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("../outputs/feature_importance.png")
        plt.close()

        print("Feature importance plot saved to ../outputs/feature_importance.png")


def main():
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../outputs", exist_ok=True)

    df = load_data("../data/housing.csv")
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            objective="reg:squarederror"
        )
    }

    best_model = None
    best_score = float("-inf")
    best_name = ""

    for name, model in models.items():
        trained_model, r2, mae, rmse = evaluate_model(
            name, model, X_train, X_test, y_train, y_test
        )

        if r2 > best_score:
            best_score = r2
            best_model = trained_model
            best_name = name

    print(f"\nBest Model: {best_name}")
    print(f"Best R2 Score: {best_score:.4f}")

    joblib.dump(best_model, "../models/best_house_price_model.pkl")
    print("Best model saved to ../models/best_house_price_model.pkl")

    save_feature_importance(best_model, X.columns)


if __name__ == "__main__":
    main()