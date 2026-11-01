
from pathlib import Path

import joblib
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "house_model.pkl"

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

TARGET_UNIT = "USD"

print("=" * 65)
print("ESTATEIQ - CALIFORNIA HOUSING MODEL TRAINING")
print("=" * 65)


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("\nLoading California Housing dataset...")

housing = fetch_california_housing(as_frame=True)

X = housing.data[FEATURES].copy()
y = housing.target.copy()

print(f"Total rows: {len(X):,}")
print(f"Number of features: {len(FEATURES)}")


# --------------------------------------------------
# 3. TRAIN / VALIDATION / TEST SPLIT
#    60% training, 20% validation, 20% testing
# --------------------------------------------------

X_train, X_remaining, y_train, y_remaining = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_remaining,
    y_remaining,
    test_size=0.50,
    random_state=42,
)

print("\nDataset split:")
print(f"Training:   {len(X_train):,} rows")
print(f"Validation: {len(X_validation):,} rows")
print(f"Testing:    {len(X_test):,} rows")


# --------------------------------------------------
# 4. DEFINE MODELS
# --------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Ridge": Ridge(alpha=1.0),

    "Lasso": Lasso(
        alpha=0.001,
        max_iter=10000,
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    ),
}


# --------------------------------------------------
# 5. TRAIN AND COMPARE MODELS
# --------------------------------------------------

validation_results = []
trained_models = {}

print("\nTraining models...")

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    validation_predictions = model.predict(X_validation)

    mae = mean_absolute_error(
        y_validation,
        validation_predictions,
    )

    mse = mean_squared_error(
        y_validation,
        validation_predictions,
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_validation,
        validation_predictions,
    )

    validation_results.append({
        "Model": name,
        "Validation_MAE": mae,
        "Validation_MSE": mse,
        "Validation_RMSE": rmse,
        "Validation_R2": r2,
    })

    trained_models[name] = model

    print(
        f"Validation MAE: ${mae * 100000:,.2f} | "
        f"RMSE: ${rmse * 100000:,.2f} | "
        f"R2: {r2:.4f}"
    )


# Save model comparison results.
validation_df = pd.DataFrame(validation_results)
validation_df = validation_df.sort_values(
    by="Validation_MAE",
    ascending=True,
)

validation_df.to_csv(
    PROJECT_DIR / "validation_results.csv",
    index=False,
)

# Choose the model with the lowest validation MAE.
best_model_name = validation_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\n" + "=" * 65)
print(f"SELECTED MODEL: {best_model_name}")
print("=" * 65)


# --------------------------------------------------
# 6. EVALUATE ON THE TEST SET
# --------------------------------------------------

# The test set is not used to select the model.
y_pred = best_model.predict(X_test)

test_mae = mean_absolute_error(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)
test_rmse = test_mse ** 0.5
test_r2 = r2_score(y_test, y_pred)

test_results = pd.DataFrame([{
    "Model": best_model_name,
    "MAE": test_mae,
    "MSE": test_mse,
    "RMSE": test_rmse,
    "R2": test_r2,
}])

test_results.to_csv(
    PROJECT_DIR / "final_test_results.csv",
    index=False,
)

print("\nFinal test-set performance:")
print(f"MAE:  ${test_mae * 100000:,.2f}")
print(f"MSE:  {test_mse * (100000 ** 2):,.2f} USD²")
print(f"RMSE: ${test_rmse * 100000:,.2f}")
print(f"R²:   {test_r2:.4f} ({test_r2 * 100:.2f}%)")


# --------------------------------------------------
# 7. SAVE THE MODEL
# --------------------------------------------------

model_artifact = {
    "model": best_model,
    "features": FEATURES,
    "target_unit": TARGET_UNIT,
    "dataset": "California Housing",
    "model_name": best_model_name,
}

joblib.dump(model_artifact, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")


# --------------------------------------------------
# 8. SAVE ACTUAL TEST-SET PREDICTIONS
# --------------------------------------------------

# Keep the original feature values for the geographic chart.
test_predictions = X_test.copy()

# California Housing target values are in units of $100,000.
test_predictions["ActualValueUSD"] = (
    y_test.to_numpy() * 100000
)

test_predictions["PredictedValueUSD"] = (
    y_pred * 100000
)

test_predictions.to_csv(
    PROJECT_DIR / "test_predictions.csv",
    index=False,
)

print(
    "Test predictions saved to: "
    f"{PROJECT_DIR / 'test_predictions.csv'}"
)


# --------------------------------------------------
# 9. FINISHED
# --------------------------------------------------

print("\nGenerated files:")
print("  house_model.pkl")
print("  validation_results.csv")
print("  final_test_results.csv")
print("  test_predictions.csv")

print("\nTraining completed successfully.")
