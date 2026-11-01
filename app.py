
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file,
)

# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)
app.secret_key = "estateiq-local-development-key"

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "house_model.pkl"
TEST_PREDICTIONS_PATH = PROJECT_DIR / "test_predictions.csv"
TEST_RESULTS_PATH = PROJECT_DIR / "final_test_results.csv"
VALIDATION_RESULTS_PATH = PROJECT_DIR / "validation_results.csv"
HISTORY_PATH = PROJECT_DIR / "prediction_history.csv"

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


# --------------------------------------------------
# 2. LOAD THE TRAINED MODEL
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "house_model.pkl not found. "
        "Run train_model.py before starting Flask."
    )

artifact = joblib.load(MODEL_PATH)

# Supports the dictionary format created by train_model.py.
if isinstance(artifact, dict) and "model" in artifact:
    model = artifact["model"]
    model_name = artifact.get("model_name", "Trained Model")
    model_features = artifact.get("features", FEATURES)
else:
    # Compatibility with a file containing only a model object.
    model = artifact
    model_name = "Trained Model"
    model_features = FEATURES


# --------------------------------------------------
# 3. HELPER FUNCTIONS
# --------------------------------------------------

def load_csv(path):
    """Load a CSV file if it exists; otherwise return an empty DataFrame."""
    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


def load_analysis_data():
    """Read saved test predictions for the dashboard charts."""

    df = load_csv(TEST_PREDICTIONS_PATH)

    required_columns = [
        *FEATURES,
        "ActualValueUSD",
        "PredictedValueUSD",
    ]

    if df.empty:
        return {
            "actual_values": [],
            "predicted_values": [],
            "map_longitudes": [],
            "map_latitudes": [],
            "map_values": [],
        }

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "test_predictions.csv is missing columns: "
            + ", ".join(missing)
        )

    # Sample only for visualization; the original CSV remains unchanged.
    chart_df = df.sample(
        n=min(500, len(df)),
        random_state=42,
    )

    map_df = df.sample(
        n=min(250, len(df)),
        random_state=42,
    )

    return {
        "actual_values": chart_df["ActualValueUSD"].round(2).tolist(),
        "predicted_values": chart_df["PredictedValueUSD"].round(2).tolist(),
        "map_longitudes": map_df["Longitude"].tolist(),
        "map_latitudes": map_df["Latitude"].tolist(),
        "map_values": map_df["ActualValueUSD"].round(2).tolist(),
    }


def get_test_metrics():
    """Read the saved final test metrics without recalculating them."""

    results = load_csv(TEST_RESULTS_PATH)

    if results.empty:
        return {
            "mae": None,
            "mse": None,
            "rmse": None,
            "r2": None,
        }

    row = results.iloc[0]

    # Metrics in final_test_results.csv are in the original
    # California Housing target units, except R2.
    return {
        "mae": float(row["MAE"]) * 100000,
        "mse": float(row["MSE"]) * (100000 ** 2),
        "rmse": float(row["RMSE"]) * 100000,
        "r2": float(row["R2"]),
    }


def get_validation_results():
    """Return model comparison results in dollars."""

    results = load_csv(VALIDATION_RESULTS_PATH)

    if results.empty:
        return []

    output = []

    for _, row in results.iterrows():
        output.append({
            "name": str(row["Model"]),
            "mae": float(row["Validation_MAE"]) * 100000,
            "rmse": float(row["Validation_RMSE"]) * 100000,
            "r2": float(row["Validation_R2"]),
        })

    return output



def load_history():
    """Load prediction history and normalize older CSV formats."""

    history = load_csv(HISTORY_PATH)

    if history.empty:
        return []

    # Support history files created by earlier versions of the app.
    aliases = {
        "PredictedPriceUSD": [
            "PredictedPriceUSD",
            "PredictedPrice",
            "prediction",
            "predicted_price",
            "Prediction",
        ],
        "Timestamp": [
            "Timestamp",
            "timestamp",
            "Date",
            "date",
        ],
    }

    for standard_name, possible_names in aliases.items():
        if standard_name not in history.columns:
            for old_name in possible_names:
                if old_name in history.columns:
                    history = history.rename(
                        columns={old_name: standard_name}
                    )
                    break

    # If an old record has no prediction column, show a safe default.
    if "PredictedPriceUSD" not in history.columns:
        history["PredictedPriceUSD"] = 0.0

    if "Timestamp" not in history.columns:
        history["Timestamp"] = ""

    return history.to_dict(orient="records")



def save_history(record):
    """Append a prediction to the history CSV."""

    new_row = pd.DataFrame([record])

    if HISTORY_PATH.exists():
        new_row.to_csv(
            HISTORY_PATH,
            mode="a",
            header=False,
            index=False,
        )
    else:
        new_row.to_csv(
            HISTORY_PATH,
            index=False,
        )


# --------------------------------------------------
# 4. HOME PAGE
# --------------------------------------------------

@app.route("/")
def index():
    analysis = load_analysis_data()
    metrics = get_test_metrics()
    comparisons = get_validation_results()
    history = load_history()

    return render_template(
        "index.html",
        model_name=model_name,
        features=model_features,
        metrics=metrics,
        comparisons=comparisons,
        history=history,
        actual_values=analysis["actual_values"],
        predicted_values=analysis["predicted_values"],
        map_longitudes=analysis["map_longitudes"],
        map_latitudes=analysis["map_latitudes"],
        map_values=analysis["map_values"],
        history_count=len(history),
    )


# --------------------------------------------------
# 5. HOUSE-PRICE PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = {}

        for feature in FEATURES:
            raw_value = request.form.get(feature, "").strip()

            if not raw_value:
                raise ValueError(
                    f"Please enter a value for {feature}."
                )

            values[feature] = float(raw_value)

        # Basic input validation.
        if values["MedInc"] < 0:
            raise ValueError("Median income cannot be negative.")

        if values["HouseAge"] < 0:
            raise ValueError("House age cannot be negative.")

        if values["AveRooms"] <= 0:
            raise ValueError("Average rooms must be greater than zero.")

        if values["AveBedrms"] <= 0:
            raise ValueError(
                "Average bedrooms must be greater than zero."
            )

        if values["Population"] < 0:
            raise ValueError("Population cannot be negative.")

        if values["AveOccup"] <= 0:
            raise ValueError(
                "Average occupancy must be greater than zero."
            )

        if not 32 <= values["Latitude"] <= 42.5:
            raise ValueError(
                "Latitude must be within the California region."
            )

        if not -125 <= values["Longitude"] <= -113:
            raise ValueError(
                "Longitude must be within the California region."
            )

        input_df = pd.DataFrame(
            [[values[feature] for feature in model_features]],
            columns=model_features,
        )

        # Model output is in units of $100,000.
        prediction_usd = float(model.predict(input_df)[0]) * 100000

        record = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **values,
            "PredictedPriceUSD": round(prediction_usd, 2),
        }

        save_history(record)

        return render_template(
            "index.html",
            model_name=model_name,
            features=model_features,
            metrics=get_test_metrics(),
            comparisons=get_validation_results(),
            history=load_history(),
            history_count=len(load_history()),
            actual_values=load_analysis_data()["actual_values"],
            predicted_values=load_analysis_data()["predicted_values"],
            map_longitudes=load_analysis_data()["map_longitudes"],
            map_latitudes=load_analysis_data()["map_latitudes"],
            map_values=load_analysis_data()["map_values"],
            prediction=prediction_usd,
            submitted_values=values,
        )

    except (ValueError, TypeError) as error:
        flash(str(error), "error")
        return redirect(url_for("index"))


# --------------------------------------------------
# 6. DOWNLOAD PREDICTION HISTORY
# --------------------------------------------------

@app.route("/download-history")
def download_history():
    if not HISTORY_PATH.exists():
        flash("There is no prediction history to download yet.", "error")
        return redirect(url_for("index"))

    return send_file(
        HISTORY_PATH,
        as_attachment=True,
        download_name="estateiq_prediction_history.csv",
        mimetype="text/csv",
    )


# --------------------------------------------------
# 7. RUN THE APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
