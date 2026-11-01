# EstateIQ — California Housing Price Prediction

EstateIQ is a machine learning web application that estimates district-level median housing values in California using regression models and a Flask-based web interface.

## Features

- Predict estimated housing values from eight input features.
- Train and compare five regression algorithms.
- Evaluate models using MAE, MSE, RMSE, and R².
- Visualize model performance using interactive charts.
- View prediction history and download prediction records as CSV.

## Tech Stack

- Python
- Flask
- scikit-learn
- Pandas and NumPy
- HTML, CSS, and JavaScript
- Chart.js

## Dataset

This project uses the California Housing dataset available through scikit-learn. It contains 20,640 district-level observations and eight numerical features, including median income, housing age, room statistics, population, occupancy, latitude, and longitude.

The target represents historical district-level median house values. Predictions are estimates, not verified current market prices for individual properties.

## Models

The training workflow compares:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Random Forest Regression
5. Gradient Boosting Regression

The selected model is evaluated on held-out test data.

## Evaluation Metrics

- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of Determination

See `validation_results.csv` and `final_test_results.csv` for recorded results.

## Run Locally

Clone the repository:

```bash
git clone https://github.com/Gaurav-coder-tech/ml-house-predictor.git
cd ml-house-predictor
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
python app.py
```

Open the local URL printed by Flask, usually `http://127.0.0.1:5000`.

## Project Structure

```text
ml-house-predictor/
├── app.py
├── train_model.py
├── house_model.pkl
├── requirements.txt
├── validation_results.csv
├── final_test_results.csv
├── test_predictions.csv
└── templates/
    └── index.html
```

## Limitations

- The dataset represents historical housing values.
- Predictions are district-level estimates, not exact individual property valuations.
- Historical test performance does not guarantee accuracy in today's housing market.
- Prediction history is currently CSV-based and is not yet isolated by user account.

## Future Improvements

- Deploy the application online.
- Add database-backed prediction history and optional user accounts.
- Evaluate performance using more recent housing data.
- Add automated tests and deployment configuration.
