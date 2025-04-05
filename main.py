import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, r2_score
from flask import Flask, request, jsonify

def load_and_preprocess_data(filepath):
    """ Load the dataset, handle missing values, and preprocess it."""
    df = pd.read_csv(filepath)
    
    # Strip whitespace and print columns
    df.columns = df.columns.str.strip()
    print("Dataset columns:", df.columns.tolist())

    # Fill missing values with the median
    df.fillna(df.median(), inplace=True)

    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Rename columns to match expected format
    df.rename(columns={"At Risk (Binary)": "At Risk"}, inplace=True)

    # Ensure target columns exist
    expected_columns = ["At Risk", "Stroke Risk (%)"]
    missing_columns = [col for col in expected_columns if col not in df.columns]

    if missing_columns:
        raise KeyError(f"Missing columns in dataset: {missing_columns}")

    # Separate features and targets
    X = df.drop(expected_columns, axis=1)
    y_classification = df["At Risk"]
    y_regression = df["Stroke Risk (%)"]

    return X, y_classification, y_regression


def train_models(X, y_classification, y_regression):
    """ Train classification and regression models."""
    X_train, X_test, y_train, y_test = train_test_split(X, y_classification, test_size=0.2, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train_reg = scaler.transform(X_train_reg)  # Use the same scaler for consistency
    X_test_reg = scaler.transform(X_test_reg)
    
    # Train classification model
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    # Train regression model
    reg = RandomForestRegressor(random_state=42)
    reg.fit(X_train_reg, y_train_reg)
    
    # Save models and scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/stroke_risk_classifier.pkl")
    joblib.dump(reg, "models/stroke_risk_regressor.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    
    return clf, reg, scaler, X_test, y_test, X_test_reg, y_test_reg

def evaluate_models(clf, reg, X_test, y_test, X_test_reg, y_test_reg):
    """ Evaluate model performance."""
    # Classification evaluation
    y_pred = clf.predict(X_test)
    print("Classification Metrics:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # Regression evaluation
    y_pred_reg = reg.predict(X_test_reg)
    print("\nRegression Metrics:")
    print("Mean Absolute Error:", mean_absolute_error(y_test_reg, y_pred_reg))
    print("R2 Score:", r2_score(y_test_reg, y_pred_reg))

def create_flask_app(clf, reg, scaler, feature_names):
    """ Create a Flask API for real-time predictions."""
    app = Flask(__name__)
    
    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.json
        
        # Extract and preprocess features
        input_data = [data[feature] for feature in feature_names]
        input_data_scaled = scaler.transform([input_data])
        
        # Make predictions
        at_risk = clf.predict(input_data_scaled)[0]
        stroke_risk_percentage = reg.predict(input_data_scaled)[0]
        
        return jsonify({
            "At Risk": int(at_risk),
            "Stroke Risk (%)": float(stroke_risk_percentage)
        })
    
    return app

if __name__ == "__main__":
    filepath = "stroke_risk_dataset.csv"  # Update this path
    X, y_classification, y_regression = load_and_preprocess_data(filepath)
    
    # Train models
    clf, reg, scaler, X_test, y_test, X_test_reg, y_test_reg = train_models(X, y_classification, y_regression)
    
    # Evaluate models
    evaluate_models(clf, reg, X_test, y_test, X_test_reg, y_test_reg)
    
    # Run Flask app
    feature_names = X.columns.tolist()
    app = create_flask_app(clf, reg, scaler, feature_names)
    app.run(debug=True)
