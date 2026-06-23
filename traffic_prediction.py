import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

DATA_FILE = "traffic_analytics.csv"


def load_dataset(path=DATA_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Run analytics_csv.py first to generate it."
        )
    return pd.read_csv(path)


def train_and_evaluate(df):
    X = df[["vehicle_count", "avg_speed", "density"]]
    y = df["congestion"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Actual:", y_test.values)
    print("Predicted:", predictions)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Classification report:")
    print(classification_report(y_test, predictions))


def main():
    parser = argparse.ArgumentParser(
        description="Train a traffic congestion prediction model."
    )
    parser.add_argument(
        "--data",
        default=DATA_FILE,
        help="Path to the traffic analytics CSV file.",
    )
    args = parser.parse_args()

    df = load_dataset(args.data)
    train_and_evaluate(df)


if __name__ == "__main__":
    main()