import pandas as pd

DATA_FILE = "traffic_analytics.csv"


def build_dataframe():
    data = {
        "vehicle_count": [100, 120, 150, 180],
        "avg_speed": [60, 55, 45, 35],
        "density": [0.4, 0.6, 0.8, 0.9],
        "congestion": [0, 1, 1, 2],
    }
    return pd.DataFrame(data)


def save_dataframe(df, path=DATA_FILE):
    df.to_csv(path, index=False)
    return path


def main():
    df = build_dataframe()
    saved_path = save_dataframe(df)
    print(f"Saved analytics CSV to {saved_path}")
    print(df)


if __name__ == "__main__":
    main()