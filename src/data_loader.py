from pathlib import Path

from ucimlrepo import fetch_ucirepo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching UCI Student Performance dataset...")

    dataset = fetch_ucirepo(id=320)

    features = dataset.data.features
    targets = dataset.data.targets

    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")

    features.to_csv(RAW_DATA_DIR / "student_performance_features.csv", index=False)
    targets.to_csv(RAW_DATA_DIR / "student_performance_targets.csv", index=False)

    print("\nDataset saved successfully.")
    print(f"Raw data directory: {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()
