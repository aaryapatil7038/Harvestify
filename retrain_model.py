import argparse
from pathlib import Path

from train_model import DATASET_PATH, train_and_save
from utils.crop_features import ensure_context_columns


def main():
    parser = argparse.ArgumentParser(
        description="Retrain the Harvestify crop model from an uploaded or staged CSV file."
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to the uploaded CSV that should be used for retraining.",
    )
    parser.add_argument(
        "--promote-to-default-dataset",
        action="store_true",
        help="Also overwrite the default project dataset with the sanitized uploaded CSV.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    results = train_and_save(dataset_path=dataset_path)

    if args.promote_to_default_dataset:
        import pandas as pd

        dataset = pd.read_csv(dataset_path)
        dataset = ensure_context_columns(dataset)
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(DATASET_PATH, index=False)
        print("Promoted dataset to:", DATASET_PATH)

    print("Retraining complete:", results)


if __name__ == "__main__":
    main()
