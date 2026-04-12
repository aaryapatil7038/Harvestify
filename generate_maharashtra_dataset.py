import argparse
from pathlib import Path

from utils.maharashtra_dataset import (
    ROWS_PER_CROP_DEFAULT,
    save_balanced_maharashtra_dataset,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = BASE_DIR / "Data" / "crop_recommendation.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Generate a balanced Maharashtra-specific crop recommendation dataset."
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to the CSV file that should be written."
    )
    parser.add_argument(
        "--rows-per-crop",
        type=int,
        default=ROWS_PER_CROP_DEFAULT,
        help="Number of rows to generate for each crop label."
    )
    parser.add_argument(
        "--existing-dataset-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to the current dataset that should be merged with the 15 newly added crops.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation."
    )
    args = parser.parse_args()

    dataset, validation = save_balanced_maharashtra_dataset(
        output_path=args.output_path,
        rows_per_crop=args.rows_per_crop,
        seed=args.seed,
        existing_dataset_path=args.existing_dataset_path,
    )

    print("Dataset written to:", args.output_path)
    print("Shape:", dataset.shape)
    print("Missing values:", validation["missing_values"])
    print("Duplicate rows:", validation["duplicate_rows"])
    print("Class distribution:", dict(sorted(validation["class_distribution"].items())))
    print("Season distribution:", dict(sorted(validation["season_distribution"].items())))


if __name__ == "__main__":
    main()
