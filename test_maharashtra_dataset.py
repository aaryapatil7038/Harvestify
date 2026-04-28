import unittest
from pathlib import Path
import shutil
import uuid

import pandas as pd

from utils.crop_features import CANONICAL_CROP_LABELS, TARGET_COLUMN
from utils.maharashtra_dataset import save_balanced_maharashtra_dataset, standardize_source_dataset


def make_row(crop_name, index):
    return {
        "N": 40 + index,
        "P": 20 + (index % 7),
        "K": 25 + (index % 5),
        "temperature": 24.5 + (index % 3),
        "humidity": 55.0 + (index % 11),
        "ph": 6.4 + ((index % 4) * 0.1),
        "rainfall": 120.0 + (index * 3),
        TARGET_COLUMN: crop_name,
    }


class MaharashtraDatasetTests(unittest.TestCase):
    def test_standardize_source_dataset_normalizes_aliases_and_context(self):
        source_df = pd.DataFrame(
            [
                {
                    "N": 80,
                    "P": 40,
                    "K": 35,
                    "temprature": 27.8,
                    "humidity": 61.2,
                    "ph": 6.8,
                    "rainfall": 220.0,
                    "crop_name": "Black Gram",
                },
                {
                    "N": 82,
                    "P": 38,
                    "K": 34,
                    "temprature": 28.0,
                    "humidity": 60.0,
                    "ph": 6.7,
                    "rainfall": 210.0,
                    "crop_name": "Cowpea",
                },
            ]
        )

        standardized = standardize_source_dataset(source_df)

        self.assertEqual(list(standardized[TARGET_COLUMN]), ["blackgram", "cowpea"])
        self.assertIn("season", standardized.columns)
        self.assertIn("water_source", standardized.columns)
        self.assertIn("water_availability", standardized.columns)
        self.assertTrue((standardized["region"] == "Maharashtra").all())

    def test_save_balanced_maharashtra_dataset_uses_local_sources(self):
        temp_path = Path(__file__).resolve().parent / "test_artifacts" / ("maharashtra_dataset_" + uuid.uuid4().hex)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            source_path = temp_path / "source.csv"
            output_path = temp_path / "merged.csv"

            rows = []
            for idx, crop_name in enumerate(CANONICAL_CROP_LABELS):
                rows.append(make_row(crop_name, idx * 2))
                rows.append(make_row(crop_name, (idx * 2) + 1))
            pd.DataFrame(rows).to_csv(source_path, index=False)

            dataset, validation = save_balanced_maharashtra_dataset(
                output_path=output_path,
                rows_per_crop=2,
                seed=42,
                source_dataset_paths=[source_path],
            )

            counts = dataset[TARGET_COLUMN].value_counts()
            self.assertEqual(len(dataset), len(CANONICAL_CROP_LABELS) * 2)
            self.assertTrue((counts == 2).all())
            self.assertEqual(validation["rows_per_crop_target"], 2)
            self.assertEqual(validation["source_dataset_paths"], [str(source_path)])
            self.assertTrue(output_path.exists())
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
