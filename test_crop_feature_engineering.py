import unittest

import pandas as pd

from utils.crop_features import (
    FEATURE_COLUMNS,
    INPUT_FEATURE_COLUMNS,
    encode_feature_frame,
)


class CropFeatureEngineeringTests(unittest.TestCase):
    def test_encode_feature_frame_appends_expected_derived_features(self):
        raw_df = pd.DataFrame(
            [[80, 40, 20, 30.0, 60.0, 6.8, 200.0, "Kharif", "canal", "high"]],
            columns=INPUT_FEATURE_COLUMNS,
        )

        encoded_df = encode_feature_frame(raw_df)

        self.assertEqual(list(encoded_df.columns), FEATURE_COLUMNS)
        row = encoded_df.iloc[0]
        self.assertAlmostEqual(row["nutrient_total"], 140.0)
        self.assertAlmostEqual(row["np_ratio"], 80.0 / 41.0, places=6)
        self.assertAlmostEqual(row["pk_ratio"], 40.0 / 21.0, places=6)
        self.assertAlmostEqual(row["temp_humidity_index"], 18.0)
        self.assertGreater(row["rainfall_log"], 5.3)
        self.assertAlmostEqual(row["ph_distance_neutral"], 0.3)
        self.assertAlmostEqual(row["water_access_score"], 6.0)
        self.assertAlmostEqual(row["season_water_pressure"], -3.0)

    def test_encode_feature_frame_keeps_input_contract_raw_columns_only(self):
        raw_df = pd.DataFrame(
            [[35, 25, 22, 24.0, 55.0, 7.1, 120.0, "Rabi", "borewell", "medium"]],
            columns=INPUT_FEATURE_COLUMNS,
        )

        encoded_df = encode_feature_frame(raw_df)

        self.assertEqual(raw_df.columns.tolist(), INPUT_FEATURE_COLUMNS)
        self.assertTrue(set(INPUT_FEATURE_COLUMNS).issubset(set(encoded_df.columns)))
        self.assertGreater(len(encoded_df.columns), len(raw_df.columns))


if __name__ == "__main__":
    unittest.main()
