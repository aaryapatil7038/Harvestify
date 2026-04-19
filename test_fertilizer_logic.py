import unittest

from utils.fertilizer_logic import (
    BALANCED_FERTILIZER_KEY,
    build_fertilizer_plan,
    build_fertilizer_reference_df,
    get_fertilizer_recommendation_key,
    get_fertilizer_recommendation_text,
)


class FertilizerLogicTests(unittest.TestCase):
    def test_reference_table_contains_all_30_merged_maharashtra_crops(self):
        reference_df = build_fertilizer_reference_df(
            "Data/fertilizer.csv",
            "Data/crop_recommendation_merged_maharashtra.csv"
        )

        self.assertEqual(reference_df["Crop"].nunique(), 30)
        self.assertFalse(reference_df[["N", "P", "K"]].isna().any().any())

    def test_exact_reference_match_returns_balanced_key_for_every_crop(self):
        reference_df = build_fertilizer_reference_df(
            "Data/fertilizer.csv",
            "Data/crop_recommendation_merged_maharashtra.csv"
        )

        for row in reference_df.itertuples(index=False):
            recommendation_key = get_fertilizer_recommendation_key(
                reference_n=row.N,
                reference_p=row.P,
                reference_k=row.K,
                actual_n=row.N,
                actual_p=row.P,
                actual_k=row.K
            )
            self.assertEqual(recommendation_key, BALANCED_FERTILIZER_KEY, row.Crop)

    def test_nitrogen_high_and_low_paths_return_correct_text(self):
        high_text = get_fertilizer_recommendation_text("NHigh", "en")
        low_text = get_fertilizer_recommendation_text("Nlow", "en")

        self.assertIn("higher than the crop's ideal range", high_text)
        self.assertIn("Reduce or pause high-nitrogen fertilizers", high_text)
        self.assertIn("lower than the crop's ideal range", low_text)
        self.assertIn("Apply a nitrogen-rich fertilizer", low_text)

    def test_largest_absolute_gap_controls_recommendation(self):
        recommendation_key = get_fertilizer_recommendation_key(
            reference_n=80,
            reference_p=40,
            reference_k=30,
            actual_n=60,
            actual_p=75,
            actual_k=25
        )

        self.assertEqual(recommendation_key, "PHigh")

    def test_balanced_plan_contains_stagewise_schedule(self):
        plan = build_fertilizer_plan(
            crop_name="rice",
            reference_n=80,
            reference_p=40,
            reference_k=40,
            actual_n=80,
            actual_p=40,
            actual_k=40
        )

        self.assertEqual(plan["recommendation_key"], BALANCED_FERTILIZER_KEY)
        self.assertTrue(plan["basal_lines"])
        self.assertTrue(plan["vegetative_lines"])
        self.assertTrue(plan["flowering_lines"])
        self.assertTrue(plan["quantity_lines"])
        self.assertTrue(plan["organic_lines"])
        self.assertIn("maintenance schedule", plan["summary"].lower())

    def test_low_nitrogen_plan_recommends_stagewise_urea_support(self):
        plan = build_fertilizer_plan(
            crop_name="tomato",
            reference_n=101,
            reference_p=45,
            reference_k=55,
            actual_n=70,
            actual_p=45,
            actual_k=55
        )

        basal_text = " ".join(plan["basal_lines"])
        vegetative_text = " ".join(plan["vegetative_lines"])
        quantity_text = " ".join(plan["quantity_lines"])

        self.assertEqual(plan["recommendation_key"], "Nlow")
        self.assertIn("Urea", basal_text)
        self.assertIn("Top-dress urea", vegetative_text)
        self.assertIn("Urea total", quantity_text)

    def test_marathi_plan_localizes_generated_sections(self):
        plan = build_fertilizer_plan(
            crop_name="jowar",
            reference_n=60,
            reference_p=33,
            reference_k=33,
            actual_n=45,
            actual_p=45,
            actual_k=45,
            lang="mr"
        )

        self.assertEqual(plan["crop_name"], "ज्वारी")
        self.assertIn("मुख्य अन्नद्रव्य मुद्दा", plan["summary"])
        self.assertTrue(any("माती" in line for line in plan["status_lines"]))
        self.assertTrue(any("युरिया" in line for line in plan["basal_lines"] + plan["vegetative_lines"] + plan["flowering_lines"] + plan["quantity_lines"]))
        self.assertIn("अंदाजित योजना", plan["note"])


if __name__ == "__main__":
    unittest.main()
