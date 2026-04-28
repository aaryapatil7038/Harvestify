import unittest

from train_model import XGBClassifier, build_model_candidates


class TrainModelSearchCandidateTests(unittest.TestCase):
    def test_build_model_candidates_uses_unique_names(self):
        candidates = build_model_candidates(num_classes=30, xgb_search_rounds=4)

        names = [candidate["name"] for candidate in candidates]
        self.assertEqual(len(names), len(set(names)))

    def test_build_model_candidates_appends_search_variants_when_xgboost_is_available(self):
        candidates = build_model_candidates(num_classes=30, xgb_search_rounds=4)
        search_candidates = [
            candidate
            for candidate in candidates
            if candidate["name"].startswith("xgboost_search_")
        ]

        if XGBClassifier is None:
            self.assertEqual(search_candidates, [])
        else:
            self.assertGreaterEqual(len(search_candidates), 1)


if __name__ == "__main__":
    unittest.main()
