import importlib
import os
import unittest
from unittest.mock import patch

import pandas as pd

from utils.crop_features import FEATURE_COLUMNS


class CropArtifactLoadingTests(unittest.TestCase):
    def test_saved_artifacts_can_generate_prediction(self):
        from xai_explainer import explain_crop_prediction

        sample_df = pd.DataFrame(
            [[86, 59, 42, 27.59, 77.34, 5.70, 205.81, "Kharif", "canal", "high"]],
            columns=FEATURE_COLUMNS
        )

        merged_df, shap_df, explanations, prediction, top3_predictions = explain_crop_prediction(sample_df)

        self.assertEqual(list(merged_df.columns), ["Feature", "Impact Value", "Effect"])
        self.assertEqual(
            list(shap_df.columns),
            ["Feature", "Impact Value", "Observed Value", "Effect", "Human Explanation"]
        )
        self.assertTrue(prediction)
        self.assertEqual(len(explanations), len(FEATURE_COLUMNS))
        self.assertEqual(len(top3_predictions), 3)
        self.assertEqual(top3_predictions[0]["crop"], prediction)

    def test_saved_artifacts_do_not_collapse_to_single_crop_for_distinct_inputs(self):
        from xai_explainer import explain_crop_prediction

        representative_samples = [
            [90, 40, 40, 28.0, 80.0, 6.5, 250.0, "Kharif", "canal", "high"],
            [20, 20, 20, 18.0, 40.0, 7.2, 40.0, "Rabi", "rainfed", "low"],
            [35, 50, 20, 24.0, 55.0, 7.8, 120.0, "Summer", "borewell", "medium"],
        ]

        predictions = []
        for sample in representative_samples:
            sample_df = pd.DataFrame([sample], columns=FEATURE_COLUMNS)
            _, _, _, prediction, top3_predictions = explain_crop_prediction(sample_df)
            predictions.append(prediction)
            self.assertEqual(top3_predictions[0]["crop"], prediction)

        self.assertGreaterEqual(len(set(predictions)), 3)
        self.assertNotEqual(predictions.count("sesame"), len(predictions))


class CropPredictRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.app_module = importlib.import_module("app")
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest("App dependencies are not installed: {0}".format(exc))

        cls.flask_app = cls.app_module.app
        cls.flask_app.testing = True

    def test_crop_predict_uses_season_reranked_top_crop_consistently(self):
        base_df = pd.DataFrame([
            {"Feature": "Rainfall", "Impact Value": 0.42, "Effect": "Positive (Helps crop)"},
            {"Feature": "Temperature", "Impact Value": 0.21, "Effect": "Positive (Helps crop)"},
            {"Feature": "Humidity", "Impact Value": 0.10, "Effect": "Positive (Helps crop)"}
        ])

        raw_top3 = [
            {"crop": "rice", "confidence": 60.0, "actual_confidence": 60.0},
            {"crop": "wheat", "confidence": 25.0, "actual_confidence": 25.0},
            {"crop": "maize", "confidence": 15.0, "actual_confidence": 15.0}
        ]
        reranked_top3 = [
            {"crop": "wheat", "confidence": 52.0, "actual_confidence": 25.0, "season_bonus": 8.0, "season_adjusted_score": 33.0},
            {"crop": "rice", "confidence": 38.0, "actual_confidence": 60.0, "season_bonus": 0.0, "season_adjusted_score": 60.0},
            {"crop": "maize", "confidence": 10.0, "actual_confidence": 15.0, "season_bonus": 0.0, "season_adjusted_score": 15.0}
        ]
        xai_runtime = {
            "explain_crop_prediction": lambda *_args, **_kwargs: (
                base_df.copy(), base_df.copy(),
                ["Rainfall positively influenced the recommendation of rice."],
                "rice",
                raw_top3
            ),
            "explain_specific_crop_prediction": lambda *_args, **_kwargs: (
                base_df.copy(), base_df.copy(),
                ["Rainfall positively influenced the recommendation of wheat."],
                "wheat",
                reranked_top3
            ),
        }

        with patch.object(self.app_module, "get_lang", return_value="en"), \
             patch.object(self.app_module, "resolve_location_and_weather", return_value={
                 "city": "Pune",
                 "temperature": 27.5,
                 "humidity": 78.0,
                 "rainfall": 210.0,
                 "latitude": 18.52,
                 "longitude": 73.85
             }), \
             patch.object(self.app_module, "get_crop_xai_runtime", return_value=xai_runtime), \
             patch.object(self.app_module, "rerank_top3_predictions_by_season", return_value=("Rabi", reranked_top3)), \
             patch.object(self.app_module, "get_msamb_live_price", return_value=None), \
             patch.object(self.app_module, "estimate_yield_and_profit", return_value=(12.5, 10000.0)), \
             patch.object(self.app_module, "generate_overall_summary", return_value="summary"), \
             patch.object(self.app_module, "get_crop_sowing_guidance", return_value={"best_sowing_window": "Now"}), \
             patch.object(self.app_module, "localize_crop_output", return_value=("wheat", reranked_top3, base_df.copy(), ["line"])), \
             patch.object(self.app_module, "render_template", side_effect=lambda template, **context: context["prediction"]):

            client = self.flask_app.test_client()
            response = client.post("/crop-predict", data={
                "nitrogen": "80",
                "phosphorous": "40",
                "pottasium": "40",
                "ph": "6.5",
                "land_area": "2",
                "land_unit": "acre",
                "water_source": "canal",
                "water_availability": "high",
                "latitude": "18.52",
                "longitude": "73.85",
                "location_method": "current"
            })

            self.assertEqual(response.status_code, 200)
            self.assertIn(b"wheat", response.data)

            with client.session_transaction() as session_data:
                self.assertEqual(session_data["last_crop_result"]["prediction_raw"], "wheat")
                self.assertEqual(session_data["last_crop_result"]["top3_predictions"][0]["crop"], "wheat")

    def test_crop_predict_rejects_invalid_land_unit(self):
        with patch.object(self.app_module, "get_lang", return_value="en"), \
             patch.object(self.app_module, "render_template", side_effect=lambda template, **context: context.get("error_message", template)):

            client = self.flask_app.test_client()
            response = client.post("/crop-predict", data={
                "nitrogen": "80",
                "phosphorous": "40",
                "pottasium": "40",
                "ph": "6.5",
                "land_area": "2",
                "land_unit": "invalid-unit",
                "water_source": "canal",
                "water_availability": "high",
                "latitude": "18.52",
                "longitude": "73.85",
                "location_method": "current"
            })

            self.assertEqual(response.status_code, 400)
            self.assertIn(b"Land unit is invalid.", response.data)

    def test_crop_predict_rejects_invalid_water_source(self):
        with patch.object(self.app_module, "get_lang", return_value="en"), \
             patch.object(self.app_module, "render_template", side_effect=lambda template, **context: context.get("error_message", template)):

            client = self.flask_app.test_client()
            response = client.post("/crop-predict", data={
                "nitrogen": "80",
                "phosphorous": "40",
                "pottasium": "40",
                "ph": "6.5",
                "land_area": "2",
                "land_unit": "acre",
                "water_source": "river",
                "water_availability": "high",
                "latitude": "18.52",
                "longitude": "73.85",
                "location_method": "current"
            })

            self.assertEqual(response.status_code, 400)
            self.assertIn(b"Water source is invalid.", response.data)

    def test_crop_predict_get_redirects_back_to_form(self):
        with patch.object(self.app_module, "get_lang", return_value="en"):
            client = self.flask_app.test_client()
            response = client.get("/crop-predict")

            self.assertEqual(response.status_code, 302)
            self.assertIn("/crop-recommend", response.headers["Location"])

    def test_crop_predict_passes_model_info_to_result_template(self):
        captured = {}
        base_df = pd.DataFrame([
            {"Feature": "Rainfall", "Impact Value": 0.42, "Effect": "Positive (Helps crop)"}
        ])

        xai_runtime = {
            "explain_crop_prediction": lambda *_args, **_kwargs: (
                base_df.copy(), base_df.copy(),
                ["Rainfall positively influenced the recommendation of rice."],
                "rice",
                [{"crop": "rice", "confidence": 100.0, "actual_confidence": 100.0}]
            ),
            "explain_specific_crop_prediction": lambda *_args, **_kwargs: (
                base_df.copy(), base_df.copy(),
                ["Rainfall positively influenced the recommendation of rice."],
                "rice",
                [{"crop": "rice", "confidence": 100.0, "actual_confidence": 100.0}]
            ),
        }

        def fake_render(template, **context):
            captured.update(context)
            return context["prediction"]

        with patch.object(self.app_module, "get_lang", return_value="en"), \
             patch.object(self.app_module, "resolve_location_and_weather", return_value={
                 "city": "Pune",
                 "temperature": 27.5,
                 "humidity": 78.0,
                 "rainfall": 210.0,
                 "latitude": 18.52,
                 "longitude": 73.85
             }), \
             patch.object(self.app_module, "get_crop_xai_runtime", return_value=xai_runtime), \
             patch.object(self.app_module, "rerank_top3_predictions_by_season", return_value=("Kharif", [{"crop": "rice", "confidence": 100.0, "actual_confidence": 100.0}])), \
             patch.object(self.app_module, "get_msamb_live_price", return_value=None), \
             patch.object(self.app_module, "estimate_yield_and_profit", return_value=(12.5, 10000.0)), \
             patch.object(self.app_module, "generate_overall_summary", return_value="summary"), \
             patch.object(self.app_module, "get_crop_sowing_guidance", return_value={"best_sowing_window": "Now"}), \
             patch.object(self.app_module, "localize_crop_output", return_value=("rice", [{"crop": "rice", "confidence": 100.0, "actual_confidence": 100.0}], base_df.copy(), ["line"])), \
             patch.object(self.app_module, "render_template", side_effect=fake_render):

            client = self.flask_app.test_client()
            response = client.post("/crop-predict", data={
                "nitrogen": "80",
                "phosphorous": "40",
                "pottasium": "40",
                "ph": "6.5",
                "land_area": "2",
                "land_unit": "acre",
                "water_source": "canal",
                "water_availability": "high",
                "latitude": "18.52",
                "longitude": "73.85",
                "location_method": "current"
            })

            self.assertEqual(response.status_code, 200)
            self.assertEqual(captured["model_info"]["version"], "v2.1")
            self.assertEqual(captured["model_info"]["last_trained"], "2026-03-28")
            self.assertEqual(captured["model_info"]["region"], "Maharashtra-focused")


class FertilizerRouteValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.app_module = importlib.import_module("app")
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest("App dependencies are not installed: {0}".format(exc))

        cls.flask_app = cls.app_module.app
        cls.flask_app.testing = True

    def test_fertilizer_predict_rejects_invalid_crop(self):
        with patch.object(self.app_module, "get_lang", return_value="en"), \
             patch.object(self.app_module, "render_template", side_effect=lambda template, **context: context.get("error_message", template)):

            client = self.flask_app.test_client()
            response = client.post("/fertilizer-predict", data={
                "nitrogen": "90",
                "phosphorous": "42",
                "pottasium": "58",
                "cropname": "invalid-crop"
            })

            self.assertEqual(response.status_code, 400)
            self.assertIn(b"Crop is invalid.", response.data)

    def test_fertilizer_predict_rejects_missing_crop_and_points_retry_to_fertilizer_form(self):
        captured = {}

        def fake_render(template, **context):
            captured.update(context)
            return context.get("error_message", template)

        with patch.object(self.app_module, "get_lang", return_value="en"), \
             patch.object(self.app_module, "render_template", side_effect=fake_render):
            client = self.flask_app.test_client()
            response = client.post("/fertilizer-predict", data={
                "nitrogen": "90",
                "phosphorous": "42",
                "pottasium": "58",
                "cropname": ""
            })

            self.assertEqual(response.status_code, 400)
            self.assertIn(b"Crop is required.", response.data)
            self.assertEqual(captured["retry_url"], "/fertilizer?lang=en")

    def test_fertilizer_predict_rejects_out_of_range_nitrogen(self):
        with patch.object(self.app_module, "get_lang", return_value="en"), \
             patch.object(self.app_module, "render_template", side_effect=lambda template, **context: context.get("error_message", template)):
            client = self.flask_app.test_client()
            response = client.post("/fertilizer-predict", data={
                "nitrogen": "301",
                "phosphorous": "42",
                "pottasium": "58",
                "cropname": "rice"
            })

            self.assertEqual(response.status_code, 400)
            self.assertIn(b"Nitrogen must be at most 300.", response.data)

    def test_fertilizer_predict_saves_result_for_language_toggle(self):
        with patch.object(
            self.app_module,
            "render_template",
            side_effect=lambda template, **context: context["current_result_url"]
        ):
            client = self.flask_app.test_client()
            response = client.post("/fertilizer-predict", data={
                "nitrogen": "90",
                "phosphorous": "42",
                "pottasium": "58",
                "cropname": "rice",
                "lang": "en"
            })

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_data(as_text=True), "/fertilizer-result?lang=en")

            with client.session_transaction() as session_data:
                self.assertIn("last_fertilizer_result", session_data)
                self.assertEqual(session_data["last_fertilizer_result"]["crop_name"], "rice")
                self.assertEqual(session_data["last_fertilizer_result"]["reference_n"], 80)
                self.assertEqual(session_data["last_fertilizer_result"]["actual_n"], 90)
                self.assertIn(
                    session_data["last_fertilizer_result"]["recommendation_key"],
                    {"Balanced", "NHigh", "Nlow", "PHigh", "Plow", "KHigh", "Klow"}
                )

    def test_fertilizer_result_page_renders_structured_plan_in_marathi(self):
        client = self.flask_app.test_client()

        with client.session_transaction() as session_data:
            session_data["last_fertilizer_result"] = {
                "crop_name": "rice",
                "recommendation_key": "NHigh",
                "reference_n": 80,
                "reference_p": 40,
                "reference_k": 40,
                "actual_n": 95,
                "actual_p": 40,
                "actual_k": 40
            }

        response = client.get("/fertilizer-result?lang=mr")
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("बेसल डोस", body)
        self.assertIn("प्रति एकर प्रमाण", body)
        self.assertIn("सेंद्रिय पर्याय", body)

    def test_fertilizer_result_page_redirects_without_saved_result(self):
        client = self.flask_app.test_client()
        response = client.get("/fertilizer-result?lang=mr")

        self.assertEqual(response.status_code, 302)
        self.assertIn("/fertilizer?lang=mr", response.headers["Location"])

    def test_fertilizer_predict_renders_stagewise_plan_sections(self):
        client = self.flask_app.test_client()
        response = client.post("/fertilizer-predict", data={
            "nitrogen": "92",
            "phosphorous": "5",
            "pottasium": "22",
            "cropname": "maize",
            "lang": "en"
        })

        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Crop-Specific Fertilizer Plan", body)
        self.assertIn("Basal Dose", body)
        self.assertIn("Vegetative Stage Dose", body)
        self.assertIn("Flowering Stage Dose", body)
        self.assertIn("Quantity Per Acre", body)
        self.assertIn("Organic Alternative", body)
        self.assertIn("Model version", body)
        self.assertIn("v2.1", body)
        self.assertIn("Last trained", body)
        self.assertIn("2026-03-28", body)
        self.assertIn("Maharashtra-focused", body)
        self.assertIn("Maize", body)

    def test_fertilizer_predict_accepts_every_available_crop(self):
        fertilizer_df = self.app_module.get_fertilizer_reference_df()

        with patch.object(
            self.app_module,
            "render_template",
            side_effect=lambda template, **context: context["current_result_url"]
        ):
            client = self.flask_app.test_client()

            for row in fertilizer_df.itertuples(index=False):
                with self.subTest(crop=row.Crop):
                    response = client.post("/fertilizer-predict", data={
                        "nitrogen": str(int(row.N)),
                        "phosphorous": str(int(row.P)),
                        "pottasium": str(int(row.K)),
                        "cropname": row.Crop,
                        "lang": "en"
                    })

                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.get_data(as_text=True), "/fertilizer-result?lang=en")

    def test_feedback_prompt_appears_only_on_third_prediction(self):
        client = self.flask_app.test_client()

        first_response = client.post("/fertilizer-predict", data={
            "nitrogen": "92",
            "phosphorous": "5",
            "pottasium": "22",
            "cropname": "maize",
            "lang": "en"
        })
        second_response = client.post("/fertilizer-predict", data={
            "nitrogen": "92",
            "phosphorous": "5",
            "pottasium": "22",
            "cropname": "maize",
            "lang": "en"
        })
        third_response = client.post("/fertilizer-predict", data={
            "nitrogen": "92",
            "phosphorous": "5",
            "pottasium": "22",
            "cropname": "maize",
            "lang": "en"
        })

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        self.assertEqual(third_response.status_code, 200)
        self.assertNotIn("Quick Feedback", first_response.get_data(as_text=True))
        self.assertNotIn("Quick Feedback", second_response.get_data(as_text=True))
        self.assertIn("Quick Feedback", third_response.get_data(as_text=True))
        self.assertIn('id="feedbackPopupBackdrop"', third_response.get_data(as_text=True))
        self.assertIn('id="feedbackPopupClose"', third_response.get_data(as_text=True))

        with client.session_transaction() as session_data:
            self.assertEqual(session_data["prediction_feedback_count"], 3)
            self.assertTrue(session_data["last_fertilizer_result"]["show_feedback_prompt"])

    def test_prediction_feedback_submission_thanks_user_and_hides_prompt(self):
        client = self.flask_app.test_client()
        feedback_file = os.path.join("Data", "test_prediction_feedback.csv")

        if os.path.exists(feedback_file):
            os.remove(feedback_file)

        with client.session_transaction() as session_data:
            session_data["prediction_feedback_count"] = 3
            session_data["last_fertilizer_result"] = {
                "crop_name": "maize",
                "recommendation_key": "Plow",
                "reference_n": 80,
                "reference_p": 40,
                "reference_k": 20,
                "actual_n": 92,
                "actual_p": 5,
                "actual_k": 22,
                "prediction_count": 3,
                "show_feedback_prompt": True,
                "feedback_submitted": False
            }

        try:
            with patch.object(self.app_module, "get_prediction_feedback_file_path", return_value=feedback_file):
                response = client.post("/prediction-feedback?lang=en", data={
                    "prediction_type": "fertilizer",
                    "prediction_name": "maize",
                    "usefulness": "useful",
                    "chosen_crop": "maize",
                    "season_result": "Good early growth.",
                    "prediction_count": "3",
                    "return_path": "/fertilizer-result?lang=en"
                }, follow_redirects=True)

            body = response.get_data(as_text=True)

            self.assertEqual(response.status_code, 200)
            self.assertIn("Feedback Saved", body)
            self.assertNotIn("Quick Feedback", body)

            with client.session_transaction() as session_data:
                self.assertTrue(session_data["last_fertilizer_result"]["feedback_submitted"])
                self.assertFalse(session_data["last_fertilizer_result"]["show_feedback_prompt"])

            with open(feedback_file, "r", encoding="utf-8") as saved_feedback:
                feedback_rows = saved_feedback.read()

            self.assertIn("fertilizer", feedback_rows)
            self.assertIn("Good early growth.", feedback_rows)
        finally:
            if os.path.exists(feedback_file):
                os.remove(feedback_file)


if __name__ == "__main__":
    unittest.main()
