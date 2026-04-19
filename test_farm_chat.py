import unittest

from utils.farm_chat import generate_farm_chat_reply


class FarmChatTests(unittest.TestCase):
    def test_typo_weather_question_uses_latest_weather_snapshot(self):
        response = generate_farm_chat_reply(
            "whats todays where at my location",
            "en",
            session_context={},
            client_context={
                "latestWeather": {
                    "city": "Karad",
                    "temperature": 31,
                    "humidity": 62,
                    "rainfall": 0,
                }
            },
            crop_translations={},
        )

        self.assertIn("Karad", response["reply"])
        self.assertIn("Temperature", response["reply"])

    def test_crop_prediction_website_help_returns_steps(self):
        response = generate_farm_chat_reply(
            "how to get crop preditoin in this website?",
            "en",
            session_context={},
            client_context={},
            crop_translations={},
        )

        self.assertIn("Crop page", response["reply"])
        self.assertIn("Predict", response["reply"])

    def test_cabbage_cultivation_returns_crop_specific_guidance(self):
        response = generate_farm_chat_reply(
            "how to cultivate cabbage?",
            "en",
            session_context={},
            client_context={},
            crop_translations={},
        )

        self.assertIn("cabbage", response["reply"].lower())
        self.assertIn("well-drained soil", response["reply"].lower())

    def test_roman_marathi_sunflower_message_returns_crop_specific_guidance(self):
        response = generate_farm_chat_reply(
            "Mala suryapulachi kshetri Karachi aaye.",
            "mr",
            session_context={},
            client_context={},
            crop_translations={},
        )

        self.assertIn("सूर्यफूल", response["reply"])
        self.assertIn("लागवडीसाठी", response["reply"])

    def test_reply_in_marathi_preference_message_returns_confirmation(self):
        response = generate_farm_chat_reply(
            "Whenever I ask something in Marathi, please reply in Marathi.",
            "mr",
            session_context={},
            client_context={},
            crop_translations={},
        )

        self.assertIn("मराठीत", response["reply"])

    def test_roman_marathi_rain_question_uses_weather_path(self):
        response = generate_farm_chat_reply(
            "Amchai kade paus khoo pachchas toh aaye darmi.",
            "mr",
            session_context={},
            client_context={
                "latestWeather": {
                    "city": "Karad",
                    "temperature": 31,
                    "humidity": 62,
                    "rainfall": 50,
                }
            },
            crop_translations={},
        )

        self.assertTrue("हवामान" in response["reply"] or "पाऊस" in response["reply"] or "Rainfall" in response["reply"])


if __name__ == "__main__":
    unittest.main()
