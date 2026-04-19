import unittest
from unittest.mock import Mock, patch

import requests

from utils.msamb_price import clear_msamb_price_cache, get_msamb_live_price, resolve_commodity_code


HTML_RESPONSE = """
<table>
    <tr><td>18-04-2026</td></tr>
    <tr>
        <td>Pune</td>
        <td>Local</td>
        <td>Quintal</td>
        <td>120</td>
        <td>2000</td>
        <td>2600</td>
        <td>2300</td>
    </tr>
    <tr>
        <td>Nashik</td>
        <td>Local</td>
        <td>Quintal</td>
        <td>80</td>
        <td>2100</td>
        <td>2700</td>
        <td>2400</td>
    </tr>
</table>
"""


class MSAMBPriceCacheTests(unittest.TestCase):
    def setUp(self):
        clear_msamb_price_cache()

    def test_reuses_crop_fetch_for_multiple_city_lookups(self):
        session = Mock()
        response = Mock()
        response.text = HTML_RESPONSE
        response.raise_for_status = Mock()
        session.get.return_value = response

        with patch("utils.msamb_price.build_session", return_value=session), \
             patch("utils.msamb_price.get_live_commodity_options", return_value=[]):
            pune_record = get_msamb_live_price("rice", "Pune")
            nashik_record = get_msamb_live_price("rice", "Nashik")

        self.assertEqual(session.get.call_count, 1)
        self.assertEqual(pune_record["market"], "Pune")
        self.assertEqual(nashik_record["market"], "Nashik")
        self.assertEqual(nashik_record["modal_price"], 2400.0)

    def test_caches_failures_and_returns_none_without_retrying_immediately(self):
        session = Mock()
        session.get.side_effect = requests.RequestException("temporary network issue")

        with patch(
            "utils.msamb_price.build_session",
            return_value=session
        ), patch("utils.msamb_price.get_live_commodity_options", return_value=[]):
            first_result = get_msamb_live_price("rice", "Pune")
            second_result = get_msamb_live_price("rice", "Pune")

        self.assertIsNone(first_result)
        self.assertIsNone(second_result)
        self.assertEqual(session.get.call_count, 1)

    def test_uses_live_commodity_options_to_resolve_current_code(self):
        with patch(
            "utils.msamb_price.get_live_commodity_options",
            return_value=[
                ("04016", "शेंगदाणे"),
                ("04017", "सोयाबिन"),
                ("08071", "टोमॅटो"),
            ]
        ):
            self.assertEqual(resolve_commodity_code("groundnut"), "04016")
            self.assertEqual(resolve_commodity_code("soybean"), "04017")
            self.assertEqual(resolve_commodity_code("tomato"), "08071")

    def test_prefers_static_code_before_fetching_live_options(self):
        with patch("utils.msamb_price.get_live_commodity_options") as live_options_mock:
            self.assertEqual(resolve_commodity_code("wheat"), "02009")

        live_options_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
