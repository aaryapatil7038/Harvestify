import argparse
import itertools
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple
from urllib.parse import urljoin, urlparse
from unittest.mock import patch

import pandas as pd
import requests
from bs4 import BeautifulSoup

from utils.crop_features import FEATURE_COLUMNS


DEFAULT_SEASON = "Summer"
DEFAULT_WEATHER = {
    "city": "Pune",
    "temperature": 33.0,
    "humidity": 45.0,
    "rainfall": 80.0,
    "latitude": 18.5204,
    "longitude": 73.8567,
}


@dataclass
class PredictionCase:
    name: str
    form_data: Dict[str, str]


@dataclass
class ValidationCase:
    name: str
    form_data: Dict[str, str]
    expected_status: int
    expected_text: str


@dataclass
class TestResult:
    suite: str
    name: str
    status: str
    details: str


def make_form_data(**overrides: str) -> Dict[str, str]:
    form_data = {
        "lang": "en",
        "nitrogen": "60",
        "phosphorous": "50",
        "pottasium": "50",
        "ph": "6.8",
        "land_area": "2",
        "land_unit": "acre",
        "water_source": "borewell",
        "water_availability": "medium",
        "latitude": "18.5204",
        "longitude": "73.8567",
        "location_method": "current",
        "manual_location": "",
    }
    form_data.update(overrides)
    return form_data


PREDICTION_CASES: List[PredictionCase] = [
    PredictionCase("baseline_medium", make_form_data()),
    PredictionCase("dry_low_input", make_form_data(
        nitrogen="20",
        phosphorous="20",
        pottasium="20",
        ph="5.8",
        water_source="rainfed",
        water_availability="low",
    )),
    PredictionCase("high_irrigated", make_form_data(
        nitrogen="100",
        phosphorous="80",
        pottasium="80",
        ph="6.5",
        water_source="canal",
        water_availability="high",
    )),
    PredictionCase("water_profile_change", make_form_data(
        water_source="drip",
        water_availability="high",
    )),
    PredictionCase("max_npk_boundary", make_form_data(
        nitrogen="300",
        phosphorous="300",
        pottasium="300",
        ph="7.0",
        water_source="canal",
        water_availability="high",
    )),
    PredictionCase("min_npk_boundary", make_form_data(
        nitrogen="0",
        phosphorous="0",
        pottasium="0",
        ph="0",
        water_source="rainfed",
        water_availability="low",
    )),
    PredictionCase("high_ph_boundary", make_form_data(
        nitrogen="45",
        phosphorous="30",
        pottasium="25",
        ph="14",
        water_source="rainfed",
        water_availability="low",
    )),
    PredictionCase("manual_coordinates_mode", make_form_data(
        location_method="manual",
        latitude="18.5204",
        longitude="73.8567",
        manual_location="",
    )),
]


VALIDATION_CASES: List[ValidationCase] = [
    ValidationCase(
        "missing_nitrogen",
        make_form_data(nitrogen=""),
        400,
        "Nitrogen is required.",
    ),
    ValidationCase(
        "invalid_water_source",
        make_form_data(water_source="river"),
        400,
        "Water source is invalid.",
    ),
    ValidationCase(
        "invalid_water_availability",
        make_form_data(water_availability="ultra"),
        400,
        "Water availability is invalid.",
    ),
    ValidationCase(
        "invalid_land_unit",
        make_form_data(land_unit="bigha"),
        400,
        "Land unit is invalid.",
    ),
    ValidationCase(
        "negative_land_area",
        make_form_data(land_area="-1"),
        400,
        "Land area must be at least 0.01.",
    ),
    ValidationCase(
        "invalid_coordinates",
        make_form_data(latitude="", longitude=""),
        400,
        "Current location coordinates are missing or invalid.",
    ),
    ValidationCase(
        "manual_without_location",
        make_form_data(location_method="manual", latitude="", longitude="", manual_location=""),
        400,
        "Please enter a manual farm location or valid coordinates.",
    ),
]


def normalize_label(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower())
    return " ".join(normalized.split())


def extract_script_json_value(html: str, key: str):
    pattern = re.compile(r"{0}\s*:\s*(\".*?\"|null|true|false|\[.*?\]|\{{.*?\}})".format(re.escape(key)), re.DOTALL)
    match = pattern.search(html)
    if not match:
        return ""

    try:
        return json.loads(match.group(1))
    except Exception:
        return ""


def extract_card_value(soup: BeautifulSoup, label_text: str) -> str:
    expected = normalize_label(label_text)

    for card in soup.select(".info-box-clean"):
        heading = card.find("h5")
        value = card.find("p")
        if not heading or not value:
            continue

        actual = normalize_label(heading.get_text(" ", strip=True))
        if actual == expected or actual.startswith(expected) or expected in actual:
            return value.get_text(" ", strip=True)

    return ""


def parse_number(text_value: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", text_value or "")
    if not match:
        raise ValueError("Could not parse numeric value from: {0}".format(text_value))
    return float(match.group(0))


def extract_error_message(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text)


def extract_prediction_payload(html: str) -> Dict[str, object]:
    soup = BeautifulSoup(html, "html.parser")
    main_result = soup.select_one(".main-result")

    if not main_result:
        raise ValueError("Could not find crop result block in the rendered page.")

    return {
        "display_prediction": main_result.get_text(" ", strip=True),
        "prediction_raw": extract_script_json_value(html, "predictionRaw"),
        "current_season_raw": extract_script_json_value(html, "currentSeasonRaw"),
        "temperature": parse_number(extract_card_value(soup, "Temperature")),
        "humidity": parse_number(extract_card_value(soup, "Humidity")),
        "rainfall": parse_number(extract_card_value(soup, "Rainfall")),
    }


def compute_expected_prediction(app_module, form_data: Dict[str, str], weather_info: Dict[str, float], season: str) -> str:
    input_df = pd.DataFrame(
        [[
            float(form_data["nitrogen"]),
            float(form_data["phosphorous"]),
            float(form_data["pottasium"]),
            float(weather_info["temperature"]),
            float(weather_info["humidity"]),
            float(form_data["ph"]),
            float(weather_info["rainfall"]),
            season,
            form_data["water_source"],
            form_data["water_availability"],
        ]],
        columns=FEATURE_COLUMNS,
    )

    crop_xai_runtime = app_module.get_crop_xai_runtime()
    _, _, _, prediction, top3_predictions = crop_xai_runtime["explain_crop_prediction"](input_df)
    _, reranked_top3 = app_module.rerank_top3_predictions_by_season(top3_predictions, season=season)
    final_prediction = reranked_top3[0]["crop"]
    return final_prediction


def run_prediction_case_embedded(app_module, test_case: PredictionCase, season: str, weather_info: Dict[str, float]) -> TestResult:
    with patch.object(app_module, "get_lang", return_value="en"), \
         patch.object(app_module, "resolve_location_and_weather", return_value=weather_info), \
         patch.object(app_module, "get_current_crop_season", return_value=season), \
         patch.object(app_module, "get_msamb_live_price", return_value=None):

        client = app_module.app.test_client()
        response = client.post("/crop-predict", data=test_case.form_data)

    if response.status_code != 200:
        return TestResult("prediction", test_case.name, "FAIL", "HTTP {0}".format(response.status_code))

    try:
        payload = extract_prediction_payload(response.data.decode("utf-8", errors="replace"))
    except Exception as exc:
        return TestResult("prediction", test_case.name, "FAIL", str(exc))

    expected_raw = compute_expected_prediction(app_module, test_case.form_data, weather_info, season)
    actual_raw = str(payload["prediction_raw"] or "")
    status = "PASS" if expected_raw == actual_raw else "FAIL"
    details = "expected={0}, actual={1}, display={2}".format(expected_raw, actual_raw, payload["display_prediction"])
    return TestResult("prediction", test_case.name, status, details)


def run_prediction_case_live(app_module, session: requests.Session, base_url: str, test_case: PredictionCase) -> TestResult:
    response = session.post(urljoin(base_url.rstrip("/") + "/", "crop-predict"), data=test_case.form_data, timeout=60, allow_redirects=True)

    if response.status_code != 200:
        return TestResult("prediction", test_case.name, "FAIL", "HTTP {0}".format(response.status_code))

    try:
        payload = extract_prediction_payload(response.text)
    except Exception as exc:
        return TestResult("prediction", test_case.name, "FAIL", str(exc))

    weather_info = {
        "temperature": payload["temperature"],
        "humidity": payload["humidity"],
        "rainfall": payload["rainfall"],
    }
    season = str(payload["current_season_raw"] or DEFAULT_SEASON)
    expected_raw = compute_expected_prediction(app_module, test_case.form_data, weather_info, season)
    actual_raw = str(payload["prediction_raw"] or "")
    status = "PASS" if expected_raw == actual_raw else "FAIL"
    details = "expected={0}, actual={1}, display={2}, season={3}".format(
        expected_raw,
        actual_raw,
        payload["display_prediction"],
        season,
    )
    return TestResult("prediction", test_case.name, status, details)


def run_validation_case_embedded(app_module, test_case: ValidationCase, season: str, weather_info: Dict[str, float]) -> TestResult:
    with patch.object(app_module, "get_lang", return_value="en"), \
         patch.object(app_module, "resolve_location_and_weather", return_value=weather_info), \
         patch.object(app_module, "get_current_crop_season", return_value=season), \
         patch.object(app_module, "get_msamb_live_price", return_value=None):

        client = app_module.app.test_client()
        response = client.post("/crop-predict", data=test_case.form_data)

    body = response.data.decode("utf-8", errors="replace")
    body_text = extract_error_message(body)
    status_ok = response.status_code == test_case.expected_status
    text_ok = test_case.expected_text in body_text
    status = "PASS" if status_ok and text_ok else "FAIL"
    details = "status={0}, expected_status={1}, contains_expected={2}".format(
        response.status_code,
        test_case.expected_status,
        text_ok,
    )
    return TestResult("validation", test_case.name, status, details)


def run_validation_case_live(session: requests.Session, base_url: str, test_case: ValidationCase) -> TestResult:
    response = session.post(urljoin(base_url.rstrip("/") + "/", "crop-predict"), data=test_case.form_data, timeout=60, allow_redirects=True)
    body_text = extract_error_message(response.text)
    status_ok = response.status_code == test_case.expected_status
    text_ok = test_case.expected_text in body_text
    status = "PASS" if status_ok and text_ok else "FAIL"
    details = "status={0}, expected_status={1}, contains_expected={2}".format(
        response.status_code,
        test_case.expected_status,
        text_ok,
    )
    return TestResult("validation", test_case.name, status, details)


def run_get_route_embedded(app_module) -> TestResult:
    with patch.object(app_module, "get_lang", return_value="en"):
        client = app_module.app.test_client()
        response = client.get("/crop-predict", follow_redirects=False)

    location = response.headers.get("Location", "")
    passed = response.status_code in (301, 302, 303, 307, 308) and "/crop-recommend" in location
    return TestResult("route", "get_crop_predict_redirect", "PASS" if passed else "FAIL", "status={0}, location={1}".format(response.status_code, location))


def run_get_route_live(session: requests.Session, base_url: str) -> TestResult:
    response = session.get(urljoin(base_url.rstrip("/") + "/", "crop-predict"), timeout=60, allow_redirects=False)
    location = response.headers.get("Location", "")
    path = urlparse(location).path if location else ""
    passed = response.status_code in (301, 302, 303, 307, 308) and path.endswith("/crop-recommend")
    return TestResult("route", "get_crop_predict_redirect", "PASS" if passed else "FAIL", "status={0}, location={1}".format(response.status_code, location))


def run_repeat_submission_live(app_module, session: requests.Session, base_url: str, first_case: PredictionCase, second_case: PredictionCase) -> TestResult:
    first_result = run_prediction_case_live(app_module, session, base_url, first_case)
    second_result = run_prediction_case_live(app_module, session, base_url, second_case)

    if first_result.status == "FAIL" or second_result.status == "FAIL":
        return TestResult("stale", "repeat_submission_freshness", "FAIL", "one of the paired prediction checks failed")

    first_actual = re.search(r"actual=([^,]+)", first_result.details)
    second_actual = re.search(r"actual=([^,]+)", second_result.details)
    if not first_actual or not second_actual:
        return TestResult("stale", "repeat_submission_freshness", "FAIL", "could not parse actual predictions")

    first_crop = first_actual.group(1)
    second_crop = second_actual.group(1)
    passed = first_crop != second_crop
    return TestResult(
        "stale",
        "repeat_submission_freshness",
        "PASS" if passed else "FAIL",
        "first={0}, second={1}".format(first_crop, second_crop),
    )


def run_diversity_matrix_embedded(app_module, season: str, weather_info: Dict[str, float]) -> TestResult:
    combinations = list(itertools.product(
        [0, 40, 120],
        [0, 40, 120],
        [0, 40, 120],
        [5.5, 6.8, 8.0],
        ["rainfed", "borewell", "canal"],
        ["low", "medium", "high"],
    ))

    predictions = []
    for n_value, p_value, k_value, ph_value, water_source, water_availability in combinations:
        form_data = make_form_data(
            nitrogen=str(n_value),
            phosphorous=str(p_value),
            pottasium=str(k_value),
            ph=str(ph_value),
            water_source=water_source,
            water_availability=water_availability,
        )
        expected = compute_expected_prediction(app_module, form_data, weather_info, season)
        predictions.append(expected)

    unique_predictions = sorted(set(predictions))
    passed = len(unique_predictions) >= 5 and len(unique_predictions) < len(predictions)
    return TestResult(
        "matrix",
        "prediction_diversity_matrix",
        "PASS" if passed else "FAIL",
        "unique_predictions={0}, sample={1}".format(len(unique_predictions), ", ".join(unique_predictions[:10])),
    )


def print_report(results: List[TestResult]) -> None:
    def safe_console_text(value: str) -> str:
        return str(value).encode("cp1252", errors="replace").decode("cp1252")

    print("\nWebsite Edge Case Report")
    print("=" * 96)
    print("{:<14} {:<30} {:<8} {}".format("Suite", "Case", "Status", "Details"))
    print("-" * 96)

    for result in results:
        print("{:<14} {:<30} {:<8} {}".format(
            safe_console_text(result.suite),
            safe_console_text(result.name),
            safe_console_text(result.status),
            safe_console_text(result.details),
        ))

    total = len(results)
    passed = sum(1 for result in results if result.status == "PASS")
    print("-" * 96)
    print("Passed {0}/{1} checks".format(passed, total))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run broad website-level crop prediction edge case tests.")
    parser.add_argument("--base-url", help="Optional running website URL, for example http://127.0.0.1:5000")
    parser.add_argument("--season", default=DEFAULT_SEASON, help="Season used in embedded mode. Default: Summer")
    args = parser.parse_args()

    try:
        import app as app_module
    except Exception as exc:
        print("Could not import app.py: {0}".format(exc))
        return 1

    results: List[TestResult] = []

    if args.base_url:
        session = requests.Session()
        results.append(run_get_route_live(session, args.base_url))
        for test_case in PREDICTION_CASES:
            results.append(run_prediction_case_live(app_module, session, args.base_url, test_case))
        for test_case in VALIDATION_CASES:
            results.append(run_validation_case_live(session, args.base_url, test_case))
        results.append(run_repeat_submission_live(app_module, session, args.base_url, PREDICTION_CASES[0], PREDICTION_CASES[1]))
    else:
        results.append(run_get_route_embedded(app_module))
        for test_case in PREDICTION_CASES:
            results.append(run_prediction_case_embedded(app_module, test_case, args.season, DEFAULT_WEATHER))
        for test_case in VALIDATION_CASES:
            results.append(run_validation_case_embedded(app_module, test_case, args.season, DEFAULT_WEATHER))
        results.append(run_diversity_matrix_embedded(app_module, args.season, DEFAULT_WEATHER))

    print_report(results)
    return 0 if all(result.status == "PASS" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
