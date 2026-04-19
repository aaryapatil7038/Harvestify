import time

import requests
from bs4 import BeautifulSoup

# crop name -> MSAMB commodity code
COMMODITY_CODES = {
    "apple": "07026",
    "banana": "07014",
    "grapes": "07008",
    "mango": "07001",
    "orange": "07027",
    "papaya": "07014",
    "pomegranate": "07007",
    "watermelon": "07015",
    "muskmelon": "07012",
    "maize": "02015",
    "wheat": "02009",
    "rice": "02023",
    "soybean": "04017",
    "groundnut": "04016",
    "cotton": "01001",
    "jowar": "02011",
    "bajra": "02002",
    "tur": "03020",
    "gram": "03006",
    "onion": "08035",
    "tomato": "08071",
    "chili": "10013",
    "turmeric": "10006",
    "sorghum": "02011",
    "ragi": "02016",
    "barley": "02004",
    "chickpea": "03006",
    "pigeonpeas": "03020",
    "mungbean": "03016",
    "blackgram": "03022",
    "lentil": "03012",
    "sugarcane": "06003"
}

BASE_URL = "https://www.msamb.com/ApmcDetail/DataGridBind"
ARRIVAL_PRICE_INFO_URL = "https://www.msamb.com/ApmcDetail/APMCPriceInformation"
PRICE_CACHE_TTL_SECONDS = 30 * 60
PRICE_FAILURE_TTL_SECONDS = 5 * 60
PRICE_REQUEST_TIMEOUT = (1.0, 1.5)
_market_records_cache = {}
_commodity_options_cache = {"options": None, "expires_at": 0}

COMMODITY_LABELS = {
    "apple": ["सफरचंद"],
    "banana": ["केळी"],
    "grapes": ["द्राक्ष"],
    "orange": ["संत्री"],
    "pomegranate": ["डाळींब"],
    "ginger": ["आले"],
    "potato": ["बटाटा"],
    "brinjal": ["वांगी"],
    "cabbage": ["कोबी"],
    "sunflower": ["सुर्यफुल"],
    "sesame": ["तील"],
    "blackgram": ["उडीद"],
    "mungbean": ["मूग"],
    "ragi": ["नाचणी"],
    "lentil": ["मसूर"],
    "rice": ["तांदूळ", "भात - धान"],
    "tomato": ["टोमॅटो"],
    "groundnut": ["शेंगदाणे"],
    "soybean": ["सोयाबिन"],
    "wheat": ["गहू"],
    "cotton": ["कापूस"],
    "tur": ["तूर"],
    "gram": ["हरभरा"],
    "jowar": ["ज्वारी"],
    "bajra": ["बाजरी"],
    "turmeric": ["हळद"],
    "sugarcane": ["उस"],
    "maize": ["मका"],
    "onion": ["कांदा"],
    "chili": ["मिरची (हिरवी)", "मिरची (लाल)"],
}


def to_float(value):
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return 0.0


def clear_msamb_price_cache():
    _market_records_cache.clear()
    _commodity_options_cache["options"] = None
    _commodity_options_cache["expires_at"] = 0


def normalize_crop_name(crop_name):
    crop_name = crop_name.strip().lower()
    return {
        "jowar": "jowar",
        "bajra": "bajra",
        "tur": "tur",
        "gram": "gram",
        "chilli": "chili",
    }.get(crop_name, crop_name)


def build_session():
    session = requests.Session()
    session.trust_env = False
    return session


def get_live_commodity_options():
    now = time.time()
    if _commodity_options_cache["options"] and _commodity_options_cache["expires_at"] > now:
        return _commodity_options_cache["options"]

    try:
        response = build_session().get(ARRIVAL_PRICE_INFO_URL, timeout=PRICE_REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException:
        return _commodity_options_cache["options"] or []

    soup = BeautifulSoup(response.text, "html.parser")
    commodity_select = soup.find(id="drpCommodities")
    options = []

    if commodity_select:
        for option in commodity_select.find_all("option"):
            value = (option.get("value") or "").strip()
            text = option.get_text(strip=True)
            if value and text:
                options.append((value, text))

    _commodity_options_cache["options"] = options
    _commodity_options_cache["expires_at"] = now + PRICE_CACHE_TTL_SECONDS
    return options


def resolve_commodity_code(crop_name):
    normalized_crop_name = normalize_crop_name(crop_name)

    # Prefer the stable built-in mapping first so common crops avoid an
    # extra HTML fetch before the price grid request.
    static_code = COMMODITY_CODES.get(normalized_crop_name)
    if static_code:
        return static_code

    live_options = get_live_commodity_options()

    for candidate_label in COMMODITY_LABELS.get(normalized_crop_name, []):
        for option_value, option_text in live_options:
            if candidate_label in option_text:
                return option_value

    return None


def fetch_market_records_for_code(commodity_code):
    if not commodity_code:
        return None

    url = f"{BASE_URL}?commodityCode={commodity_code}&apmcCode=null"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://www.msamb.com/ApmcDetail/APMCPriceInformation"
    }

    response = build_session().get(url, headers=headers, timeout=PRICE_REQUEST_TIMEOUT)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    rows = soup.find_all("tr")

    latest_date = None
    records = []

    for row in rows:
        cols = [td.get_text(strip=True) for td in row.find_all("td")]

        if len(cols) == 1:
            latest_date = cols[0]
        elif len(cols) == 7:
            records.append({
                "date": latest_date,
                "market": cols[0],
                "variety": cols[1],
                "unit": cols[2],
                "arrival": to_float(cols[3]),
                "min_price": to_float(cols[4]),
                "max_price": to_float(cols[5]),
                "modal_price": to_float(cols[6])
            })

    return records or None


def get_cached_market_records(crop_name):
    normalized_crop_name = normalize_crop_name(crop_name)
    cached_entry = _market_records_cache.get(normalized_crop_name)
    now = time.time()

    if cached_entry and cached_entry["expires_at"] > now:
        return cached_entry["records"]

    commodity_code = COMMODITY_CODES.get(normalized_crop_name)
    if not commodity_code:
        commodity_code = resolve_commodity_code(normalized_crop_name)

    if not commodity_code:
        _market_records_cache[normalized_crop_name] = {
            "records": None,
            "expires_at": now + PRICE_FAILURE_TTL_SECONDS,
        }
        return None

    try:
        records = fetch_market_records_for_code(commodity_code)
    except requests.RequestException:
        if cached_entry and cached_entry["records"]:
            return cached_entry["records"]

        _market_records_cache[normalized_crop_name] = {
            "records": None,
            "expires_at": now + PRICE_FAILURE_TTL_SECONDS,
        }
        return None

    if not records:
        live_commodity_code = resolve_commodity_code(normalized_crop_name)
        if live_commodity_code and live_commodity_code != commodity_code:
            try:
                records = fetch_market_records_for_code(live_commodity_code)
            except requests.RequestException:
                records = None

    _market_records_cache[normalized_crop_name] = {
        "records": records or None,
        "expires_at": now + (PRICE_CACHE_TTL_SECONDS if records else PRICE_FAILURE_TTL_SECONDS),
    }
    return records or None


def get_msamb_live_price(crop_name, city):
    city = city.strip().lower()
    records = get_cached_market_records(crop_name)

    if not records:
        return None

    fallback_match = records[0]
    for record in records:
        if city and city in record["market"].lower():
            return record

    return fallback_match
