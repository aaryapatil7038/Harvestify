import requests
from bs4 import BeautifulSoup
from functools import lru_cache

# crop name -> MSAMB commodity code
COMMODITY_CODES = {
    "apple": "07026",
    "banana": "01001",
    "grapes": "06004",
    "mango": "07001",
    "orange": "06011",
    "papaya": "07014",
    "pomegranate": "06010",
    "watermelon": "07015",
    "muskmelon": "07012",
    "maize": "04006",
    "wheat": "04009",
    "rice": "04008",
    "soybean": "04023",
    "groundnut": "04015",
    "cotton": "04035",
    "jowar": "04007",
    "bajra": "04013",
    "tur": "04020",
    "gram": "04018",
    "onion": "06016",
    "tomato": "07017",
    "chili": "07003",
    "turmeric": "07114",
    "sorghum": "04007",
    "ragi": "04012",
    "barley": "04002",
    "chickpea": "04018",
    "pigeonpeas": "04020",
    "mungbean": "04021",
    "blackgram": "04022",
    "lentil": "04019",
    "sugarcane": "06015"
}

BASE_URL = "https://www.msamb.com/ApmcDetail/DataGridBind"


def to_float(value):
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return 0.0


@lru_cache(maxsize=128)
def get_msamb_live_price(crop_name, city):
    crop_name = crop_name.strip().lower()
    city = city.strip()

    crop_name = {
        "jowar": "jowar",
        "bajra": "bajra",
        "tur": "tur",
        "gram": "gram",
        "chilli": "chili",
    }.get(crop_name, crop_name)

    commodity_code = COMMODITY_CODES.get(crop_name)
    if not commodity_code:
        return None

    url = f"{BASE_URL}?commodityCode={commodity_code}&apmcCode=null"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://www.msamb.com/ApmcDetail/APMCPriceInformation"
    }

    response = requests.get(url, headers=headers, timeout=6)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    rows = soup.find_all("tr")

    latest_date = None
    city_match = None
    fallback_match = None

    for row in rows:
        cols = [td.get_text(strip=True) for td in row.find_all("td")]

        # date row
        if len(cols) == 1:
            latest_date = cols[0]

        # price row
        elif len(cols) == 7:
            market = cols[0]
            variety = cols[1]
            unit = cols[2]
            arrival = cols[3]
            min_price = cols[4]
            max_price = cols[5]
            modal_price = cols[6]

            record = {
                "date": latest_date,
                "market": market,
                "variety": variety,
                "unit": unit,
                "arrival": to_float(arrival),
                "min_price": to_float(min_price),
                "max_price": to_float(max_price),
                "modal_price": to_float(modal_price)
            }

            if city in market:
                city_match = record
                break

            if fallback_match is None:
                fallback_match = record

    if city_match:
        return city_match

    return fallback_match
