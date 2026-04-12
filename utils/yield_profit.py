# utils/yield_profit.py

def estimate_yield_and_profit(crop, land_area, land_unit):
    """
    Estimate yield and profit based on crop type and land size.
    Yield values are approximate quintals per acre.
    """

    # average yield per acre (quintal)
    yield_data = {
        "rice": 22,
        "wheat": 18,
        "maize": 20,
        "soybean": 10,
        "cotton": 8,
        "sugarcane": 400,
        "groundnut": 12,
        "jowar": 14,
        "bajra": 13,
        "tur": 9,
        "gram": 11,
        "onion": 95,
        "tomato": 110,
        "chili": 32,
        "turmeric": 65,
        "banana": 180,
        "orange": 85,
        "grapes": 95,
        "pomegranate": 70,
        "ginger": 60,
        "potato": 85,
        "brinjal": 120,
        "cabbage": 130,
        "sunflower": 8,
        "sesame": 4,
        "ragi": 15,
        "sorghum": 14,
        "millet": 13,
        "barley": 16,
        "chickpea": 11,
        "pigeonpeas": 9,
        "mungbean": 7,
        "blackgram": 7,
        "lentil": 8,
        "cowpea": 6
    }

    crop = crop.lower()

    # default yield if crop not in list
    yield_per_acre = yield_data.get(crop, 10)

    # convert hectare → acre
    if land_unit == "hectare":
        land_area = land_area * 2.471

    estimated_yield = round(yield_per_acre * land_area, 2)

    # rough price assumption (₹ per quintal)
    avg_price = 3000

    estimated_profit = round(estimated_yield * avg_price * 0.65, 2)

    return estimated_yield, estimated_profit
