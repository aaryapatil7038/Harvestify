from difflib import SequenceMatcher
import re


DEFAULT_SUGGESTIONS = {
    "en": [
        "What was my last crop recommendation?",
        "Show my saved farms",
        "How do I use crop prediction?",
        "Explain Kharif and Rabi",
    ],
    "mr": [
        "माझी शेवटची पीक शिफारस कोणती होती?",
        "माझी सेव्ह केलेली शेतं दाखवा",
        "पीक prediction कसे वापरायचे?",
        "खरीप आणि रब्बी समजावून सांगा",
    ],
}


CROP_GUIDES = {
    "sunflower": {
        "aliases": [
            "sunflower", "sun flower", "surajmukhi", "suraj mukhi",
            "suryaphul", "suryaphool", "suryaphulachi",
            "suryaful", "suryafulachi",
            "suryafool", "suryafoolachi",
            "suryafl", "suryaphul", "suryaphulachi",
            "suryapul", "suryapulachi",
        ],
        "answer_en": (
            "For sunflower cultivation, choose a sunny field with good drainage and medium fertile soil. "
            "Sow in a suitable season with proper spacing, keep the field weed free in the early stage, and avoid waterlogging. "
            "Apply balanced fertilizer based on soil condition, give irrigation at critical stages like flowering and seed filling, "
            "and watch for leaf eating caterpillars, sucking pests, and head rot. Harvest when the flower head turns yellow to brown and seeds harden."
        ),
        "answer_mr": (
            "सूर्यफूल लागवडीसाठी चांगला सूर्यप्रकाश, निचरा होणारे शेत आणि मध्यम सुपीक जमीन निवडा. "
            "योग्य हंगामात योग्य अंतरावर पेरणी करा, सुरुवातीच्या अवस्थेत तण नियंत्रण ठेवा आणि पाणी साचू देऊ नका. "
            "मातीच्या स्थितीनुसार संतुलित खत द्या, फुलोरा आणि बियाणे भरण्याच्या टप्प्यावर आवश्यक सिंचन करा, "
            "आणि पाने खाणाऱ्या अळ्या, रसशोषक किडी व बोंड कुज यावर लक्ष ठेवा. फुलाचा डोका पिवळसर-तपकिरी झाला आणि बिया घट्ट झाल्या की काढणी करा."
        ),
    },
    "cabbage": {
        "aliases": ["cabbage", "patta gobhi", "gobhi"],
        "answer_en": (
            "For cabbage cultivation, plant in cool weather in fertile, well-drained soil with good organic matter. "
            "Raise healthy seedlings first, then transplant them with proper spacing so heads can form well. "
            "Keep moisture steady with light regular irrigation, avoid waterlogging, apply balanced fertilizer in split doses, "
            "and monitor for caterpillars, aphids, and rot. Harvest when the head becomes firm and compact."
        ),
        "answer_mr": (
            "कोबी लागवडीसाठी थंड हवामान, सुपीक व चांगला निचरा होणारी जमीन आणि भरपूर सेंद्रिय अन्नद्रव्ये उपयुक्त असतात. "
            "आधी निरोगी रोपे तयार करा, मग योग्य अंतरावर लागवड करा जेणेकरून गड्डा व्यवस्थित तयार होईल. "
            "हलके पण नियमित पाणी द्या, पाणी साचू देऊ नका, संतुलित खत विभागून द्या, आणि अळी, मावा व कुज रोगांवर लक्ष ठेवा. "
            "गड्डा घट्ट व भरदार झाला की काढणी करा."
        ),
    },
    "tomato": {
        "aliases": ["tomato", "tamatar"],
        "answer_en": (
            "Tomato grows best in fertile well-drained soil with strong sunlight and regular moisture. "
            "Use healthy seedlings, keep proper spacing, support plants when needed, feed balanced nutrients in splits, "
            "and watch for blight, fruit borer, and sucking pests."
        ),
        "answer_mr": (
            "टोमॅटो लागवडीसाठी सुपीक, चांगला निचरा होणारी जमीन, भरपूर सूर्यप्रकाश आणि नियमित ओलावा उपयुक्त असतो. "
            "निरोगी रोपे वापरा, योग्य अंतर ठेवा, गरजेनुसार आधार द्या, संतुलित अन्नद्रव्ये विभागून द्या, "
            "आणि करपा, फळ छिद्रक व रसशोषक किडींवर लक्ष ठेवा."
        ),
    },
    "onion": {
        "aliases": ["onion", "kanda"],
        "answer_en": (
            "For onion, use a fine nursery or seedbed, transplant healthy seedlings, and maintain uniform moisture without waterlogging. "
            "Balanced sulfur and potassium help bulb quality, while too much late nitrogen can reduce storage life. "
            "Harvest after neck fall and cure bulbs well before storage."
        ),
        "answer_mr": (
            "कांदा लागवडीसाठी बारीक भुसभुशीत रोपवाटिका तयार करा, निरोगी रोपे लावा आणि पाणी साचू न देता समान ओलावा ठेवा. "
            "गंधक आणि पोटॅशियममुळे कांद्याची गुणवत्ता सुधारते, तर उशिरा जास्त नायट्रोजन दिल्यास साठवण क्षमता कमी होऊ शकते. "
            "मान पडल्यावर काढणी करा आणि साठवणुकीपूर्वी चांगले वाळवून घ्या."
        ),
    },
    "rice": {
        "aliases": ["rice", "paddy", "dhan"],
        "answer_en": (
            "Rice needs timely sowing or nursery raising, level fields, good weed control, and dependable water management. "
            "Apply nitrogen in split doses and monitor for stem borer, leaf folder, and blast."
        ),
        "answer_mr": (
            "भातासाठी वेळेवर पेरणी किंवा रोपवाटिका, समतल शेत, चांगले तण नियंत्रण आणि खात्रीशीर पाणी व्यवस्थापन आवश्यक असते. "
            "नायट्रोजन विभागून द्या आणि खोड किडा, पान गुंडाळणारी अळी आणि करपा यावर लक्ष ठेवा."
        ),
    },
    "wheat": {
        "aliases": ["wheat", "gehun"],
        "answer_en": (
            "Wheat performs best with timely sowing, proper seed rate, clean early weed control, and irrigation at critical stages like crown root initiation, flowering, and grain filling. "
            "Use balanced nutrients and avoid heavy late irrigation near maturity."
        ),
        "answer_mr": (
            "गहूसाठी वेळेवर पेरणी, योग्य बियाणे प्रमाण, सुरुवातीचे स्वच्छ तण नियंत्रण आणि मुकुटमुळ अवस्था, फुलोरा व दाणे भरण्याच्या महत्त्वाच्या टप्प्यांवर सिंचन आवश्यक असते. "
            "संतुलित अन्नद्रव्ये द्या आणि पक्वतेच्या जवळ अतिपाणी देणे टाळा."
        ),
    },
}


FARM_KNOWLEDGE = [
    {
        "keywords": ["kharif", "rabi", "zaid", "season", "monsoon", "हंगाम", "खरीप", "रब्बी", "जायद"],
        "answer_en": (
            "In India, Kharif crops are usually sown around the monsoon and harvested in autumn. "
            "Rabi crops are usually sown after the monsoon and harvested in spring. "
            "Zaid crops are short-duration summer crops grown between the main seasons."
        ),
        "answer_mr": (
            "भारतात खरीप पिके साधारणपणे पावसाळ्याच्या सुरुवातीला घेतली जातात आणि नंतर कापणी होते. "
            "रब्बी पिके पावसाळ्यानंतर घेतली जातात आणि वसंत ऋतूत कापणी होते. "
            "जायद पिके ही दोन मुख्य हंगामांमधील कमी कालावधीची उन्हाळी पिके असतात."
        ),
    },
    {
        "keywords": ["soil", "ph", "npk", "nitrogen", "phosphorus", "potassium", "माती", "नायट्रोजन", "फॉस्फरस", "पोटॅशियम"],
        "answer_en": (
            "A soil test helps you check pH and nutrient status. "
            "Nitrogen supports leaf growth, phosphorus supports roots and flowering, and potassium supports vigor and stress tolerance. "
            "Balanced fertilizer decisions should follow both crop need and actual soil test values."
        ),
        "answer_mr": (
            "माती चाचणीमुळे pH आणि अन्नद्रव्यांची स्थिती समजते. "
            "नायट्रोजन पानांची वाढ, फॉस्फरस मुळे व फुलोरा, आणि पोटॅशियम जोम व ताण सहनशक्ती यासाठी उपयोगी असते. "
            "खताचा निर्णय घेताना पिकाची गरज आणि मातीतील प्रत्यक्ष मूल्ये दोन्ही पाहणे महत्त्वाचे आहे."
        ),
    },
    {
        "keywords": ["fertilizer", "basal", "top dress", "urea", "dap", "mop", "खत", "युरिया", "डीएपी", "बेसल"],
        "answer_en": (
            "Apply fertilizers in split doses when possible. "
            "Use basal fertilizer near sowing, then add top-dressing based on crop stage, moisture, and visible deficiency. "
            "Avoid using extra urea or phosphatic fertilizer blindly when soil values are already high."
        ),
        "answer_mr": (
            "शक्य असल्यास खताचे विभाजित डोस द्या. "
            "पेरणीच्या वेळी बेसल डोस द्या आणि नंतर वाढीच्या टप्प्यानुसार, ओलाव्यानुसार आणि कमतरतेच्या लक्षणांनुसार वरखत द्या. "
            "मातीमध्ये अन्नद्रव्ये आधीच जास्त असल्यास युरिया किंवा फॉस्फरसयुक्त खते अंधाधुंद देऊ नयेत."
        ),
    },
    {
        "keywords": ["irrigation", "water", "drip", "canal", "rainfed", "पाणी", "सिंचन", "ड्रिप", "कालवा", "पावसावर"],
        "answer_en": (
            "Irrigation planning should match crop stage and soil moisture, not only the calendar. "
            "Critical stages like germination, flowering, and grain or pod filling usually need more attention. "
            "Where possible, drip or controlled irrigation reduces waste and improves fertilizer efficiency."
        ),
        "answer_mr": (
            "सिंचनाचे नियोजन फक्त दिनदर्शिकेनुसार नव्हे तर पिकाच्या टप्प्यानुसार आणि मातीतील ओलाव्यानुसार करा. "
            "अंकुरण, फुलोरा आणि दाणे किंवा शेंगा भरण्याच्या काळात पाण्याकडे जास्त लक्ष द्यावे लागते. "
            "शक्य असल्यास ड्रिप किंवा नियंत्रित सिंचनाने पाण्याची बचत होते आणि खताचा उपयोगही चांगला होतो."
        ),
    },
    {
        "keywords": ["pest", "disease", "spray", "ipm", "कीड", "रोग", "फवारणी"],
        "answer_en": (
            "For pest and disease control, start with field scouting, remove badly affected parts, and avoid unnecessary spraying. "
            "Use Integrated Pest Management: clean seed, crop rotation, correct spacing, balanced fertilizer, and targeted spray only when needed. "
            "For exact chemical choice and dosage, match the crop, pest, and crop stage carefully."
        ),
        "answer_mr": (
            "कीड व रोग नियंत्रणासाठी आधी शेताची नियमित पाहणी करा, जास्त बाधित भाग काढून टाका आणि विनाकारण फवारणी करू नका. "
            "एकात्मिक व्यवस्थापन वापरा: स्वच्छ बियाणे, पीक फेरपालट, योग्य अंतर, संतुलित खत आणि गरज असल्यासच लक्षित फवारणी. "
            "अचूक औषध आणि डोस निवडताना पीक, कीड किंवा रोग आणि पिकाचा टप्पा काळजीपूर्वक जुळवणे गरजेचे आहे."
        ),
    },
    {
        "keywords": ["rotation", "intercrop", "companion", "फेरपालट", "आंतरपीक"],
        "answer_en": (
            "Crop rotation helps reduce pest pressure, improves soil structure, and can balance nutrient demand. "
            "Avoid growing the same crop repeatedly on the same field when possible. "
            "Adding pulses or legumes in rotation can help improve soil health."
        ),
        "answer_mr": (
            "पीक फेरपालट केल्याने कीडदाब कमी होतो, मातीची रचना सुधारते आणि अन्नद्रव्यांच्या गरजांचा समतोल राखण्यास मदत होते. "
            "शक्य असल्यास एकाच शेतात सतत तेच पीक घेणे टाळा. "
            "कडधान्यांचा समावेश केल्यास मातीच्या आरोग्यास मदत होऊ शकते."
        ),
    },
    {
        "keywords": ["organic", "compost", "fym", "manure", "शेणखत", "कंपोस्ट", "सेंद्रिय"],
        "answer_en": (
            "Organic matter like compost, FYM, or well-rotted manure improves soil structure, moisture holding, and long-term fertility. "
            "It usually works best when combined with measured fertilizer use rather than replacing all nutrients blindly."
        ),
        "answer_mr": (
            "कंपोस्ट, शेणखत किंवा चांगले कुजलेले सेंद्रिय खत मातीची रचना, पाणी धरून ठेवण्याची क्षमता आणि दीर्घकालीन सुपीकता सुधारते. "
            "बहुतेक वेळा ही खते मोजून दिलेल्या रासायनिक खतांसोबत वापरल्यास चांगला परिणाम मिळतो."
        ),
    },
    {
        "keywords": ["market", "mandi", "price", "profit", "बाजार", "मंडी", "भाव", "नफा"],
        "answer_en": (
            "Before selling, check the latest nearby mandi prices, expected transport cost, crop quality, and moisture content. "
            "A higher mandi price does not always mean better net profit if transport or quality deductions are high."
        ),
        "answer_mr": (
            "विक्रीपूर्वी जवळच्या मंडीतील ताजे भाव, वाहतूक खर्च, मालाची गुणवत्ता आणि ओलावा तपासा. "
            "मंडीचा भाव जास्त असला तरी वाहतूक किंवा गुणवत्ता कपात जास्त असल्यास निव्वळ नफा कमी होऊ शकतो."
        ),
    },
]


def _clean_text(value):
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _contains_phrase(cleaned_text, phrases):
    return any(phrase and phrase in cleaned_text for phrase in phrases)


def _contains_fuzzy_token(cleaned_text, tokens, threshold=0.82):
    message_words = [word for word in re.split(r"[^a-z0-9]+", cleaned_text) if word]
    for token in tokens:
        normalized = _clean_text(token)
        if not normalized:
            continue
        if normalized in cleaned_text:
            return True
        for word in message_words:
            if SequenceMatcher(None, word, normalized).ratio() >= threshold:
                return True
    return False


def _looks_like_weather_question(message):
    cleaned = _clean_text(message)
    weather_tokens = [
        "weather", "wether", "whether", "forecast", "temperature", "humidity",
        "rain", "rainfall", "climate", "today weather", "weather today",
        "havaman", "havaaman", "hawa", "paus", "pavas", "pavus", "paani padat", "barish"
    ]

    if _contains_phrase(cleaned, weather_tokens):
        return True

    if _contains_fuzzy_token(cleaned, ["weather", "forecast", "temperature", "humidity", "rain", "paus", "havaman"], threshold=0.78):
        return True

    mentions_time = any(token in cleaned for token in ["today", "todays", "today's", "now", "current", "aaj", "atta", "aj"])
    mentions_place = any(token in cleaned for token in ["location", "my location", "farm", "place", "city", "area", "ikade", "kade", "gav", "gaon"])

    if mentions_time and mentions_place:
        return True

    return any(token in cleaned for token in ["paus", "pavas", "pavus", "havaman", "havaaman"])


def _answer_weather_help(weather_snapshot, lang):
    weather_answer = _answer_weather(weather_snapshot, lang)
    if weather_answer:
        return weather_answer

    if lang == "mr":
        return (
            "तुमच्या सध्याच्या स्थानासाठी सेव्ह केलेले हवामान मला अजून दिसत नाही. "
            "Crop पेजवर location निवडा किंवा current location वापरा, हवामान fetch होऊ द्या, मग मी त्यावर उत्तर देऊ शकतो."
        )

    return (
        "I cannot see a saved weather snapshot for your current location yet. "
        "Open the Crop page, choose your location or use current location, let the weather load, and then I can answer using that data."
    )


def _translate_crop_name(crop_name, lang, crop_translations):
    if not crop_name:
        return "--"
    if lang == "mr" and crop_translations:
        return crop_translations.get(str(crop_name).lower(), crop_name)
    return crop_name


def _history_summary(lang, saved_farms, recent_predictions):
    if lang == "mr":
        if not saved_farms and not recent_predictions:
            return "आत्तापर्यंत सेव्ह केलेली शेतं किंवा अलीकडील prediction इतिहास उपलब्ध नाही."
        parts = []
        if saved_farms:
            farm_names = [str(item.get("name") or "शेत").strip() for item in saved_farms[:5]]
            parts.append("सेव्ह केलेली शेतं: " + ", ".join(farm_names))
        if recent_predictions:
            crops = [str(item.get("predictionRaw") or item.get("prediction") or "--").strip() for item in recent_predictions[:3]]
            parts.append("अलीकडील पीक शिफारसी: " + ", ".join(crops))
        return " | ".join(parts)

    if not saved_farms and not recent_predictions:
        return "I cannot see any saved farms or recent prediction history yet."

    parts = []
    if saved_farms:
        farm_names = [str(item.get("name") or "Farm").strip() for item in saved_farms[:5]]
        parts.append("Saved farms: " + ", ".join(farm_names))
    if recent_predictions:
        crops = [str(item.get("predictionRaw") or item.get("prediction") or "--").strip() for item in recent_predictions[:3]]
        parts.append("Recent crop recommendations: " + ", ".join(crops))
    return " | ".join(parts)


def _answer_from_last_crop(last_crop, lang, crop_translations):
    if not last_crop:
        return None

    crop_name = _translate_crop_name(last_crop.get("prediction_raw"), lang, crop_translations)
    season_name = last_crop.get("current_season") or "--"
    weather_city = last_crop.get("weather_city") or "--"
    estimated_profit = last_crop.get("estimated_profit")

    if lang == "mr":
        response = f"तुमची शेवटची पीक शिफारस {crop_name} होती."
        if season_name and season_name != "--":
            response += f" हंगाम: {season_name}."
        if weather_city and weather_city != "--":
            response += f" स्थान: {weather_city}."
        if estimated_profit:
            response += f" अंदाजित नफा: {estimated_profit}."
        return response

    response = f"Your last crop recommendation was {crop_name}."
    if season_name and season_name != "--":
        response += f" Season: {season_name}."
    if weather_city and weather_city != "--":
        response += f" Location: {weather_city}."
    if estimated_profit:
        response += f" Estimated profit: {estimated_profit}."
    return response


def _answer_from_last_fertilizer(last_plan, lang):
    if not last_plan:
        return None

    crop_name = last_plan.get("crop_name") or "--"
    summary = last_plan.get("summary") or ""
    quantity_lines = last_plan.get("quantity_lines") or []
    first_quantity = quantity_lines[0] if quantity_lines else ""

    if lang == "mr":
        response = f"तुमची शेवटची खत योजना {crop_name} साठी आहे."
        if summary:
            response += f" {summary}"
        if first_quantity:
            response += f" उदाहरण: {first_quantity}"
        return response

    response = f"Your last fertilizer plan is for {crop_name}."
    if summary:
        response += f" {summary}"
    if first_quantity:
        response += f" Example: {first_quantity}"
    return response


def _answer_saved_farms(saved_farms, lang):
    if not saved_farms:
        return "अजून कोणतेही सेव्ह केलेले शेत नाही." if lang == "mr" else "You do not have any saved farms yet."

    farm_lines = []
    for farm in saved_farms[:5]:
        name = str(farm.get("name") or ("शेत" if lang == "mr" else "Farm")).strip()
        place = str(farm.get("manual_location") or "").strip()
        lat = str(farm.get("latitude") or "").strip()
        lon = str(farm.get("longitude") or "").strip()

        if lang == "mr":
            detail = name
            if place:
                detail += f" ({place})"
            elif lat and lon:
                detail += f" ({lat}, {lon})"
        else:
            detail = name
            if place:
                detail += f" ({place})"
            elif lat and lon:
                detail += f" ({lat}, {lon})"
        farm_lines.append(detail)

    if lang == "mr":
        return "तुमची सेव्ह केलेली शेतं: " + ", ".join(farm_lines) + "."
    return "Your saved farms: " + ", ".join(farm_lines) + "."


def _answer_recent_predictions(recent_predictions, lang, crop_translations):
    if not recent_predictions:
        return "अजून कोणताही अलीकडील prediction इतिहास नाही." if lang == "mr" else "There is no recent prediction history yet."

    entries = []
    for item in recent_predictions[:3]:
        crop_name = _translate_crop_name(item.get("predictionRaw") or item.get("prediction"), lang, crop_translations)
        weather_city = item.get("weatherCity")
        if weather_city:
            entries.append(f"{crop_name} ({weather_city})")
        else:
            entries.append(crop_name)

    if lang == "mr":
        return "अलीकडील पीक शिफारसी: " + ", ".join(entries) + "."
    return "Your recent crop recommendations are: " + ", ".join(entries) + "."


def _answer_weather(weather_snapshot, lang):
    if not weather_snapshot:
        return None

    city = weather_snapshot.get("city") or "--"
    temperature = weather_snapshot.get("temperature")
    humidity = weather_snapshot.get("humidity")
    rainfall = weather_snapshot.get("rainfall")

    if lang == "mr":
        parts = [f"शेवटचे जतन केलेले हवामान स्थान: {city}."]
        if temperature not in (None, ""):
            parts.append(f"तापमान: {temperature}°C.")
        if humidity not in (None, ""):
            parts.append(f"आर्द्रता: {humidity}%.")
        if rainfall not in (None, ""):
            parts.append(f"पर्जन्यमान: {rainfall} mm.")
        return " ".join(parts)

    parts = [f"The latest saved weather snapshot is for {city}."]
    if temperature not in (None, ""):
        parts.append(f"Temperature: {temperature}°C.")
    if humidity not in (None, ""):
        parts.append(f"Humidity: {humidity}%.")
    if rainfall not in (None, ""):
        parts.append(f"Rainfall: {rainfall} mm.")
    return " ".join(parts)


def _answer_app_help(lang):
    if lang == "mr":
        return (
            "या अॅपमध्ये तुम्ही तीन मुख्य गोष्टी करू शकता: "
            "1. Crop पेजवर मातीची मूल्ये, स्थान आणि हवामान वापरून पीक शिफारस मिळवा. "
            "2. Fertilizer पेजवर NPK आणि पीक निवडून खत योजना मिळवा. "
            "3. सेव्ह केलेली शेतं आणि अलीकडील prediction इतिहास वापरून पुढचे निर्णय पटकन घ्या."
        )

    return (
        "In this app you can do three main things: "
        "1. Use the Crop page to get a crop recommendation from soil, location, and weather. "
        "2. Use the Fertilizer page to get a fertilizer plan from NPK values and crop choice. "
        "3. Reuse saved farms and recent prediction history for faster decisions."
    )


def _answer_specific_app_help(message, lang):
    cleaned = _clean_text(message)

    crop_help_tokens = [
        "crop prediction",
        "predict crop",
        "crop recommend",
        "crop recommendation",
        "how to get crop",
        "how do i get crop",
        "use crop page",
        "prediction in this website",
    ]
    fertilizer_help_tokens = [
        "fertilizer plan",
        "fertilizer recommendation",
        "how to use fertilizer",
        "use fertilizer page",
        "fertilizer page",
    ]
    website_help_tokens = [
        "website",
        "this website",
        "this app",
        "how to use",
        "how do i use",
        "how can i use",
    ]

    if any(token in cleaned for token in crop_help_tokens) or (
        "crop" in cleaned and any(token in cleaned for token in ["how", "use", "get", "predict", "prediction", "website", "app"])
    ):
        if lang == "mr":
            return (
                "या वेबसाइटवर crop prediction मिळवण्यासाठी Crop पेज उघडा, नंतर N, P, K, pH, जमीन क्षेत्र, पाण्याचा स्रोत आणि पाण्याची उपलब्धता भरा. "
                "त्यानंतर location निवडा, हवामान डेटा मिळवा आणि Predict बटण दाबा. मग सिस्टम तुम्हाला शिफारस केलेले पीक, top matches आणि स्पष्टीकरण दाखवेल."
            )
        return (
            "To get crop prediction on this website, open the Crop page, enter soil values like N, P, K and pH, add land area, water source, and water availability, then choose the location. "
            "After weather is fetched, click Predict. The system will show the recommended crop, top matches, and explanation."
        )

    if any(token in cleaned for token in fertilizer_help_tokens) or (
        "fertilizer" in cleaned and any(token in cleaned for token in ["page", "website", "app", "how to use", "use fertilizer"])
    ):
        if lang == "mr":
            return (
                "खत योजना मिळवण्यासाठी Fertilizer पेज उघडा, crop निवडा आणि NPK values भरा. "
                "त्यानंतर submit केल्यावर सिस्टम योग्य खत शिफारस, summary आणि quantity guidance दाखवेल."
            )
        return (
            "To get a fertilizer suggestion, open the Fertilizer page, choose the crop, enter the NPK values, and submit the form. "
            "The website will return a fertilizer recommendation, summary, and quantity guidance."
        )

    if any(token in cleaned for token in website_help_tokens):
        if lang == "mr":
            return (
                "या अॅपमध्ये तुम्ही crop prediction, fertilizer planning, saved farms reuse आणि basic farming help मिळवू शकता. "
                "तुम्ही मला असेही विचारू शकता: crop prediction कसे घ्यायचे, माझी saved farms दाखवा, किंवा एखाद्या पिकाची लागवड कशी करायची."
            )
        return (
            "In this website you can get crop prediction, fertilizer planning, saved farm reuse, and basic farming guidance. "
            "You can also ask me things like how to get crop prediction, show saved farms, or how to cultivate a crop."
        )

    return None


def _answer_direct_farming_help(message, lang):
    cleaned = _clean_text(message)

    if lang != "en":
        return None

    if any(token in cleaned for token in ["irrigation", "irrigate", "watering", "watering schedule", "how often should i water", "how often should i irrigate"]):
        return (
            "Irrigation should depend on crop stage, soil type, and moisture in the field, not only on a fixed calendar. "
            "Give more attention during establishment, flowering, and grain, fruit, or head formation. "
            "For exact advice, ask with the crop name, soil type, and current weather."
        )

    if any(token in cleaned for token in ["what fertilizer", "which fertilizer", "fertilizer should i use", "best fertilizer", "npk for", "manure for"]):
        return (
            "The right fertilizer depends on the crop, growth stage, and soil test. "
            "As a general rule, start with well-rotted organic matter and use balanced NPK based on crop need instead of applying extra urea blindly. "
            "If you tell me the crop and soil test values, I can guide you better."
        )

    if any(token in cleaned for token in ["aphid", "pest", "disease", "spray", "yellow leaves", "yellow leaf", "turning yellow", "yellowing", "leaves yellow", "wilting", "wilt", "leaf spot", "blight", "rot", "insect"]):
        return (
            "That can be caused by more than one issue, including pest attack, disease, nutrient deficiency, or water stress. "
            "First check the crop closely for insects, spots, curling, wilting, or root damage, and avoid spraying blindly. "
            "If you tell me the crop name and the exact symptom, I can guide you better."
        )

    if any(token in cleaned for token in ["black soil", "red soil", "clay soil", "loamy soil", "sandy soil", "soil type", "soil suitable", "ph"]):
        return (
            "Crop suitability depends on drainage, pH, organic matter, and water availability, not only the soil name. "
            "Black soil can suit many crops if drainage is managed well, while lighter soils may need more frequent moisture management. "
            "If you tell me the crop and soil type, I can give a more specific answer."
        )

    if any(token in cleaned for token in ["best season", "best time", "sowing time", "when should i sow", "when to sow"]):
        return (
            "The best season depends on the crop and your local climate. In general, Kharif crops are grown around the monsoon, Rabi crops after the monsoon in cooler months, and Zaid crops in the short summer period. "
            "If you tell me the crop name, I can narrow it down."
        )

    if any(token in cleaned for token in ["what is this website", "what does this website do", "what can this website do", "how does this website work"]):
        return (
            "This website helps with crop prediction, fertilizer planning, saved farm reuse, and farming guidance. "
            "Use the Crop page for soil plus weather based crop recommendation, the Fertilizer page for NPK-based fertilizer suggestions, and the assistant for project-specific farm questions."
        )

    return None


def _best_knowledge_answer(message, lang):
    cleaned = _clean_text(message)
    if not cleaned:
        return None

    best_entry = None
    best_score = 0.0

    for entry in FARM_KNOWLEDGE:
        keyword_score = 0.0
        for keyword in entry["keywords"]:
            normalized_keyword = _clean_text(keyword)
            if normalized_keyword and normalized_keyword in cleaned:
                keyword_score += 1.0
            else:
                keyword_score = max(keyword_score, SequenceMatcher(None, cleaned, normalized_keyword).ratio() * 0.6)

        if keyword_score > best_score:
            best_score = keyword_score
            best_entry = entry

    if best_score < 0.72:
        return None

    return best_entry["answer_mr"] if lang == "mr" else best_entry["answer_en"]


def _extract_crop_guide(message):
    cleaned = _clean_text(message)
    compact = re.sub(r"[^a-z0-9]+", "", cleaned)
    words = [word for word in re.split(r"[^a-z0-9]+", cleaned) if word]
    compact_chunks = []
    for start_index in range(len(words)):
        for size in range(1, 4):
            chunk_words = words[start_index:start_index + size]
            if len(chunk_words) != size:
                continue
            compact_chunks.append("".join(chunk_words))

    for guide in CROP_GUIDES.values():
        for alias in guide["aliases"]:
            normalized_alias = _clean_text(alias)
            compact_alias = re.sub(r"[^a-z0-9]+", "", normalized_alias)
            if normalized_alias and normalized_alias in cleaned:
                return guide
            if compact_alias and compact_alias in compact:
                return guide
            if compact_alias:
                for chunk in compact_chunks:
                    if chunk == compact_alias:
                        return guide
                    if len(chunk) >= max(5, len(compact_alias) - 2):
                        if SequenceMatcher(None, chunk, compact_alias).ratio() >= 0.82:
                            return guide
    return None


def _looks_like_cultivation_question(message):
    cleaned = _clean_text(message)
    cultivation_tokens = [
        "cultivate",
        "cultivation",
        "how to grow",
        "how to plant",
        "how to cultivate",
        "grow ",
        "plant ",
        "sow ",
        "farming of",
        "crop care",
        "harvest",
        "लागवड",
        "कशी करावी",
        "कसे करायचे",
        "कसं करायचं",
        "पीक कसे घ्यावे",
        "lagvad",
        "sheti",
        "ksheti",
        "kshetri",
        "kashe karayche",
        "kashi karavi",
        "karaychi aahe",
        "karaycha ahe",
        "karachi aahe",
        "karachi ahe",
        "karachi aaye",
        "karaychi aaye",
    ]
    if any(token in cleaned for token in cultivation_tokens):
        return True

    roman_marathi_patterns = [
        r"\bmala\s+.+\s+(?:sheti|ksheti|kshetri|lagvad)\s+(?:karaychi|karaycha|karachi)\s+(?:aahe|ahe|aaye)\b",
        r"\b.+\s+(?:sheti|ksheti|kshetri|lagvad)\s+(?:kashi|kase)\s+(?:karavi|karaychi|karayche)\b",
    ]
    return any(re.search(pattern, cleaned) for pattern in roman_marathi_patterns)


def _generic_cultivation_answer(message, lang):
    guide = _extract_crop_guide(message)
    if guide:
        return guide.get("answer_mr") if lang == "mr" else guide["answer_en"]

    cleaned = _clean_text(message)
    crop_phrase = None
    patterns = [
        r"how to (?:grow|cultivate|plant)\s+([a-z ]+)",
        r"(?:cultivation|farming)\s+of\s+([a-z ]+)",
        r"([a-z ]+)\s+cultivation",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            crop_phrase = re.sub(r"\s+", " ", match.group(1)).strip(" ?.,")
            if crop_phrase:
                break

    if lang == "mr":
        crop_label = crop_phrase or "या पिकासाठी"
        return (
            f"{crop_label} योग्य हंगाम, हवामान आणि चांगला निचरा होणारी जमीन निवडा. "
            "निरोगी बियाणे किंवा रोपे वापरा, योग्य अंतर ठेवा, सेंद्रिय खत व संतुलित अन्नद्रव्ये द्या, "
            "मातीतील ओलाव्यानुसार सिंचन करा आणि तण, कीड व रोग यावर सुरुवातीपासून लक्ष ठेवा. "
            "अधिक अचूक मार्गदर्शन हवे असल्यास पिकाचे नाव, हंगाम, मातीचा प्रकार किंवा लक्षणे लिहा."
        )

    crop_label = crop_phrase or "this crop"
    return (
        f"For {crop_label}, start with the right season, climate, and well-drained soil. "
        "Use healthy seed or seedlings, maintain proper spacing, add organic matter with balanced nutrients, "
        "give irrigation based on soil moisture instead of overwatering, and monitor weeds, pests, and disease early. "
        "If you want a more exact answer, ask with the crop name, season, soil type, or symptoms."
    )


def _fallback_answer(lang, history_summary):
    if lang == "mr":
        base = "मी या अॅपमधील तुमचा इतिहास, सेव्ह केलेली शेतं, अलीकडील prediction आणि शेवटची खत योजना वापरून मदत करू शकतो."
        if history_summary:
            base += " " + history_summary
        base += " तुम्ही पीक, खत, हवामान, हंगाम, सिंचन, माती चाचणी किंवा या अॅपचा वापर याबद्दल प्रश्न विचारू शकता."
        return base

    base = "I can help with this project's farming guidance, crop prediction flow, fertilizer use, weather, soil, irrigation, pests, and how to use the website."
    if history_summary:
        base += " " + history_summary
    base += " Ask about crop cultivation, fertilizer choice, weather at your location, symptoms, pests, soil, seasons, or website usage. If needed, rephrase the question with more detail and I will try again."
    return base


def generate_farm_chat_reply(message, lang, session_context=None, client_context=None, crop_translations=None):
    session_context = session_context or {}
    client_context = client_context or {}
    crop_translations = crop_translations or {}

    cleaned_message = _clean_text(message)

    saved_farms = list(client_context.get("savedFarms") or [])[:5]
    recent_predictions = list(client_context.get("recentPredictions") or [])[:5]
    weather_snapshot = client_context.get("latestWeather") or {}
    last_crop = session_context.get("last_crop_result") or {}
    last_fertilizer_plan = session_context.get("last_fertilizer_plan") or {}

    history_summary = _history_summary(lang, saved_farms, recent_predictions)

    if not cleaned_message:
        return {
            "reply": _fallback_answer(lang, history_summary),
            "suggestions": DEFAULT_SUGGESTIONS[lang],
        }

    if "reply in marathi" in cleaned_message or "answer in marathi" in cleaned_message or "respond in marathi" in cleaned_message:
        return {
            "reply": "हो, आता पुढे तुम्ही विचारलेल्या प्रश्नांना मी मराठीत उत्तर देईन.",
            "suggestions": DEFAULT_SUGGESTIONS["mr"],
        }

    if "reply in english" in cleaned_message or "answer in english" in cleaned_message or "respond in english" in cleaned_message:
        return {
            "reply": "Okay, I will answer your next questions in English.",
            "suggestions": DEFAULT_SUGGESTIONS["en"],
        }

    greeting_words = ["hi", "hello", "hey", "namaste", "नमस्कार", "हॅलो", "hello"]
    if any(word == cleaned_message for word in greeting_words):
        if lang == "mr":
            return {
                "reply": "नमस्कार. मी तुमचा मोफत शेत सहाय्यक आहे. मी तुमची सेव्ह केलेली शेतं, अलीकडील prediction आणि शेवटची खत योजना लक्षात ठेवून मदत करू शकतो.",
                "suggestions": DEFAULT_SUGGESTIONS[lang],
            }
        return {
            "reply": "Hello. I am your free farm assistant. I can use your saved farms, recent predictions, and latest fertilizer plan to answer questions inside this app.",
            "suggestions": DEFAULT_SUGGESTIONS[lang],
        }

    if any(token in cleaned_message for token in ["saved farm", "saved farms", "my farm", "my farms", "सेव्ह", "शेतं", "शेत"]):
        return {
            "reply": _answer_saved_farms(saved_farms, lang),
            "suggestions": DEFAULT_SUGGESTIONS[lang],
        }

    if any(token in cleaned_message for token in ["recent", "history", "last prediction", "recent prediction", "इतिहास", "अलीकडील", "शेवटची शिफारस"]):
        last_crop_answer = _answer_from_last_crop(last_crop, lang, crop_translations)
        if last_crop_answer:
            reply = last_crop_answer + " " + _answer_recent_predictions(recent_predictions, lang, crop_translations)
        else:
            reply = _answer_recent_predictions(recent_predictions, lang, crop_translations)
        return {
            "reply": reply,
            "suggestions": DEFAULT_SUGGESTIONS[lang],
        }

    if any(token in cleaned_message for token in ["last crop", "recommended crop", "crop recommendation", "पीक शिफारस", "शेवटचे पीक", "last recommended crop"]):
        last_crop_answer = _answer_from_last_crop(last_crop, lang, crop_translations)
        return {
            "reply": last_crop_answer or (_fallback_answer(lang, history_summary)),
            "suggestions": DEFAULT_SUGGESTIONS[lang],
        }

    if any(token in cleaned_message for token in ["fertilizer plan", "last fertilizer", "fertilizer recommendation", "खत योजना", "खत शिफारस", "शेवटची खत योजना"]):
        last_fertilizer_answer = _answer_from_last_fertilizer(last_fertilizer_plan, lang)
        return {
            "reply": last_fertilizer_answer or (_fallback_answer(lang, history_summary)),
            "suggestions": DEFAULT_SUGGESTIONS[lang],
        }

    if any(token in cleaned_message for token in ["weather", "rain", "temperature", "humidity", "हवामान", "पाऊस", "तापमान", "आर्द्रता"]):
        weather_answer = _answer_weather(weather_snapshot, lang)
        if weather_answer:
            return {
                "reply": weather_answer,
                "suggestions": DEFAULT_SUGGESTIONS[lang],
            }

    if any(token in cleaned_message for token in ["help", "how to use", "use this app", "app help", "मदत", "कसे वापरायचे", "अॅप कसे वापरायचे"]):
        return {
            "reply": _answer_app_help(lang),
            "suggestions": DEFAULT_SUGGESTIONS[lang],
        }

    if _looks_like_weather_question(message):
        return {
            "reply": _answer_weather_help(weather_snapshot, lang),
            "suggestions": DEFAULT_SUGGESTIONS[lang],
        }

    specific_help_answer = _answer_specific_app_help(message, lang)
    if specific_help_answer:
        return {
            "reply": specific_help_answer,
            "suggestions": DEFAULT_SUGGESTIONS[lang],
        }

    direct_farming_answer = _answer_direct_farming_help(message, lang)
    if direct_farming_answer:
        return {
            "reply": direct_farming_answer,
            "suggestions": DEFAULT_SUGGESTIONS[lang],
        }

    crop_guide = _extract_crop_guide(message)
    if crop_guide and any(token in cleaned_message for token in ["mala", "sheti", "ksheti", "kshetri", "lagvad", "farming", "cultivation"]):
        return {
            "reply": crop_guide.get("answer_mr") if lang == "mr" else crop_guide["answer_en"],
            "suggestions": DEFAULT_SUGGESTIONS[lang],
        }

    if _looks_like_cultivation_question(message):
        return {
            "reply": _generic_cultivation_answer(message, lang),
            "suggestions": DEFAULT_SUGGESTIONS[lang],
        }

    knowledge_answer = _best_knowledge_answer(message, lang)
    if knowledge_answer:
        return {
            "reply": knowledge_answer,
            "suggestions": DEFAULT_SUGGESTIONS[lang],
        }

    return {
        "reply": _fallback_answer(lang, history_summary),
        "suggestions": DEFAULT_SUGGESTIONS[lang],
    }
