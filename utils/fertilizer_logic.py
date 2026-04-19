import pandas as pd

from utils.fertilizer import fertilizer_dic, fertilizer_dic_mr


BALANCED_FERTILIZER_KEY = "Balanced"


NITROGEN_ADVICE = {
    "en": {
        "NHigh": """The nitrogen (N) value in your soil is higher than the crop's ideal range.
<br/> Please consider the following suggestions:
<br/><br/>1. Reduce or pause high-nitrogen fertilizers such as urea or ammonium-based products.
<br/>2. Avoid adding fresh manure until the nitrogen level comes closer to the crop requirement.
<br/>3. Irrigate carefully so excess soluble nitrogen can move deeper into the soil profile when appropriate.
<br/>4. Prefer balanced or low-nitrogen fertilizer grades for the next application cycle.
<br/>5. Re-test the soil before the next top-dressing so you do not over-correct.""",
        "Nlow": """The nitrogen (N) value in your soil is lower than the crop's ideal range.
<br/> Please consider the following suggestions:
<br/><br/>1. Apply a nitrogen-rich fertilizer such as urea, ammonium sulfate, or another suitable N source.
<br/>2. Add well-decomposed compost or manure to improve nitrogen availability gradually.
<br/>3. Split nitrogen application into smaller doses to improve uptake and reduce loss.
<br/>4. Keep adequate soil moisture after application so the crop can absorb the added nitrogen.
<br/>5. Re-check soil values after correction to avoid applying more nitrogen than needed.""",
        BALANCED_FERTILIZER_KEY: """Your soil nitrogen (N), phosphorus (P), and potassium (K) values are already close to this crop's reference range.
<br/> Please consider the following suggestions:
<br/><br/>1. Use a balanced maintenance fertilizer only if your agronomy plan requires it.
<br/>2. Avoid heavy corrective fertilizer doses right now because no major nutrient gap is detected.
<br/>3. Continue normal irrigation and crop-stage based nutrient management.
<br/>4. Re-test the soil before the next major fertilizer application.""",
    },
    "mr": {
        "NHigh": """\u0924\u0941\u092e\u091a\u094d\u092f\u093e \u092e\u093e\u0924\u0940\u0924\u0940\u0932 \u0928\u093e\u092f\u091f\u094d\u0930\u094b\u091c\u0928 (N) \u092a\u0940\u0915\u093e\u0938\u093e\u0920\u0940 \u0906\u0926\u0930\u094d\u0936 \u092a\u093e\u0924\u0933\u0940\u092a\u0947\u0915\u094d\u0937\u093e \u091c\u093e\u0938\u094d\u0924 \u0906\u0939\u0947.
<br/> \u0915\u0943\u092a\u092f\u093e \u0916\u093e\u0932\u0940\u0932 \u0938\u0942\u091a\u0928\u093e \u0935\u093f\u091a\u093e\u0930\u093e\u0924 \u0918\u094d\u092f\u093e:
<br/><br/>1. \u092f\u0941\u0930\u093f\u092f\u093e \u0915\u093f\u0902\u0935\u093e \u0905\u092e\u094b\u0928\u093f\u092f\u092e-\u0906\u0927\u093e\u0930\u093f\u0924 \u0916\u0924\u093e\u0902\u091a\u093e \u0935\u093e\u092a\u0930 \u0915\u092e\u0940 \u0915\u0930\u093e \u0915\u093f\u0902\u0935\u093e \u0924\u093e\u0924\u094d\u092a\u0941\u0930\u0924\u093e \u0925\u093e\u0902\u092c\u0935\u093e.
<br/>2. \u0928\u093e\u092f\u091f\u094d\u0930\u094b\u091c\u0928 \u091c\u093e\u0938\u094d\u0924 \u0905\u0938\u0932\u0947\u0932\u0947 \u0924\u093e\u091c\u0947 \u0936\u0947\u0923\u0916\u0924 \u0938\u0927\u094d\u092f\u093e \u091f\u093e\u0933\u093e.
<br/>3. \u092a\u093e\u0923\u094d\u092f\u093e\u091a\u0947 \u092f\u094b\u0917\u094d\u092f \u0928\u093f\u092f\u094b\u091c\u0928 \u0915\u0930\u093e \u091c\u0947\u0923\u0947\u0915\u0930\u0942\u0928 \u0935\u093f\u0930\u0918\u0933\u0923\u093e\u0930\u0947 \u0928\u093e\u092f\u091f\u094d\u0930\u094b\u091c\u0928 \u092e\u093e\u0924\u0940\u0924 \u0916\u093e\u0932\u0940 \u0938\u0930\u0915\u0947\u0932.
<br/>4. \u092a\u0941\u0922\u0940\u0932 \u0921\u094b\u0938\u0938\u093e\u0920\u0940 \u0938\u0902\u0924\u0941\u0932\u093f\u0924 \u0915\u093f\u0902\u0935\u093e \u0915\u092e\u0940-\u0928\u093e\u092f\u091f\u094d\u0930\u094b\u091c\u0928 \u0916\u0924 \u0935\u093e\u092a\u0930\u093e.
<br/>5. \u092a\u0941\u0922\u091a\u094d\u092f\u093e \u0916\u0924 \u0935\u093e\u092a\u0930\u093e\u092a\u0942\u0930\u094d\u0935\u0940 \u092e\u093e\u0924\u0940 \u092a\u0941\u0928\u094d\u0939\u093e \u0924\u092a\u093e\u0938\u093e.""",
        "Nlow": """\u0924\u0941\u092e\u091a\u094d\u092f\u093e \u092e\u093e\u0924\u0940\u0924\u0940\u0932 \u0928\u093e\u092f\u091f\u094d\u0930\u094b\u091c\u0928 (N) \u092a\u0940\u0915\u093e\u0938\u093e\u0920\u0940 \u0906\u0926\u0930\u094d\u0936 \u092a\u093e\u0924\u0933\u0940\u092a\u0947\u0915\u094d\u0937\u093e \u0915\u092e\u0940 \u0906\u0939\u0947.
<br/> \u0915\u0943\u092a\u092f\u093e \u0916\u093e\u0932\u0940\u0932 \u0938\u0942\u091a\u0928\u093e \u0935\u093f\u091a\u093e\u0930\u093e\u0924 \u0918\u094d\u092f\u093e:
<br/><br/>1. \u092f\u0941\u0930\u093f\u092f\u093e, \u0905\u092e\u094b\u0928\u093f\u092f\u092e \u0938\u0932\u094d\u092b\u0947\u091f \u0915\u093f\u0902\u0935\u093e \u0907\u0924\u0930 \u0928\u093e\u092f\u091f\u094d\u0930\u094b\u091c\u0928-\u092f\u0941\u0915\u094d\u0924 \u0916\u0924\u093e\u0902\u091a\u093e \u0935\u093e\u092a\u0930 \u0915\u0930\u093e.
<br/>2. \u0915\u0941\u091c\u0932\u0947\u0932\u0947 \u0936\u0947\u0923\u0916\u0924 \u0915\u093f\u0902\u0935\u093e \u0915\u0902\u092a\u094b\u0938\u094d\u091f \u092e\u093e\u0924\u0940\u0924 \u092e\u093f\u0938\u0933\u093e.
<br/>3. \u0928\u093e\u092f\u091f\u094d\u0930\u094b\u091c\u0928\u091a\u093e \u0921\u094b\u0938 \u090f\u0915\u0926\u092e \u0926\u0947\u0923\u094d\u092f\u093e\u090f\u0935\u091c\u0940 \u091f\u092a\u094d\u092a\u094d\u092f\u093e\u091f\u092a\u094d\u092a\u094d\u092f\u093e\u0928\u0947 \u0926\u094d\u092f\u093e.
<br/>4. \u0916\u0924 \u0926\u0947\u0924\u093e\u0928\u093e \u092e\u093e\u0924\u0940\u0924 \u092a\u0941\u0930\u0947\u0938\u0940 \u0906\u0930\u094d\u0926\u094d\u0930\u0924\u093e \u0920\u0947\u0935\u093e \u091c\u0947\u0923\u0947\u0915\u0930\u0942\u0928 \u092a\u0940\u0915 \u0928\u093e\u092f\u091f\u094d\u0930\u094b\u091c\u0928 \u0936\u094b\u0937\u0942 \u0936\u0915\u0947\u0932.
<br/>5. \u0938\u0941\u0927\u093e\u0930\u0923\u093e \u0915\u0947\u0932\u094d\u092f\u093e\u0928\u0902\u0924\u0930 \u092e\u093e\u0924\u0940 \u092a\u0941\u0928\u094d\u0939\u093e \u0924\u092a\u093e\u0938\u093e \u091c\u0947\u0923\u0947\u0915\u0930\u0942\u0928 \u0905\u0924\u093f\u0930\u093f\u0915\u094d\u0924 \u0916\u0924 \u091f\u093e\u0933\u0924\u093e \u092f\u0947\u0908\u0932.""",
        BALANCED_FERTILIZER_KEY: """\u0924\u0941\u092e\u091a\u094d\u092f\u093e \u092e\u093e\u0924\u0940\u0924\u0940\u0932 N, P \u0906\u0923\u093f K \u092e\u0942\u0932\u094d\u092f\u0947 \u092f\u093e \u092a\u093f\u0915\u093e\u0938\u093e\u0920\u0940 \u0938\u0902\u0926\u0930\u094d\u092d \u092a\u093e\u0924\u0933\u0940\u091c\u0935\u0933 \u0906\u0939\u0947\u0924.
<br/> \u0915\u0943\u092a\u092f\u093e \u0916\u093e\u0932\u0940\u0932 \u0938\u0942\u091a\u0928\u093e \u0935\u093f\u091a\u093e\u0930\u093e\u0924 \u0918\u094d\u092f\u093e:
<br/><br/>1. \u0924\u0941\u092e\u091a\u094d\u092f\u093e \u0915\u0943\u0937\u0940 \u092f\u094b\u091c\u0928\u0947\u0928\u0941\u0938\u093e\u0930 \u0917\u0930\u091c \u0905\u0938\u0947\u0932 \u0924\u0930\u091a \u0938\u0902\u0924\u0941\u0932\u093f\u0924 \u0926\u0947\u0916\u092d\u093e\u0932 \u0916\u0924 \u0935\u093e\u092a\u0930\u093e.
<br/>2. \u0938\u0927\u094d\u092f\u093e \u0915\u094b\u0923\u0924\u0940\u0939\u0940 \u092e\u094b\u0920\u0940 \u0938\u0941\u0927\u093e\u0930\u0923\u093e\u0924\u094d\u092e\u0915 \u0916\u0924 \u0921\u094b\u0938 \u0926\u0947\u0923\u094d\u092f\u093e\u091a\u0940 \u0917\u0930\u091c \u0926\u093f\u0938\u0924 \u0928\u093e\u0939\u0940.
<br/>3. \u092a\u0940\u0915\u093e\u091a\u094d\u092f\u093e \u091f\u092a\u094d\u092a\u094d\u092f\u093e\u0928\u0941\u0938\u093e\u0930 \u0938\u093e\u092e\u093e\u0928\u094d\u092f \u0905\u0928\u094d\u0928\u0926\u094d\u0930\u0935\u094d\u092f \u0935\u094d\u092f\u0935\u0938\u094d\u0925\u093e\u092a\u0928 \u0938\u0941\u0930\u0942 \u0920\u0947\u0935\u093e.
<br/>4. \u092a\u0941\u0922\u091a\u094d\u092f\u093e \u092e\u094b\u0920\u094d\u092f\u093e \u0916\u0924 \u0935\u093e\u092a\u0930\u093e\u092a\u0942\u0930\u094d\u0935\u0940 \u092e\u093e\u0924\u0940 \u092a\u0941\u0928\u094d\u0939\u093e \u0924\u092a\u093e\u0938\u093e.""",
    },
}


def get_fertilizer_recommendation_key(reference_n, reference_p, reference_k, actual_n, actual_p, actual_k):
    deltas = [
        ("N", actual_n - reference_n),
        ("P", actual_p - reference_p),
        ("K", actual_k - reference_k),
    ]

    if all(delta == 0 for _, delta in deltas):
        return BALANCED_FERTILIZER_KEY

    dominant_nutrient, dominant_delta = deltas[0]
    for nutrient, delta in deltas[1:]:
        if abs(delta) > abs(dominant_delta):
            dominant_nutrient, dominant_delta = nutrient, delta

    if dominant_nutrient == "N":
        return "NHigh" if dominant_delta > 0 else "Nlow"
    if dominant_nutrient == "P":
        return "PHigh" if dominant_delta > 0 else "Plow"
    return "KHigh" if dominant_delta > 0 else "Klow"


def get_fertilizer_recommendation_text(recommendation_key, lang):
    normalized_lang = "mr" if lang == "mr" else "en"

    if recommendation_key in NITROGEN_ADVICE[normalized_lang]:
        return NITROGEN_ADVICE[normalized_lang][recommendation_key]

    source = fertilizer_dic_mr if normalized_lang == "mr" else fertilizer_dic
    return source[recommendation_key]


def build_fertilizer_reference_df(fertilizer_path, crop_reference_path):
    fertilizer_df = pd.read_csv(fertilizer_path)[["Crop", "N", "P", "K"]].copy()
    fertilizer_df["Crop"] = fertilizer_df["Crop"].astype(str).str.strip().str.lower()
    fertilizer_df = fertilizer_df.dropna(subset=["Crop"]).drop_duplicates(subset=["Crop"], keep="first")

    crop_df = pd.read_csv(crop_reference_path)[["label", "N", "P", "K"]].copy()
    crop_df["label"] = crop_df["label"].astype(str).str.strip().str.lower()
    crop_reference_df = (
        crop_df.groupby("label", as_index=False)[["N", "P", "K"]]
        .mean()
        .round()
        .rename(columns={"label": "Crop"})
    )

    merged_df = crop_reference_df.merge(
        fertilizer_df,
        on="Crop",
        how="left",
        suffixes=("_crop", "_fert")
    )

    for nutrient in ["N", "P", "K"]:
        merged_df[nutrient] = (
            merged_df["{0}_fert".format(nutrient)]
            .fillna(merged_df["{0}_crop".format(nutrient)])
            .round()
            .astype(int)
        )

    return merged_df[["Crop", "N", "P", "K"]].sort_values("Crop").reset_index(drop=True)


CROP_FAMILY = {
    "bajra": "cereal",
    "jowar": "cereal",
    "maize": "cereal",
    "ragi": "cereal",
    "rice": "cereal",
    "wheat": "cereal",
    "blackgram": "pulse",
    "cowpea": "pulse",
    "gram": "pulse",
    "lentil": "pulse",
    "mungbean": "pulse",
    "tur": "pulse",
    "groundnut": "oilseed",
    "sesame": "oilseed",
    "soybean": "oilseed",
    "sunflower": "oilseed",
    "brinjal": "vegetable",
    "cabbage": "vegetable",
    "chili": "vegetable",
    "onion": "vegetable",
    "potato": "vegetable",
    "tomato": "vegetable",
    "banana": "fruit",
    "grapes": "fruit",
    "orange": "fruit",
    "pomegranate": "fruit",
    "ginger": "rhizome",
    "turmeric": "rhizome",
    "cotton": "cash",
    "sugarcane": "cash",
}


FAMILY_MULTIPLIER = {
    "cereal": 1.0,
    "pulse": 0.8,
    "oilseed": 0.9,
    "vegetable": 1.15,
    "fruit": 1.2,
    "rhizome": 1.2,
    "cash": 1.25,
}


UREA_STAGE_SPLITS = {
    "cereal": (0.25, 0.5, 0.25),
    "pulse": (0.2, 0.45, 0.35),
    "oilseed": (0.25, 0.45, 0.3),
    "vegetable": (0.3, 0.4, 0.3),
    "fruit": (0.35, 0.35, 0.3),
    "rhizome": (0.35, 0.4, 0.25),
    "cash": (0.3, 0.45, 0.25),
}


P_STAGE_SPLITS = (0.75, 0.25, 0.0)
K_STAGE_SPLITS = (0.5, 0.2, 0.3)


COMPOST_PER_ACRE = {
    "cereal": 1000,
    "pulse": 800,
    "oilseed": 900,
    "vegetable": 1500,
    "fruit": 2000,
    "rhizome": 1800,
    "cash": 1800,
}


DISPLAY_NAME_OVERRIDES = {
    "bajra": "Bajra",
    "jowar": "Jowar",
    "ragi": "Ragi",
    "tur": "Tur",
}

DISPLAY_NAME_OVERRIDES_MR = {
    "bajra": "बाजरी",
    "banana": "केळी",
    "blackgram": "उडीद",
    "brinjal": "वांगी",
    "cabbage": "कोबी",
    "chili": "मिरची",
    "cotton": "कापूस",
    "cowpea": "चवळी",
    "ginger": "आले",
    "gram": "हरभरा",
    "grapes": "द्राक्ष",
    "groundnut": "शेंगदाणे",
    "jowar": "ज्वारी",
    "lentil": "मसूर",
    "maize": "मका",
    "mungbean": "मूग",
    "onion": "कांदा",
    "orange": "संत्री",
    "pomegranate": "डाळिंब",
    "potato": "बटाटा",
    "ragi": "नाचणी",
    "rice": "तांदूळ",
    "sesame": "तीळ",
    "soybean": "सोयाबीन",
    "sugarcane": "ऊस",
    "sunflower": "सूर्यफूल",
    "tomato": "टोमॅटो",
    "tur": "तूर",
    "turmeric": "हळद",
    "wheat": "गहू",
}


def _crop_family(crop_name):
    return CROP_FAMILY.get(str(crop_name).strip().lower(), "cereal")


def _crop_display_name(crop_name, lang="en"):
    normalized = str(crop_name).strip().lower()
    if lang == "mr" and normalized in DISPLAY_NAME_OVERRIDES_MR:
        return DISPLAY_NAME_OVERRIDES_MR[normalized]
    if normalized in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[normalized]
    return normalized.replace("_", " ").title()


def _round_kg(value):
    return int(round(value))


def _dose_from_gap(gap, nutrient_code, crop_family):
    if gap <= 0:
        return 0

    multiplier = FAMILY_MULTIPLIER[crop_family]

    if nutrient_code == "N":
        base = 15 if gap <= 10 else 30 if gap <= 25 else 45
    elif nutrient_code == "P_DAP":
        base = 10 if gap <= 10 else 20 if gap <= 25 else 30
    elif nutrient_code == "P_SSP":
        base = 25 if gap <= 10 else 50 if gap <= 25 else 75
    else:
        base = 8 if gap <= 10 else 15 if gap <= 25 else 25

    return max(_round_kg(base * multiplier), 0)


def _split_total(total, splits):
    first = _round_kg(total * splits[0])
    second = _round_kg(total * splits[1])
    third = max(total - first - second, 0)
    return first, second, third


def _nutrient_status_lines(actual_n, actual_p, actual_k, reference_n, reference_p, reference_k, lang="en"):
    status_lines = []
    values = [
        ("Nitrogen", actual_n, reference_n),
        ("Phosphorous", actual_p, reference_p),
        ("Potassium", actual_k, reference_k),
    ]

    for name, actual, reference in values:
        display_name = {
            "Nitrogen": "नायट्रोजन",
            "Phosphorous": "फॉस्फरस",
            "Potassium": "पोटॅशियम",
        }.get(name, name) if lang == "mr" else name
        difference = actual - reference
        if abs(difference) <= 5:
            status_lines.append(
                "{0}: माती {1}, पीक लक्ष्य {2} (लक्ष्याच्या जवळ).".format(display_name, actual, reference)
                if lang == "mr"
                else "{0}: soil {1}, crop target {2} (close to target).".format(display_name, actual, reference)
            )
        elif difference > 0:
            status_lines.append(
                "{0}: माती {1}, पीक लक्ष्य {2} (जास्त {3}).".format(display_name, actual, reference, difference)
                if lang == "mr"
                else "{0}: soil {1}, crop target {2} (higher by {3}).".format(display_name, actual, reference, difference)
            )
        else:
            status_lines.append(
                "{0}: माती {1}, पीक लक्ष्य {2} (कमी {3}).".format(display_name, actual, reference, abs(difference))
                if lang == "mr"
                else "{0}: soil {1}, crop target {2} (lower by {3}).".format(display_name, actual, reference, abs(difference))
            )

    return status_lines


def _major_issue_text(crop_display_name, recommendation_key, lang="en"):
    if lang == "mr":
        if recommendation_key == "NHigh":
            return "{0} साठी मुख्य अन्नद्रव्य मुद्दा: नायट्रोजन आधीच लक्ष्यापेक्षा जास्त आहे, त्यामुळे अतिरिक्त नायट्रोजन मर्यादित ठेवा.".format(crop_display_name)
        if recommendation_key == "Nlow":
            return "{0} साठी मुख्य अन्नद्रव्य मुद्दा: नायट्रोजन लक्ष्यापेक्षा कमी आहे, त्यामुळे योजनेत नायट्रोजन पूरकतेला प्राधान्य दिले आहे.".format(crop_display_name)
        if recommendation_key == "PHigh":
            return "{0} साठी मुख्य अन्नद्रव्य मुद्दा: फॉस्फरस आधीच जास्त आहे, त्यामुळे फॉस्फरसयुक्त खत सध्या टाळावे.".format(crop_display_name)
        if recommendation_key == "Plow":
            return "{0} साठी मुख्य अन्नद्रव्य मुद्दा: फॉस्फरस कमी आहे, त्यामुळे सुरुवातीच्या मुळांच्या वाढीसाठी आणि बेसल डोससाठी फॉस्फरसवर भर दिला आहे.".format(crop_display_name)
        if recommendation_key == "KHigh":
            return "{0} साठी मुख्य अन्नद्रव्य मुद्दा: पोटॅशियम आधीच जास्त आहे, त्यामुळे अतिरिक्त पोटॅश कमी ठेवला आहे.".format(crop_display_name)
        if recommendation_key == "Klow":
            return "{0} साठी मुख्य अन्नद्रव्य मुद्दा: पोटॅशियम कमी आहे, त्यामुळे खोडाची ताकद, फुलोरा आणि उत्पादनासाठी पूरक डोस सुचवला आहे.".format(crop_display_name)
        return "{0} साठी मोठा अन्नद्रव्य असमतोल आढळला नाही. हलकी देखभाल योजना पुरेशी आहे; जड सुधारक डोस टाळा.".format(crop_display_name)

    if recommendation_key == "NHigh":
        return "Main nutrient concern for {0}: nitrogen is already above the crop target, so extra nitrogen should be restricted.".format(crop_display_name)
    if recommendation_key == "Nlow":
        return "Main nutrient concern for {0}: nitrogen is below the crop target, so the plan gives priority to nitrogen support.".format(crop_display_name)
    if recommendation_key == "PHigh":
        return "Main nutrient concern for {0}: phosphorus is already high, so phosphorus-heavy fertilizers should be avoided for now.".format(crop_display_name)
    if recommendation_key == "Plow":
        return "Main nutrient concern for {0}: phosphorus is low, so the plan strengthens early root and basal phosphorus support.".format(crop_display_name)
    if recommendation_key == "KHigh":
        return "Main nutrient concern for {0}: potassium is already high, so additional potash should be minimized.".format(crop_display_name)
    if recommendation_key == "Klow":
        return "Main nutrient concern for {0}: potassium is low, so the plan improves stem strength, flowering, and grain or fruit support.".format(crop_display_name)
    return "No major nutrient imbalance was detected for {0}. Use a light maintenance schedule and avoid heavy corrective doses.".format(crop_display_name)


def build_fertilizer_plan(crop_name, reference_n, reference_p, reference_k, actual_n, actual_p, actual_k, lang="en"):
    normalized_lang = "mr" if lang == "mr" else "en"
    crop_family = _crop_family(crop_name)
    crop_display_name = _crop_display_name(crop_name, normalized_lang)
    recommendation_key = get_fertilizer_recommendation_key(
        reference_n=reference_n,
        reference_p=reference_p,
        reference_k=reference_k,
        actual_n=actual_n,
        actual_p=actual_p,
        actual_k=actual_k
    )

    n_gap = max(reference_n - actual_n, 0)
    p_gap = max(reference_p - actual_p, 0)
    k_gap = max(reference_k - actual_k, 0)

    compost_qty = COMPOST_PER_ACRE[crop_family]
    if normalized_lang == "mr":
        basal_lines = [
            "प्रती एकर {0} किलो चांगले कुजलेले शेणखत किंवा कंपोस्ट द्या आणि पेरणी किंवा लागवडीपूर्वी मातीमध्ये मिसळा.".format(compost_qty)
        ]
        quantity_lines = ["शेणखत किंवा कंपोस्ट: {0} किलो/एकर".format(compost_qty)]
    else:
        basal_lines = [
            "Apply {0} kg well-decomposed FYM or compost per acre and mix it into soil before sowing or transplanting.".format(compost_qty)
        ]
        quantity_lines = ["FYM or compost: {0} kg/acre".format(compost_qty)]

    vegetative_lines = []
    flowering_lines = []
    organic_lines = []

    if recommendation_key == BALANCED_FERTILIZER_KEY:
        basal_lines.append(
            "नीम केक 25 किलो/एकर हलक्या बेसल देखभाल डोस म्हणून द्या."
            if normalized_lang == "mr"
            else "Apply neem cake 25 kg/acre as a mild basal maintenance input."
        )
        vegetative_lines.append(
            "वाढीचा वेग कमी दिसल्यासच 19:19:19 पाण्यात विरघळणारे खत जास्तीत जास्त 5 किलो/एकर द्या."
            if normalized_lang == "mr"
            else "Use 19:19:19 water-soluble fertilizer up to 5 kg/acre only if crop growth slows during the vegetative stage."
        )
        flowering_lines.append(
            "फुलोरा किंवा फळधारणा कमकुवत दिसल्यासच SOP किंवा समतुल्य पोटॅश पूरकता जास्तीत जास्त 5 किलो/एकर द्या."
            if normalized_lang == "mr"
            else "Use SOP or equivalent potash support up to 5 kg/acre only if flowering or fruit set looks weak."
        )
        quantity_lines.extend([
            "नीम केक: 25 किलो/एकर" if normalized_lang == "mr" else "Neem cake: 25 kg/acre",
            "19:19:19 देखभाल डोस: जास्तीत जास्त 5 किलो/एकर" if normalized_lang == "mr" else "19:19:19 maintenance feed: up to 5 kg/acre",
            "फुलोरा पूरक डोस: जास्तीत जास्त 5 किलो/एकर" if normalized_lang == "mr" else "Flowering support feed: up to 5 kg/acre",
        ])
        organic_lines.extend([
            "पीक सामान्य वाढत असेल तर फक्त चांगले कुजलेले कंपोस्ट, मल्चिंग आणि हलके जैवखत वापरा."
            if normalized_lang == "mr"
            else "Continue only mature compost, mulching, and light biofertilizer use if the crop is growing normally.",
            "जोरदार रासायनिक सुधारणा करण्याऐवजी जीवामृत किंवा वर्मी-कंपोस्ट अर्क हलक्या देखभाल डोससाठी वापरता येईल."
            if normalized_lang == "mr"
            else "Jeevamrut or vermicompost tea can be used as a light maintenance input instead of a strong chemical correction.",
        ])
    else:
        p_source = "SSP" if actual_n > reference_n else "DAP"
        p_code = "P_SSP" if p_source == "SSP" else "P_DAP"

        urea_total = _dose_from_gap(n_gap, "N", crop_family)
        p_total = _dose_from_gap(p_gap, p_code, crop_family)
        mop_total = _dose_from_gap(k_gap, "K", crop_family)

        basal_urea, vegetative_urea, flowering_urea = _split_total(
            urea_total, UREA_STAGE_SPLITS[crop_family]
        )
        basal_p, vegetative_p, flowering_p = _split_total(p_total, P_STAGE_SPLITS)
        basal_mop, vegetative_mop, flowering_mop = _split_total(mop_total, K_STAGE_SPLITS)

        if basal_urea > 0:
            basal_lines.append(
                "युरिया: बेसल टप्प्यावर {0} किलो/एकर.".format(basal_urea)
                if normalized_lang == "mr"
                else "Urea: {0} kg/acre at basal stage.".format(basal_urea)
            )
        elif actual_n > reference_n:
            basal_lines.append(
                "नायट्रोजन आधीच लक्ष्यापेक्षा जास्त असल्याने बेसल टप्प्यावर युरिया देऊ नका."
                if normalized_lang == "mr"
                else "Do not add urea at basal stage because nitrogen is already above target."
            )

        if basal_p > 0:
            basal_lines.append(
                "{0}: बेसल टप्प्यावर {1} किलो/एकर.".format(p_source, basal_p)
                if normalized_lang == "mr"
                else "{0}: {1} kg/acre at basal stage.".format(p_source, basal_p)
            )
        elif actual_p > reference_p:
            basal_lines.append(
                "फॉस्फरस आधीच जास्त असल्याने सध्या फॉस्फरसयुक्त बेसल खत टाळा."
                if normalized_lang == "mr"
                else "Skip phosphorus-heavy basal fertilizer for now because phosphorus is already high."
            )

        if basal_mop > 0:
            basal_lines.append(
                "MOP: बेसल टप्प्यावर {0} किलो/एकर.".format(basal_mop)
                if normalized_lang == "mr"
                else "MOP: {0} kg/acre at basal stage.".format(basal_mop)
            )
        elif actual_k > reference_k:
            basal_lines.append(
                "पोटॅशियम आधीच जास्त असल्याने सध्या पोटॅशयुक्त बेसल खत टाळा."
                if normalized_lang == "mr"
                else "Skip potash-heavy basal fertilizer for now because potassium is already high."
            )

        if vegetative_urea > 0:
            vegetative_lines.append(
                "पेरणीनंतर किंवा लागवडीनंतर 20-25 दिवसांनी युरियाचा वरखत डोस: {0} किलो/एकर.".format(vegetative_urea)
                if normalized_lang == "mr"
                else "Top-dress urea: {0} kg/acre at 20-25 days after sowing or transplanting.".format(vegetative_urea)
            )
        if vegetative_p > 0:
            vegetative_lines.append(
                "मुळांची वाढ कमी दिसल्यास सुरुवातीच्या वाढीच्या टप्प्यात उरलेले {0}: {1} किलो/एकर द्या.".format(p_source, vegetative_p)
                if normalized_lang == "mr"
                else "Apply remaining {0}: {1} kg/acre during early vegetative growth if root development is weak.".format(p_source, vegetative_p)
            )
        if vegetative_mop > 0:
            vegetative_lines.append(
                "वाढीच्या टप्प्यात MOP: {0} किलो/एकर द्या.".format(vegetative_mop)
                if normalized_lang == "mr"
                else "Apply MOP: {0} kg/acre during vegetative growth.".format(vegetative_mop)
            )
        if not vegetative_lines:
            vegetative_lines.append(
                "पीक वाढ सामान्य असल्यास वाढीच्या टप्प्यात मोठ्या रासायनिक सुधारक डोसची गरज नाही."
                if normalized_lang == "mr"
                else "No major vegetative-stage chemical correction is needed if crop growth remains normal."
            )

        if flowering_urea > 0:
            flowering_lines.append(
                "फुलोऱ्यापूर्वी पाने फिकट दिसल्यासच हलका युरिया डोस: {0} किलो/एकर द्या.".format(flowering_urea)
                if normalized_lang == "mr"
                else "Use light urea support: {0} kg/acre only if foliage looks pale before flowering.".format(flowering_urea)
            )
        if flowering_p > 0:
            flowering_lines.append(
                "फुलोरा उशिरा येत असेल आणि फॉस्फरस कमतरतेची लक्षणे दिसत असल्यासच {0}: {1} किलो/एकर द्या.".format(p_source, flowering_p)
                if normalized_lang == "mr"
                else "Apply {0}: {1} kg/acre only if flowering is delayed and phosphorus shortage symptoms remain visible.".format(p_source, flowering_p)
            )
        if flowering_mop > 0:
            flowering_lines.append(
                "फुलोरा, बोंडधारणा, कंद वाढ किंवा फळधारणा वेळी MOP: {0} किलो/एकर द्या.".format(flowering_mop)
                if normalized_lang == "mr"
                else "Apply MOP: {0} kg/acre around flowering, boll formation, tuber bulking, or fruit set.".format(flowering_mop)
            )
        if not flowering_lines:
            flowering_lines.append(
                "पीक निरोगी असल्यास फुलोऱ्याच्या टप्प्यात जड रासायनिक सुधारणा आवश्यक नाही."
                if normalized_lang == "mr"
                else "No heavy flowering-stage chemical correction is needed if the crop remains healthy."
            )

        quantity_lines.extend([
            ("युरिया एकूण: {0} किलो/एकर".format(urea_total) if urea_total > 0 else "युरिया एकूण: सध्या 0 किलो/एकर")
            if normalized_lang == "mr"
            else ("Urea total: {0} kg/acre".format(urea_total) if urea_total > 0 else "Urea total: 0 kg/acre for now"),
            ("{0} एकूण: {1} किलो/एकर".format(p_source, p_total) if p_total > 0 else "{0} एकूण: सध्या 0 किलो/एकर".format(p_source))
            if normalized_lang == "mr"
            else ("{0} total: {1} kg/acre".format(p_source, p_total) if p_total > 0 else "{0} total: 0 kg/acre for now".format(p_source)),
            ("MOP एकूण: {0} किलो/एकर".format(mop_total) if mop_total > 0 else "MOP एकूण: सध्या 0 किलो/एकर")
            if normalized_lang == "mr"
            else ("MOP total: {0} kg/acre".format(mop_total) if mop_total > 0 else "MOP total: 0 kg/acre for now"),
        ])

        if n_gap > 0:
            organic_lines.append(
                "नायट्रोजन पूरकतेसाठी 300-500 किलो वर्मी-कंपोस्ट किंवा चांगले कुजलेले शेणखत प्रति एकर टप्प्याटप्प्याने द्या."
                if normalized_lang == "mr"
                else "For nitrogen support, use 300-500 kg vermicompost or well-rotted manure per acre in split applications."
            )
        elif actual_n > reference_n:
            organic_lines.append(
                "सध्या ताजे शेणखत आणि जड युरिया डोस टाळा; फक्त हलक्या प्रमाणात चांगले कुजलेले कंपोस्ट वापरा."
                if normalized_lang == "mr"
                else "Avoid fresh manure and heavy urea for now; use only mature compost in light quantity."
            )

        if p_gap > 0:
            organic_lines.append(
                "फॉस्फरस पूरकतेसाठी बेसल टप्प्यात कंपोस्टसोबत रॉक फॉस्फेट किंवा बोन मील वापरा."
                if normalized_lang == "mr"
                else "For phosphorus support, use rock phosphate or bone meal with compost at basal stage."
            )
        elif actual_p > reference_p:
            organic_lines.append(
                "पीक उपलब्ध फॉस्फरस वापरेपर्यंत DAP आणि फॉस्फरसयुक्त सेंद्रिय इनपुट टाळा."
                if normalized_lang == "mr"
                else "Avoid DAP and phosphorus-heavy manure until the crop uses the available phosphorus."
            )

        if k_gap > 0:
            organic_lines.append(
                "पोटॅशियम पूरकतेसाठी नियंत्रित प्रमाणात लाकडाची राख किंवा केळी अवशेष कंपोस्ट वापरा."
                if normalized_lang == "mr"
                else "For potassium support, use wood ash or banana-residue compost in a controlled quantity."
            )
        elif actual_k > reference_k:
            organic_lines.append(
                "सध्या म्युरिएट ऑफ पोटॅश आणि इतर पोटॅशयुक्त इनपुट टाळा."
                if normalized_lang == "mr"
                else "Avoid muriate of potash and other potash-heavy inputs for now."
            )

        if not organic_lines:
            organic_lines.append(
                "मोठ्या सेंद्रिय सुधारक डोसची गरज नाही; फक्त हलके कंपोस्ट आणि जैवखत पुरेसे आहे."
                if normalized_lang == "mr"
                else "Use only light compost and biofertilizer support because no strong organic correction is needed."
            )

    return {
        "crop_name": crop_display_name,
        "recommendation_key": recommendation_key,
        "summary": _major_issue_text(crop_display_name, recommendation_key, normalized_lang),
        "status_lines": _nutrient_status_lines(
            actual_n, actual_p, actual_k, reference_n, reference_p, reference_k, normalized_lang
        ),
        "basal_lines": basal_lines,
        "vegetative_lines": vegetative_lines,
        "flowering_lines": flowering_lines,
        "quantity_lines": quantity_lines,
        "organic_lines": organic_lines,
        "note": (
            "ही प्रति एकर अंदाजित योजना डेटासेटमधील पीक संदर्भ NPK मूल्ये आणि तुम्ही दिलेल्या मातीच्या मूल्यांवर आधारित आहे. स्थानिक कृषी सल्ला, पाण्याची उपलब्धता आणि पिकाच्या वयानुसार आवश्यक बदल करा."
            if normalized_lang == "mr"
            else "This is an approximate per-acre schedule derived from the crop reference NPK in the dataset and your submitted soil values. Adjust it with local agronomy guidance, irrigation availability, and crop age."
        ),
    }
