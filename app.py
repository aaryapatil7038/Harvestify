# Importing essential libraries and modules
import csv
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
from markupsafe import Markup
import pandas as pd
import pickle
import io
import os
import secrets
import requests
from datetime import date, timedelta, datetime
from functools import lru_cache

from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic, fertilizer_dic_mr
from utils.fertilizer_logic import (
    build_fertilizer_reference_df,
    build_fertilizer_plan,
    get_fertilizer_recommendation_key,
    get_fertilizer_recommendation_text,
)
from utils.crop_features import WATER_AVAILABILITY_OPTIONS, WATER_SOURCE_OPTIONS
from utils.msamb_price import get_msamb_live_price
from utils.yield_profit import estimate_yield_and_profit

# ==============================================================================================
# ------------------------- LOADING THE TRAINED MODELS ------------------------------------------

# Disease classes are kept in memory, but the heavy torch stack is loaded on demand.
disease_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy',
    'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]


@lru_cache(maxsize=1)
def get_crop_xai_runtime():
    from xai_explainer import explain_crop_prediction, explain_specific_crop_prediction

    return {
        "explain_crop_prediction": explain_crop_prediction,
        "explain_specific_crop_prediction": explain_specific_crop_prediction,
    }


@lru_cache(maxsize=1)
def get_disease_runtime():
    import torch
    from PIL import Image
    from torchvision import transforms
    from utils.model import ResNet9

    disease_model_path = 'models/plant_disease_model.pth'
    disease_model = ResNet9(3, len(disease_classes))
    disease_model.load_state_dict(
        torch.load(disease_model_path, map_location=torch.device('cpu'))
    )
    disease_model.eval()

    return {
        "torch": torch,
        "Image": Image,
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]),
        "model": disease_model,
    }

# ==============================================================================================
# ------------------------- LANGUAGE TRANSLATIONS -----------------------------------------------

UI_TEXT = {
    "en": {
        "language": "Language",
        "english": "English",
        "marathi": "मराठी",

        "nav_home": "Home",
        "nav_crop": "Crop",
        "nav_fertilizer": "Fertilizer",

        "home_title": "XAI Based Crop Recommendation System",
        "home_subtitle": "Get informed decisions about your farming strategy.",
        "home_questions_title": "Here are some questions we'll answer",
        "home_q1": "1. What crop to plant here?",
        "home_q2": "2. What fertilizer to use?",
        "services_title": "Our Services",
        "service_crop_title": "Crop",
        "service_crop_desc": "Recommendation about the type of crops to be cultivated which is best suited for the respective conditions",
        "service_fertilizer_title": "Fertilizer",
        "service_fertilizer_desc": "Recommendation about the type of fertilizer best suited for the particular soil and the recommended crop",

        "page_title": "Find out the most suitable crop to grow in your farm",
        "nitrogen": "Nitrogen",
        "phosphorous": "Phosphorous",
        "potassium": "Potassium",
        "ph_level": "pH level",
        "rainfall": "Rainfall (in mm)",
        "state": "State",
        "city": "City",
        "predict": "Predict",

        "nitrogen_placeholder": "Enter the value (example: 50)",
        "phosphorous_placeholder": "Enter the value (example: 50)",
        "potassium_placeholder": "Enter the value (example: 50)",
        "enter_value": "Enter the value",
        "land_area": "Land Area",
        "land_area_placeholder": "Enter land area",
        "land_unit": "Land Unit",
        "select_land_unit": "Select land unit",
        "acre": "Acre",
        "hectare": "Hectare",
        "water_source": "Water Source Available",
        "select_water_source": "Select water source",
        "water_availability": "Water Availability Level",
        "select_water_availability": "Select water availability",
        "rainfed": "Rainfed",
        "borewell": "Borewell",
        "canal": "Canal",
        "drip": "Drip Irrigation",
        "low": "Low",
        "medium": "Medium",
        "high": "High",

        "best_crop": "Best crop for your farm is",
        "top3": "Top 3 Recommended Crops",
        "rank": "Rank",
        "crop": "Crop",
        "relative_confidence": "Relative Confidence",
        "relative_confidence_info_title": "What does Relative Confidence mean?",
        "relative_confidence_info_body": "These percentages are calculated only among the top 3 recommended crops, so they sum to 100%. This helps compare the best three crop options more clearly for practical decision-making.",
        "xai_title": "Why the best crop was recommended (Explainable AI)",
        "impact_title": "Feature-wise Impact for",
        "feature": "Feature",
        "impact_value": "Impact Value",
        "effect": "Effect",
        "positive": "🟢 Positive (Helps crop)",
        "negative": "🔴 Negative (Harms crop)",
        "neutral": "⚪ Neutral",
        "overall_summary_title": "Overall Conclusion",

        "estimated_yield": "Estimated Yield",
        "estimated_profit": "Estimated Profit",
        "yield_unit": "quintal",
        "live_market_price": "Live Market Price",
        "market_name": "Market",
        "price_date": "Price Date",

        "fertilizer_title": "Get informed advice on fertilizer based on soil",
        "crop_you_want_to_grow": "Crop you want to grow",
        "select_crop": "Select crop",
        "fertilizer_result_title": "Recommended Fertilizer Advice",

        "try_again_msg": "Sorry we couldn't process your request currently. Please try again",
        "try_again_btn": "Try again",
        "error_details": "Validation details",

        "weather_source_note": "Weather values are auto-fetched from your selected farm location.",
        "current_location_error": "Current location not received. Please allow location access.",
        "manual_location_error": "Please enter a manual farm location or coordinates.",
        "weather_fetch_error": "Weather data could not be fetched for this location.",

        "location_method": "Location Method",
        "use_current_location": "Use Current Location",
        "enter_location_manually": "Enter Location Manually",
        "enter_farm_location": "Enter Farm Location",
        "manual_location_placeholder": "Enter village, city, or farm location",
        "or_enter_coordinates": "Or Enter Geo Coordinates",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "weather_details": "Weather Details",
        "city_location": "City / Location",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "coordinates": "निर्देशांक",
        "fetch_weather_manual": "Enter farm location or coordinates to fetch weather details.",
        "fetching_current_weather": "Fetching current location and weather details...",
        "fetching_manual_weather": "Fetching weather for entered farm location...",
        "weather_success": "Weather details fetched successfully.",
        "manual_weather_required": "Please fetch weather for manual farm location or coordinates before prediction.",
        "current_weather_required": "Please allow current location and fetch weather before prediction.",

        "sowing_guidance_title": "Sowing Window Guidance",
        "best_sowing_window": "Best Sowing Window",
        "transplanting_time": "Transplanting Time",
        "harvesting_time": "Harvesting Time",
        "crop_duration": "Approximate Crop Duration",
        "not_required": "Not required",
        "guidance_note": "This is approximate guidance and may vary by district, irrigation, and local farming practices."
    },

    "mr": {
        "language": "भाषा",
        "english": "English",
        "marathi": "मराठी",

        "nav_home": "मुख्यपृष्ठ",
        "nav_crop": "पीक",
        "nav_fertilizer": "खत",

        "home_title": "स्पष्टीकरणीय AI आधारित पीक शिफारस प्रणाली",
        "home_subtitle": "तुमच्या शेतीसाठी योग्य निर्णय घेण्यास मदत करणारी प्रणाली.",
        "home_questions_title": "आम्ही या प्रश्नांची उत्तरे देऊ",
        "home_q1": "1. येथे कोणते पीक घ्यावे?",
        "home_q2": "2. कोणते खत वापरावे?",
        "services_title": "आमच्या सेवा",
        "service_crop_title": "पीक",
        "service_crop_desc": "दिलेल्या परिस्थितीनुसार सर्वात योग्य पीक कोणते हे सुचवले जाते",
        "service_fertilizer_title": "खत",
        "service_fertilizer_desc": "माती आणि शिफारस केलेल्या पिकानुसार योग्य खताचा सल्ला दिला जातो",

        "page_title": "तुमच्या शेतासाठी सर्वात योग्य पीक शोधा",
        "nitrogen": "नायट्रोजन",
        "phosphorous": "फॉस्फरस",
        "potassium": "पोटॅशियम",
        "ph_level": "pH पातळी",
        "rainfall": "पाऊस (मिमी मध्ये)",
        "state": "राज्य",
        "city": "शहर",
        "predict": "शोधा",

        "nitrogen_placeholder": "मूल्य टाका (उदा.: ५०)",
        "phosphorous_placeholder": "मूल्य टाका (उदा.: ५०)",
        "potassium_placeholder": "मूल्य टाका (उदा.: ५०)",
        "enter_value": "मूल्य टाका",
        "land_area": "जमिनीचे क्षेत्रफळ",
        "land_area_placeholder": "जमिनीचे क्षेत्रफळ टाका",
        "land_unit": "जमिनीचे एकक",
        "select_land_unit": "जमिनीचे एकक निवडा",
        "acre": "एकर",
        "hectare": "हेक्टर",

        "best_crop": "तुमच्या शेतासाठी सर्वोत्तम पीक आहे",
        "top3": "सर्वोत्तम ३ शिफारस केलेली पिके",
        "rank": "क्रमांक",
        "crop": "पीक",
        "relative_confidence": "तुलनात्मक खात्री",
        "relative_confidence_info_title": "तुलनात्मक खात्री म्हणजे काय?",
        "relative_confidence_info_body": "ही टक्केवारी फक्त सर्वोत्तम ३ पिकांमध्ये मोजली जाते, त्यामुळे त्यांची बेरीज १००% होते. यामुळे शेतकऱ्याला सर्वोत्तम तीन पर्याय स्पष्टपणे तुलना करता येतात.",
        "xai_title": "हे सर्वोत्तम पीक का शिफारस केले गेले (Explainable AI)",
        "impact_title": "यासाठी घटकनिहाय परिणाम",
        "feature": "घटक",
        "impact_value": "परिणाम मूल्य",
        "effect": "परिणाम",
        "positive": "🟢 सकारात्मक (पीकासाठी चांगले)",
        "negative": "🔴 नकारात्मक (पीकासाठी कमी योग्य)",
        "neutral": "⚪ तटस्थ",
        "overall_summary_title": "एकूण निष्कर्ष",

        "estimated_yield": "अपेक्षित उत्पादन",
        "estimated_profit": "अंदाजित नफा",
        "yield_unit": "क्विंटल",
        "live_market_price": "सध्याचा बाजारभाव",
        "market_name": "बाजार",
        "price_date": "भाव दिनांक",

        "fertilizer_title": "मातीच्या आधारे योग्य खताचा सल्ला मिळवा",
        "crop_you_want_to_grow": "तुम्हाला घ्यायचे पीक",
        "select_crop": "पीक निवडा",
        "fertilizer_result_title": "शिफारस केलेला खत सल्ला",

        "try_again_msg": "माफ करा, सध्या तुमची विनंती प्रक्रिया करता आली नाही. कृपया पुन्हा प्रयत्न करा",
        "try_again_btn": "पुन्हा प्रयत्न करा",
        "error_details": "तपासणी तपशील",

        "weather_source_note": "हवामानाची मूल्ये निवडलेल्या शेताच्या स्थानावरून आपोआप घेतली जातात.",
        "current_location_error": "सध्याचे स्थान मिळाले नाही. कृपया location access द्या.",
        "manual_location_error": "कृपया शेताचे स्थान किंवा coordinates टाका.",
        "weather_fetch_error": "या स्थानासाठी हवामानाची माहिती मिळाली नाही.",

        "location_method": "स्थान पद्धत",
        "use_current_location": "सध्याचे स्थान वापरा",
        "enter_location_manually": "स्थान स्वतः टाका",
        "enter_farm_location": "शेताचे स्थान टाका",
        "manual_location_placeholder": "गाव, शहर किंवा शेताचे स्थान टाका",
        "or_enter_coordinates": "किंवा Geo Coordinates टाका",
        "latitude": "अक्षांश",
        "longitude": "रेखांश",
        "weather_details": "हवामान तपशील",
        "city_location": "शहर / स्थान",
        "temperature": "तापमान",
        "humidity": "आर्द्रता",
        "coordinates": "Coordinates",
        "fetch_weather_manual": "हवामान मिळवण्यासाठी शेताचे स्थान किंवा coordinates टाका.",
        "fetching_current_weather": "सध्याचे स्थान आणि हवामान घेत आहे...",
        "fetching_manual_weather": "दिलेल्या स्थानासाठी हवामान घेत आहे...",
        "weather_success": "हवामान तपशील यशस्वीरीत्या मिळाले.",
        "manual_weather_required": "Prediction करण्यापूर्वी manual location किंवा coordinates साठी weather fetch करा.",
        "current_weather_required": "Prediction करण्यापूर्वी current location allow करून weather fetch करा.",

        "sowing_guidance_title": "पेरणी मार्गदर्शन",
        "best_sowing_window": "सर्वोत्तम पेरणी कालावधी",
        "transplanting_time": "लागवड / पुनर्लागवड वेळ",
        "harvesting_time": "कापणीची वेळ",
        "crop_duration": "अंदाजे पीक कालावधी",
        "not_required": "गरज नाही",
        "guidance_note": "हे अंदाजे मार्गदर्शन आहे. जिल्हा, सिंचन आणि स्थानिक शेती पद्धतीनुसार यात बदल होऊ शकतो."
    }
}

UI_TEXT["en"].update({
    "water_source": "Water Source Available",
    "select_water_source": "Select water source",
    "water_availability": "Water Availability Level",
    "select_water_availability": "Select water availability",
    "rainfed": "Rainfed",
    "borewell": "Borewell",
    "canal": "Canal",
    "drip": "Drip Irrigation",
    "low": "Low",
    "medium": "Medium",
    "high": "High"
})

UI_TEXT["mr"].update({
    "water_source": "पाण्याचा स्रोत",
    "select_water_source": "पाण्याचा स्रोत निवडा",
    "water_availability": "पाणी उपलब्धता पातळी",
    "select_water_availability": "पाणी उपलब्धता निवडा",
    "rainfed": "पावसावर अवलंबून",
    "borewell": "बोअरवेल",
    "canal": "कालवा",
    "drip": "ठिबक सिंचन",
    "low": "कमी",
    "medium": "मध्यम",
    "high": "जास्त"
})

CROP_TRANSLATIONS = {
    "apple": "सफरचंद",
    "banana": "केळी",
    "blackgram": "उडीद",
    "chickpea": "हरभरा",
    "coconut": "नारळ",
    "coffee": "कॉफी",
    "cotton": "कापूस",
    "grapes": "द्राक्षे",
    "jute": "ज्यूट",
    "kidneybeans": "राजमा",
    "lentil": "मसूर",
    "maize": "मका",
    "mango": "आंबा",
    "mothbeans": "मटकी",
    "mungbean": "मूग",
    "muskmelon": "खरबूज",
    "orange": "संत्रे",
    "papaya": "पपई",
    "pigeonpeas": "तूर",
    "pomegranate": "डाळिंब",
    "rice": "तांदूळ",
    "watermelon": "टरबूज",
    "barley": "जव",
    "millet": "बाजरी",
    "wheat": "गहू",
    "soybean": "सोयाबीन",
    "groundnut": "शेंगदाणा",
    "sugarcane": "ऊस",
    "ragi": "नाचणी",
    "sorghum": "ज्वारी",
    "cowpea": "चवळी",
    "field pea": "मटार"
}

CROP_TRANSLATIONS.update({
    "jowar": "ज्वारी",
    "bajra": "बाजरी",
    "tur": "तूर",
    "gram": "हरभरा",
    "onion": "कांदा",
    "tomato": "टोमॅटो",
    "chili": "मिरची",
    "turmeric": "हळद",
    "ginger": "आले",
    "potato": "बटाटा",
    "brinjal": "वांगे",
    "cabbage": "कोबी",
    "sunflower": "सूर्यफूल",
    "sesame": "तीळ"
})

SEASON_TRANSLATIONS = {
    "Summer": "उन्हाळा",
    "Monsoon": "पावसाळा",
    "Winter": "हिवाळा"
}

FEATURE_TRANSLATIONS = {
    "Nitrogen": "नायट्रोजन",
    "Phosphorous": "फॉस्फरस",
    "Potassium": "पोटॅशियम",
    "Temperature": "तापमान",
    "Humidity": "आर्द्रता",
    "pH": "pH",
    "Rainfall": "पाऊस"
}

FEATURE_TRANSLATIONS.update({
    "Season": "हंगाम",
    "Water Source": "पाण्याचा स्रोत",
    "Water Availability": "पाणी उपलब्धता"
})

# ==============================================================================================
# ------------------------- CUSTOM FUNCTIONS -----------------------------------------------------

ALLOWED_LAND_UNITS = {"acre", "hectare"}
ALLOWED_LOCATION_METHODS = {"current", "manual"}
ALLOWED_WATER_SOURCES = set(WATER_SOURCE_OPTIONS)
ALLOWED_WATER_AVAILABILITY = set(WATER_AVAILABILITY_OPTIONS)
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
MODEL_INFO = {
    "version": "v2.1",
    "last_trained": "2026-03-28",
    "region_en": "Maharashtra-focused",
    "region_mr": "महाराष्ट्र-केंद्रित"
}


class ValidationError(ValueError):
    pass


def register_prediction_feedback_prompt():
    prediction_count = int(session.get("prediction_feedback_count", 0)) + 1
    session["prediction_feedback_count"] = prediction_count

    return {
        "prediction_count": prediction_count,
        "show_feedback_prompt": prediction_count % 3 == 0,
    }


def get_prediction_feedback_file_path():
    return os.path.join("Data", "prediction_feedback.csv")


def append_prediction_feedback_entry(entry):
    feedback_path = get_prediction_feedback_file_path()
    os.makedirs(os.path.dirname(feedback_path), exist_ok=True)

    fieldnames = [
        "submitted_at",
        "prediction_count",
        "prediction_type",
        "prediction_name",
        "usefulness",
        "chosen_crop",
        "season_result",
    ]

    file_exists = os.path.exists(feedback_path)

    with open(feedback_path, "a", newline="", encoding="utf-8") as feedback_file:
        writer = csv.DictWriter(feedback_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "submitted_at": datetime.now().isoformat(timespec="seconds"),
            "prediction_count": entry.get("prediction_count", ""),
            "prediction_type": entry.get("prediction_type", ""),
            "prediction_name": entry.get("prediction_name", ""),
            "usefulness": entry.get("usefulness", ""),
            "chosen_crop": entry.get("chosen_crop", ""),
            "season_result": entry.get("season_result", ""),
        })


def build_feedback_redirect_path(return_path, lang):
    if not return_path or not str(return_path).startswith("/"):
        return url_for("home", lang=lang)

    return "{0}{1}feedback=submitted".format(
        return_path,
        "&" if "?" in return_path else "?"
    )


def get_model_info(lang):
    return {
        "version_label": "मॉडेल आवृत्ती" if lang == "mr" else "Model version",
        "last_trained_label": "शेवटचे प्रशिक्षण" if lang == "mr" else "Last trained",
        "region_label": "प्रदेश" if lang == "mr" else "Region",
        "version": MODEL_INFO["version"],
        "last_trained": MODEL_INFO["last_trained"],
        "region": MODEL_INFO["region_mr"] if lang == "mr" else MODEL_INFO["region_en"],
    }


def render_input_error(lang, message, status_code=400, retry_endpoint="home"):
    return render_template(
        "try_again.html",
        lang=lang,
        ui=UI_TEXT[lang],
        error_message=message,
        retry_url=url_for(retry_endpoint, lang=lang),
        title="Try Again"
    ), status_code


def parse_numeric_field(form_data, field_name, display_name, minimum=None, maximum=None, integer_only=False):
    raw_value = str(form_data.get(field_name, "")).strip()

    if raw_value == "":
        raise ValidationError("{0} is required.".format(display_name))

    try:
        value = float(raw_value)
    except ValueError:
        raise ValidationError("{0} must be a valid number.".format(display_name))

    if integer_only and not value.is_integer():
        raise ValidationError("{0} must be a whole number.".format(display_name))

    if minimum is not None and value < minimum:
        raise ValidationError("{0} must be at least {1}.".format(display_name, minimum))

    if maximum is not None and value > maximum:
        raise ValidationError("{0} must be at most {1}.".format(display_name, maximum))

    return int(value) if integer_only else value


def validate_choice_field(form_data, field_name, display_name, allowed_values):
    value = str(form_data.get(field_name, "")).strip()

    if not value:
        raise ValidationError("{0} is required.".format(display_name))

    if value not in allowed_values:
        raise ValidationError("{0} is invalid.".format(display_name))

    return value


def validate_optional_text_field(form_data, field_name, display_name, max_length=120):
    value = str(form_data.get(field_name, "")).strip()

    if len(value) > max_length:
        raise ValidationError("{0} must be shorter than {1} characters.".format(display_name, max_length + 1))

    return value


def validate_crop_form_input(form_data):
    validated = {
        "N": parse_numeric_field(form_data, "nitrogen", "Nitrogen", minimum=0, maximum=300),
        "P": parse_numeric_field(form_data, "phosphorous", "Phosphorous", minimum=0, maximum=300),
        "K": parse_numeric_field(form_data, "pottasium", "Potassium", minimum=0, maximum=300),
        "ph": parse_numeric_field(form_data, "ph", "pH", minimum=0, maximum=14),
        "land_area": parse_numeric_field(form_data, "land_area", "Land area", minimum=0.01, maximum=100000),
        "land_unit": validate_choice_field(form_data, "land_unit", "Land unit", ALLOWED_LAND_UNITS),
        "water_source": validate_choice_field(form_data, "water_source", "Water source", ALLOWED_WATER_SOURCES),
        "water_availability": validate_choice_field(
            form_data,
            "water_availability",
            "Water availability",
            ALLOWED_WATER_AVAILABILITY
        ),
        "location_method": validate_choice_field(form_data, "location_method", "Location method", ALLOWED_LOCATION_METHODS),
        "manual_location": validate_optional_text_field(form_data, "manual_location", "Manual location", max_length=120)
    }

    rainfall_raw = str(form_data.get("rainfall", "")).strip()
    validated["user_rainfall"] = None
    if rainfall_raw:
        validated["user_rainfall"] = parse_numeric_field(
            form_data,
            "rainfall",
            "Rainfall",
            minimum=0,
            maximum=5000
        )

    validated["latitude"] = str(form_data.get("latitude", "")).strip()
    validated["longitude"] = str(form_data.get("longitude", "")).strip()
    validated["weather_city"] = validate_optional_text_field(form_data, "weather_city", "Weather city", max_length=120)
    validated["weather_temperature"] = None
    validated["weather_humidity"] = None

    weather_temperature_raw = str(form_data.get("weather_temperature", "")).strip()
    if weather_temperature_raw:
        validated["weather_temperature"] = parse_numeric_field(
            form_data,
            "weather_temperature",
            "Temperature",
            minimum=-50,
            maximum=80
        )

    weather_humidity_raw = str(form_data.get("weather_humidity", "")).strip()
    if weather_humidity_raw:
        validated["weather_humidity"] = parse_numeric_field(
            form_data,
            "weather_humidity",
            "Humidity",
            minimum=0,
            maximum=100
        )

    if validated["location_method"] == "manual":
        if validated["manual_location"]:
            pass
        elif validated["latitude"] and validated["longitude"]:
            if not is_valid_coordinate_pair(validated["latitude"], validated["longitude"]):
                raise ValidationError("Manual latitude and longitude are invalid.")
        else:
            raise ValidationError("Please enter a manual farm location or valid coordinates.")
    else:
        if not is_valid_coordinate_pair(validated["latitude"], validated["longitude"]):
            raise ValidationError("Current location coordinates are missing or invalid.")

    return validated


def validate_fertilizer_form_input(form_data, fertilizer_df):
    validated = {
        "crop_name": validate_choice_field(
            form_data,
            "cropname",
            "Crop",
            set(fertilizer_df["Crop"].astype(str).str.strip().tolist())
        ),
        "N": parse_numeric_field(form_data, "nitrogen", "Nitrogen", minimum=0, maximum=300),
        "P": parse_numeric_field(form_data, "phosphorous", "Phosphorous", minimum=0, maximum=300),
        "K": parse_numeric_field(form_data, "pottasium", "Potassium", minimum=0, maximum=300)
    }
    return validated


def resolve_existing_data_path(*candidate_paths):
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path
    raise FileNotFoundError("Required data file was not found.")


@lru_cache(maxsize=1)
def get_fertilizer_reference_df():
    fertilizer_path = resolve_existing_data_path("Data/fertilizer.csv", "data/fertilizer.csv")
    crop_reference_path = resolve_existing_data_path(
        "Data/crop_recommendation_merged_maharashtra.csv",
        "data/crop_recommendation_merged_maharashtra.csv",
        "Data/crop_recommendation.csv",
        "data/crop_recommendation.csv"
    )
    return build_fertilizer_reference_df(fertilizer_path, crop_reference_path)


def validate_disease_upload(file_storage):
    if not file_storage or file_storage.filename == '':
        raise ValidationError("Please upload an image file.")

    extension = os.path.splitext(file_storage.filename)[1].lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValidationError("Only JPG, JPEG, PNG, and WEBP images are allowed.")

    image_bytes = file_storage.read()
    file_storage.stream.seek(0)

    if not image_bytes:
        raise ValidationError("Uploaded image is empty.")

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise ValidationError("Uploaded image is too large.")

    try:
        preview_image = Image.open(io.BytesIO(image_bytes))
        preview_image.verify()
    except Exception:
        raise ValidationError("Uploaded file is not a valid image.")

    return image_bytes

def geocode_location(location_text):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": location_text,
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 1
    }
    headers = {
        "User-Agent": "Harvestify/1.0 (educational project)"
    }

    response = requests.get(url, params=params, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()

    if not data:
        return None, None

    lat = data[0].get("lat")
    lon = data[0].get("lon")
    return lat, lon


@lru_cache(maxsize=256)
def reverse_geocode_city(lat, lon):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "format": "jsonv2",
        "addressdetails": 1
    }
    headers = {
        "User-Agent": "Harvestify/1.0 (educational project)"
    }

    response = requests.get(url, params=params, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()

    address = data.get("address", {})
    city = (
        address.get("city")
        or address.get("town")
        or address.get("village")
        or address.get("county")
        or address.get("state_district")
        or address.get("state")
        or "Unknown"
    )
    return city


@lru_cache(maxsize=256)
def get_recent_weather_summary(lat, lon, days=120):
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=days - 1)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum",
        "timezone": "auto"
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    daily = data.get("daily", {})
    temperatures = daily.get("temperature_2m_mean", [])
    humidities = daily.get("relative_humidity_2m_mean", [])
    precipitation = daily.get("precipitation_sum", [])

    temps = [float(x) for x in temperatures if x is not None]
    hums = [float(x) for x in humidities if x is not None]
    rains = [float(x) for x in precipitation if x is not None]

    if not temps or not hums:
        raise ValueError("Weather data not available for this location.")

    avg_temp = round(sum(temps) / len(temps), 2)
    avg_humidity = round(sum(hums) / len(hums), 2)
    total_rainfall = round(sum(rains), 2) if rains else None

    return avg_temp, avg_humidity, total_rainfall


def is_valid_coordinate_pair(lat, lon):
    try:
        lat = float(lat)
        lon = float(lon)
        return -90 <= lat <= 90 and -180 <= lon <= 180
    except Exception:
        return False


def resolve_location_and_weather(lang, latitude="", longitude="", manual_location="", location_method="current"):
    latitude = (latitude or "").strip()
    longitude = (longitude or "").strip()
    manual_location = (manual_location or "").strip()
    location_method = (location_method or "current").strip()

    if location_method == "manual":
        if is_valid_coordinate_pair(latitude, longitude):
            city = reverse_geocode_city(latitude, longitude)
        elif manual_location:
            latitude, longitude = geocode_location(manual_location)

            if not latitude or not longitude:
                raise ValueError("Could not find latitude and longitude for the entered location.")

            city = manual_location
        else:
            raise ValueError(UI_TEXT[lang]["manual_location_error"])
    else:
        if not is_valid_coordinate_pair(latitude, longitude):
            raise ValueError(UI_TEXT[lang]["current_location_error"])

        city = reverse_geocode_city(latitude, longitude)

    temperature, humidity, api_rainfall = get_recent_weather_summary(latitude, longitude, days=120)

    return {
        "latitude": round(float(latitude), 6),
        "longitude": round(float(longitude), 6),
        "city": city,
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": api_rainfall
    }


def get_current_crop_season():
    month = datetime.now().month

    if month in [6, 7, 8, 9]:
        return "Kharif"
    if month in [10, 11, 12, 1, 2]:
        return "Rabi"
    return "Summer"


def get_season_crop_bonus_map():
    return {
        "Kharif": {
            "rice": 8.0,
            "soybean": 7.0,
            "cotton": 7.0,
            "banana": 6.0,
            "maize": 6.0,
            "tur": 6.0,
            "orange": 5.0,
            "ginger": 5.0,
            "groundnut": 5.0,
            "jowar": 5.0,
            "bajra": 5.0,
            "ragi": 5.0,
            "blackgram": 4.5,
            "sesame": 4.5,
            "cowpea": 4.5,
            "chili": 4.0,
            "sunflower": 4.0,
            "turmeric": 4.0,
            "sugarcane": 3.0
        },
        "Rabi": {
            "wheat": 8.0,
            "gram": 7.0,
            "potato": 7.0,
            "onion": 6.0,
            "tomato": 6.0,
            "grapes": 6.0,
            "cabbage": 6.0,
            "brinjal": 5.0,
            "lentil": 5.0,
            "orange": 4.0,
            "sunflower": 4.0,
            "jowar": 5.0,
            "chili": 5.0,
            "maize": 3.0,
            "sugarcane": 2.0
        },
        "Summer": {
            "sugarcane": 8.0,
            "banana": 7.0,
            "grapes": 7.0,
            "pomegranate": 6.0,
            "tomato": 6.0,
            "onion": 6.0,
            "mungbean": 5.0,
            "rice": 5.0,
            "maize": 5.0,
            "groundnut": 4.0,
            "bajra": 4.0,
            "sesame": 4.0,
            "cowpea": 4.0
        }
    }


def rerank_top3_predictions_by_season(top3_predictions, season=None):
    season = season or get_current_crop_season()
    bonus_map = get_season_crop_bonus_map().get(season, {})

    adjusted_predictions = []

    for item in top3_predictions:
        crop_name = str(item.get("crop", "")).lower()
        actual_confidence = float(item.get("actual_confidence", item.get("confidence", 0)))
        season_bonus = float(bonus_map.get(crop_name, 0.0))
        adjusted_score = actual_confidence + season_bonus

        adjusted_predictions.append({
            "crop": item["crop"],
            "actual_confidence": round(actual_confidence, 2),
            "season_bonus": round(season_bonus, 2),
            "season_adjusted_score": round(adjusted_score, 2)
        })

    adjusted_predictions.sort(
        key=lambda x: (x["season_adjusted_score"], x["actual_confidence"]),
        reverse=True
    )

    total_adjusted_score = sum(max(item["season_adjusted_score"], 0.01) for item in adjusted_predictions)

    for item in adjusted_predictions:
        relative_confidence = (max(item["season_adjusted_score"], 0.01) / total_adjusted_score) * 100
        item["confidence"] = round(relative_confidence, 2)

    return season, adjusted_predictions


def get_crop_sowing_guidance(crop_name, lang="en"):
    crop_key = str(crop_name).strip().lower()

    crop_alias_map = {
        "mung bean": "mungbean",
        "mungbean": "mungbean",
        "moth bean": "mothbeans",
        "mothbeans": "mothbeans",
        "pigeon pea": "pigeonpeas",
        "pigeon peas": "pigeonpeas",
        "pigeonpeas": "pigeonpeas",
        "black gram": "blackgram",
        "blackgram": "blackgram",
        "kidney beans": "kidneybeans",
        "kidneybeans": "kidneybeans",
        "fieldpea": "field pea",
        "field pea": "field pea",
        "ground nut": "groundnut",
        "groundnut": "groundnut",
        "water melon": "watermelon",
        "watermelon": "watermelon",
        "muskmelon": "muskmelon",
        "muskmelon ": "muskmelon",
        "sugar cane": "sugarcane",
        "sugarcane": "sugarcane"
    }

    crop_alias_map.update({
        "jowar": "sorghum",
        "bajra": "millet",
        "tur": "pigeonpeas",
        "gram": "chickpea",
        "chili": "chili",
        "chilli": "chili",
        "onion": "onion",
        "tomato": "tomato",
        "turmeric": "turmeric"
    })

    crop_key = crop_alias_map.get(crop_key, crop_key)

    guidance_data = {
        "rice": {
            "en": {
                "sowing_window": "June 15 – July 15",
                "transplanting_time": "20–30 days after nursery sowing",
                "harvesting_time": "October – November",
                "crop_duration": "110–140 days"
            },
            "mr": {
                "sowing_window": "१५ जून – १५ जुलै",
                "transplanting_time": "रोपवाटिकेनंतर २०–३० दिवसांनी पुनर्लागवड",
                "harvesting_time": "ऑक्टोबर – नोव्हेंबर",
                "crop_duration": "११०–१४० दिवस"
            }
        },
        "soybean": {
            "en": {
                "sowing_window": "June 15 – July 10",
                "transplanting_time": "Not required",
                "harvesting_time": "October – November",
                "crop_duration": "90–110 days"
            },
            "mr": {
                "sowing_window": "१५ जून – १० जुलै",
                "transplanting_time": "गरज नाही",
                "harvesting_time": "ऑक्टोबर – नोव्हेंबर",
                "crop_duration": "९०–११० दिवस"
            }
        },
        "cotton": {
            "en": {
                "sowing_window": "June 1 – July 15",
                "transplanting_time": "Not required",
                "harvesting_time": "October onwards",
                "crop_duration": "150–180 days"
            },
            "mr": {
                "sowing_window": "१ जून – १५ जुलै",
                "transplanting_time": "गरज नाही",
                "harvesting_time": "ऑक्टोबरपासून पुढे",
                "crop_duration": "१५०–१८० दिवस"
            }
        },
        "maize": {
            "en": {
                "sowing_window": "June 15 – July 15",
                "transplanting_time": "Not required",
                "harvesting_time": "September – October",
                "crop_duration": "90–120 days"
            },
            "mr": {
                "sowing_window": "१५ जून – १५ जुलै",
                "transplanting_time": "गरज नाही",
                "harvesting_time": "सप्टेंबर – ऑक्टोबर",
                "crop_duration": "९०–१२० दिवस"
            }
        },
        "pigeonpeas": {
            "en": {
                "sowing_window": "June 15 – July 15",
                "transplanting_time": "Not required",
                "harvesting_time": "December – January",
                "crop_duration": "150–180 days"
            },
            "mr": {
                "sowing_window": "१५ जून – १५ जुलै",
                "transplanting_time": "गरज नाही",
                "harvesting_time": "डिसेंबर – जानेवारी",
                "crop_duration": "१५०–१८० दिवस"
            }
        },
        "groundnut": {
            "en": {
                "sowing_window": "June 10 – July 10",
                "transplanting_time": "Not required",
                "harvesting_time": "September – October",
                "crop_duration": "100–120 days"
            },
            "mr": {
                "sowing_window": "१० जून – १० जुलै",
                "transplanting_time": "गरज नाही",
                "harvesting_time": "सप्टेंबर – ऑक्टोबर",
                "crop_duration": "१००–१२० दिवस"
            }
        },
        "sorghum": {
            "en": {
                "sowing_window": "June 15 – July 15",
                "transplanting_time": "Not required",
                "harvesting_time": "September – October",
                "crop_duration": "100–120 days"
            },
            "mr": {
                "sowing_window": "१५ जून – १५ जुलै",
                "transplanting_time": "गरज नाही",
                "harvesting_time": "सप्टेंबर – ऑक्टोबर",
                "crop_duration": "१००–१२० दिवस"
            }
        },
        "millet": {
            "en": {
                "sowing_window": "June 15 – July 15",
                "transplanting_time": "Not required",
                "harvesting_time": "September – October",
                "crop_duration": "75–100 days"
            },
            "mr": {
                "sowing_window": "१५ जून – १५ जुलै",
                "transplanting_time": "गरज नाही",
                "harvesting_time": "सप्टेंबर – ऑक्टोबर",
                "crop_duration": "७५–१०० दिवस"
            }
        },
        "ragi": {
            "en": {
                "sowing_window": "June 15 – July 15",
                "transplanting_time": "20–25 days after nursery sowing if transplanted",
                "harvesting_time": "October – November",
                "crop_duration": "100–120 days"
            },
            "mr": {
                "sowing_window": "१५ जून – १५ जुलै",
                "transplanting_time": "रोपवाटिकेनंतर २०–२५ दिवसांनी, जर पुनर्लागवड केली तर",
                "harvesting_time": "ऑक्टोबर – नोव्हेंबर",
                "crop_duration": "१००–१२० दिवस"
            }
        },
        "jute": {
            "en": {
                "sowing_window": "March – May",
                "transplanting_time": "Not required",
                "harvesting_time": "July – September",
                "crop_duration": "120–150 days"
            },
            "mr": {
                "sowing_window": "मार्च – मे",
                "transplanting_time": "गरज नाही",
                "harvesting_time": "जुलै – सप्टेंबर",
                "crop_duration": "१२०–१५० दिवस"
            }
        },
        "sugarcane": {
            "en": {
                "sowing_window": "January – March or October – November",
                "transplanting_time": "Sett planting / transplanting as per method used",
                "harvesting_time": "10–14 months after planting",
                "crop_duration": "300–420 days"
            },
            "mr": {
                "sowing_window": "जानेवारी – मार्च किंवा ऑक्टोबर – नोव्हेंबर",
                "transplanting_time": "वापरलेल्या पद्धतीनुसार लागवड",
                "harvesting_time": "लागवडीनंतर १०–१४ महिने",
                "crop_duration": "३००–४२० दिवस"
            }
        },
        "wheat": {
            "en": {
                "sowing_window": "October 15 – November 30",
                "transplanting_time": "Not required",
                "harvesting_time": "February – March",
                "crop_duration": "110–140 days"
            },
            "mr": {
                "sowing_window": "१५ ऑक्टोबर – ३० नोव्हेंबर",
                "transplanting_time": "गरज नाही",
                "harvesting_time": "फेब्रुवारी – मार्च",
                "crop_duration": "११०–१४० दिवस"
            }
        },
        "mustard": {
            "en": {
                "sowing_window": "October – November",
                "transplanting_time": "Not required",
                "harvesting_time": "February – March",
                "crop_duration": "110–140 days"
            },
            "mr": {
                "sowing_window": "ऑक्टोबर – नोव्हेंबर",
                "transplanting_time": "गरज नाही",
                "harvesting_time": "फेब्रुवारी – मार्च",
                "crop_duration": "११०–१४० दिवस"
            }
        },
        "barley": {
            "en": {
                "sowing_window": "October – November",
                "transplanting_time": "Not required",
                "harvesting_time": "March – April",
                "crop_duration": "120–140 days"
            },
            "mr": {
                "sowing_window": "ऑक्टोबर – नोव्हेंबर",
                "transplanting_time": "गरज नाही",
                "harvesting_time": "मार्च – एप्रिल",
                "crop_duration": "१२०–१४० दिवस"
            }
        },
        "tea": {
            "en": {
                "sowing_window": "June – September",
                "transplanting_time": "Plant nursery saplings",
                "harvesting_time": "Plucking starts after establishment period",
                "crop_duration": "Perennial crop"
            },
            "mr": {
                "sowing_window": "जून – सप्टेंबर",
                "transplanting_time": "रोपवाटिकेतील रोपे लावा",
                "harvesting_time": "स्थापना झाल्यानंतर तोडणी सुरू होते",
                "crop_duration": "बहुवर्षायू पीक"
            }
        },
        "chickpea": {
            "en": {
                "sowing_window": "October – November",
                "transplanting_time": "Not required",
                "harvesting_time": "February – March",
                "crop_duration": "100–120 days"
            },
            "mr": {
                "sowing_window": "ऑक्टोबर – नोव्हेंबर",
                "transplanting_time": "गरज नाही",
                "harvesting_time": "फेब्रुवारी – मार्च",
                "crop_duration": "१००–१२० दिवस"
            }
        }
    }

    guidance_data.update({
        "onion": {
            "en": {
                "sowing_window": "October - January",
                "transplanting_time": "30-40 days after nursery sowing",
                "harvesting_time": "February - May",
                "crop_duration": "90-130 days"
            },
            "mr": {
                "sowing_window": "ऑक्टोबर - जानेवारी",
                "transplanting_time": "रोपवाटिकेनंतर ३०-४० दिवसांनी लागवड",
                "harvesting_time": "फेब्रुवारी - मे",
                "crop_duration": "९०-१३० दिवस"
            }
        },
        "tomato": {
            "en": {
                "sowing_window": "September - January",
                "transplanting_time": "25-30 days after nursery sowing",
                "harvesting_time": "December onwards",
                "crop_duration": "100-130 days"
            },
            "mr": {
                "sowing_window": "सप्टेंबर - जानेवारी",
                "transplanting_time": "रोपवाटिकेनंतर २५-३० दिवसांनी लागवड",
                "harvesting_time": "डिसेंबरपासून पुढे",
                "crop_duration": "१००-१३० दिवस"
            }
        },
        "chili": {
            "en": {
                "sowing_window": "June - July or October - November",
                "transplanting_time": "30-40 days after nursery sowing",
                "harvesting_time": "October onwards",
                "crop_duration": "150-210 days"
            },
            "mr": {
                "sowing_window": "जून - जुलै किंवा ऑक्टोबर - नोव्हेंबर",
                "transplanting_time": "रोपवाटिकेनंतर ३०-४० दिवसांनी लागवड",
                "harvesting_time": "ऑक्टोबरपासून पुढे",
                "crop_duration": "१५०-२१० दिवस"
            }
        },
        "turmeric": {
            "en": {
                "sowing_window": "May - June",
                "transplanting_time": "Rhizome planting directly in field",
                "harvesting_time": "January - March",
                "crop_duration": "210-270 days"
            },
            "mr": {
                "sowing_window": "मे - जून",
                "transplanting_time": "बियाणे गाठींची थेट लागवड",
                "harvesting_time": "जानेवारी - मार्च",
                "crop_duration": "२१०-२७० दिवस"
            }
        }
    })

    if crop_key in guidance_data:
        return guidance_data[crop_key][lang]

    if lang == "mr":
        return {
            "sowing_window": "स्थानिक हंगामानुसार",
            "transplanting_time": "पीक प्रकारानुसार",
            "harvesting_time": "पिकाच्या कालावधीनुसार",
            "crop_duration": "स्थानिक शिफारशीनुसार"
        }

    return {
        "sowing_window": "As per local season",
        "transplanting_time": "Depends on crop type",
        "harvesting_time": "As per crop maturity",
        "crop_duration": "As per local recommendation"
    }


def predict_image(img):
    disease_runtime = get_disease_runtime()
    image = disease_runtime["Image"].open(io.BytesIO(img)).convert("RGB")
    img_t = disease_runtime["transform"](image)
    img_u = disease_runtime["torch"].unsqueeze(img_t, 0)

    with disease_runtime["torch"].no_grad():
        yb = disease_runtime["model"](img_u)
        _, preds = disease_runtime["torch"].max(yb, dim=1)

    prediction = disease_classes[preds[0].item()]
    return prediction


def get_lang():
    lang = request.form.get("lang") or request.args.get("lang") or "en"
    return "mr" if lang == "mr" else "en"


def translate_crop_name(crop_name, lang):
    if lang == "mr":
        return CROP_TRANSLATIONS.get(crop_name.lower(), crop_name)
    return crop_name


def translate_season_name(season_name, lang):
    if lang == "mr":
        return SEASON_TRANSLATIONS.get(season_name, season_name)
    return season_name


def translate_effect(effect_text, lang):
    if lang == "mr":
        if "Positive" in effect_text:
            return UI_TEXT["mr"]["positive"]
        elif "Negative" in effect_text:
            return UI_TEXT["mr"]["negative"]
        else:
            return UI_TEXT["mr"]["neutral"]
    return effect_text


def translate_explanation_line(line, prediction, lang):
    if lang == "en":
        return line

    crop_mr = translate_crop_name(prediction, "mr")

    positive_templates = [
        "positively influenced the recommendation of",
        "positively influenced the selection of",
        "positively influenced"
    ]
    negative_templates = [
        "negatively affected the suitability of",
        "negatively affected suitability",
        "negatively affected"
    ]
    neutral_templates = [
        "had a neutral effect on the prediction",
        "had a neutral effect"
    ]

    matched_feature_en = None
    matched_feature_mr = None

    for eng_feature, mr_feature in FEATURE_TRANSLATIONS.items():
        if line.startswith(eng_feature):
            matched_feature_en = eng_feature
            matched_feature_mr = mr_feature
            break

    if matched_feature_mr is None:
        return line

    remaining_text = line[len(matched_feature_en):].strip().lower()

    for phrase in positive_templates:
        if phrase in remaining_text:
            return f"{matched_feature_mr}मुळे {crop_mr} या पिकाची शिफारस करण्यात मदत होते."

    for phrase in negative_templates:
        if phrase in remaining_text:
            return f"{matched_feature_mr}मुळे {crop_mr} या पिकाच्या शिफारशीमध्ये काही प्रमाणात अडथळा येतो."

    for phrase in neutral_templates:
        if phrase in remaining_text:
            return f"{matched_feature_mr}चा या भविष्यवाणीवर तटस्थ परिणाम होतो."

    return f"{matched_feature_mr}मुळे {crop_mr} या पिकाच्या शिफारसीवर परिणाम होतो."


def generate_overall_summary(prediction, top3_predictions, explanation_df, lang):
    positive_count = int((explanation_df["Impact Value"] > 0).sum())
    negative_count = int((explanation_df["Impact Value"] < 0).sum())
    neutral_count = int((explanation_df["Impact Value"] == 0).sum())

    best_crop = translate_crop_name(prediction, lang)
    second_crop = translate_crop_name(top3_predictions[1]["crop"], lang) if len(top3_predictions) > 1 else None
    third_crop = translate_crop_name(top3_predictions[2]["crop"], lang) if len(top3_predictions) > 2 else None

    if lang == "mr":
        if positive_count > negative_count:
            base = (
                f"दिलेल्या जमिनीतील पोषकद्रव्ये, pH, पाऊस, तापमान आणि आर्द्रता यांचा एकत्रित विचार करता, "
                f"{best_crop} हे पीक तुमच्या शेतासाठी सर्वात योग्य दिसत आहे. बहुतेक घटक या पिकाच्या "
                f"शिफारशीस अनुकूल आहेत, तर काही घटक थोड्या प्रमाणात अडथळा निर्माण करतात."
            )
        elif positive_count == negative_count:
            base = (
                f"दिलेल्या माहितीचा विचार करता, {best_crop} हे पीक तुमच्या शेतासाठी योग्य पर्याय दिसत आहे. "
                f"काही घटक या पिकासाठी अनुकूल आहेत, तर काही घटक मर्यादा दाखवतात, त्यामुळे काळजीपूर्वक निर्णय घेणे योग्य ठरेल."
            )
        else:
            base = (
                f"दिलेल्या माहितीच्या आधारे {best_crop} हे पीक सर्वोत्तम पर्याय म्हणून सुचवले गेले आहे, "
                f"परंतु काही महत्त्वाचे घटक या पिकासाठी पूर्णपणे अनुकूल नाहीत. त्यामुळे प्रत्यक्ष लागवडीपूर्वी स्थानिक परिस्थितीचा विचार करणे उपयुक्त ठरेल."
            )

        alt = ""
        if second_crop and third_crop:
            alt = f" {second_crop} आणि {third_crop} ही पिके पर्यायी पर्याय म्हणून देखील सुचवली आहेत."
        elif second_crop:
            alt = f" {second_crop} हे पीक पर्यायी पर्याय म्हणून सुचवले आहे."

        neutral_text = " काही घटकांचा परिणाम तटस्थ देखील आहे." if neutral_count > 0 else ""
        return base + neutral_text + alt

    else:
        if positive_count > negative_count:
            base = (
                f"Considering the given soil nutrients, pH, rainfall, temperature, humidity, season, and irrigation context together, "
                f"{best_crop} appears to be the most suitable crop for your farm. Most parameters support this recommendation, "
                f"while a few factors create only minor limitations."
            )
        elif positive_count == negative_count:
            base = (
                f"Based on the given input values, {best_crop} appears to be a suitable choice for your farm. "
                f"Some parameters support this crop, while some show limitations, so the recommendation should be interpreted with care."
            )
        else:
            base = (
                f"Based on the given input values, {best_crop} has been recommended as the best available option, "
                f"but some important parameters are not fully favorable for this crop. It is better to consider local farming conditions before final cultivation."
            )

        alt = ""
        if second_crop and third_crop:
            alt = f" {second_crop} and {third_crop} are also suggested as alternative crop options."
        elif second_crop:
            alt = f" {second_crop} is also suggested as an alternative crop option."

        neutral_text = " Some parameters also show a neutral effect." if neutral_count > 0 else ""
        return base + neutral_text + alt


def localize_crop_output(prediction, top3_predictions, explanation_df, explanations, lang):
    localized_prediction = translate_crop_name(prediction, lang)

    localized_top3 = []
    for item in top3_predictions:
        localized_top3.append({
            "crop": translate_crop_name(item["crop"], lang),
            "confidence": item["confidence"],
            "actual_confidence": item.get("actual_confidence", item["confidence"]),
            "season_bonus": item.get("season_bonus", 0),
            "season_adjusted_score": item.get("season_adjusted_score", item.get("actual_confidence", item["confidence"]))
        })

    localized_explanation_df = explanation_df.copy()

    if lang == "mr":
        localized_explanation_df["Feature"] = localized_explanation_df["Feature"].replace(FEATURE_TRANSLATIONS)
        localized_explanation_df["Effect"] = localized_explanation_df["Effect"].apply(
            lambda x: translate_effect(x, lang)
        )

    localized_explanations = [
        translate_explanation_line(line, prediction, lang) for line in explanations
    ]

    return localized_prediction, localized_top3, localized_explanation_df, localized_explanations


def render_saved_crop_result(saved_result, lang):
    current_season_raw = saved_result.get("current_season")
    feedback_prompt_visible = saved_result.get("show_feedback_prompt", False) and not saved_result.get("feedback_submitted", False)

    return render_template(
        "crop-result.html",
        lang=lang,
        ui=UI_TEXT[lang],
        prediction=translate_crop_name(saved_result["prediction_raw"], lang),
        prediction_raw=saved_result["prediction_raw"],
        top3_predictions=[
            {
                "crop": translate_crop_name(item["crop"], lang),
                "confidence": item["confidence"],
                "actual_confidence": item.get("actual_confidence", item["confidence"]),
                "season_bonus": item.get("season_bonus", 0),
                "season_adjusted_score": item.get(
                    "season_adjusted_score",
                    item.get("actual_confidence", item["confidence"])
                )
            }
            for item in saved_result["top3_predictions"]
        ],
        top3_predictions_raw=saved_result["top3_predictions"],
        explanation=pd.DataFrame(saved_result["explanation"]),
        explanations=[
            translate_explanation_line(line, saved_result["prediction_raw"], lang)
            for line in saved_result["explanations"]
        ],
        overall_summary=generate_overall_summary(
            saved_result["prediction_raw"],
            saved_result["top3_predictions"],
            pd.DataFrame(saved_result["explanation"]),
            lang
        ),
        land_area=saved_result["land_area"],
        land_unit_display=("एकर" if saved_result["land_unit"] == "acre" else "हेक्टर") if lang == "mr"
                          else ("Acre" if saved_result["land_unit"] == "acre" else "Hectare"),
        estimated_yield=saved_result["estimated_yield"],
        estimated_profit=saved_result["estimated_profit"],
        yield_profit_summary=saved_result["yield_profit_summary_mr"] if lang == "mr"
                            else saved_result["yield_profit_summary_en"],
        live_modal_price=saved_result["live_modal_price"],
        market_name=saved_result["market_name"],
        price_date=saved_result["price_date"],
        weather_city=saved_result.get("weather_city"),
        weather_temperature=saved_result.get("weather_temperature"),
        weather_humidity=saved_result.get("weather_humidity"),
        weather_rainfall=saved_result.get("weather_rainfall"),
        weather_latitude=saved_result.get("weather_latitude"),
        weather_longitude=saved_result.get("weather_longitude"),
        current_season=translate_season_name(current_season_raw, lang),
        current_season_raw=current_season_raw,
        sowing_guidance=get_crop_sowing_guidance(saved_result["prediction_raw"], lang),
        model_info=get_model_info(lang),
        show_feedback_prompt=feedback_prompt_visible,
        feedback_status=request.args.get("feedback") == "submitted",
        feedback_prediction_type="crop",
        feedback_prediction_name=saved_result["prediction_raw"],
        feedback_prediction_count=saved_result.get("prediction_count"),
        current_result_url=url_for("crop_result_page", lang=lang),
        title="पीक निकाल" if lang == "mr" else "Crop Result"
    )

def render_saved_fertilizer_result(saved_result, lang):
    return render_fertilizer_result_page(saved_result, lang)

    recommendation_key = saved_result.get("recommendation_key")

    if not recommendation_key:
        return redirect(url_for("fertilizer_recommendation", lang=lang))

    try:
        recommendation_text = get_fertilizer_recommendation_text(recommendation_key, lang)
    except KeyError:
        return redirect(url_for("fertilizer_recommendation", lang=lang))

    return render_template(
        "fertilizer-result.html",
        recommendation=Markup(str(recommendation_text)),
        lang=lang,
        ui=UI_TEXT[lang],
        current_result_url=url_for("fertilizer_result_page", lang=lang),
        title="à¤–à¤¤ à¤¨à¤¿à¤•à¤¾à¤²" if lang == "mr" else "Fertilizer Result"
    )

def render_fertilizer_result_page(saved_result, lang):
    recommendation_key = saved_result.get("recommendation_key")
    feedback_prompt_visible = saved_result.get("show_feedback_prompt", False) and not saved_result.get("feedback_submitted", False)

    if not recommendation_key:
        return redirect(url_for("fertilizer_recommendation", lang=lang))

    crop_name = saved_result.get("crop_name")
    reference_n = saved_result.get("reference_n")
    reference_p = saved_result.get("reference_p")
    reference_k = saved_result.get("reference_k")
    actual_n = saved_result.get("actual_n")
    actual_p = saved_result.get("actual_p")
    actual_k = saved_result.get("actual_k")

    if None in {reference_n, reference_p, reference_k, actual_n, actual_p, actual_k}:
        if not crop_name:
            return redirect(url_for("fertilizer_recommendation", lang=lang))

        fertilizer_df = get_fertilizer_reference_df()
        crop_row = fertilizer_df[fertilizer_df["Crop"] == crop_name]

        if crop_row.empty:
            return redirect(url_for("fertilizer_recommendation", lang=lang))

        reference_n = int(crop_row["N"].iloc[0])
        reference_p = int(crop_row["P"].iloc[0])
        reference_k = int(crop_row["K"].iloc[0])
        actual_n = reference_n
        actual_p = reference_p
        actual_k = reference_k

    plan = build_fertilizer_plan(
        crop_name=crop_name,
        reference_n=int(reference_n),
        reference_p=int(reference_p),
        reference_k=int(reference_k),
        actual_n=int(actual_n),
        actual_p=int(actual_p),
        actual_k=int(actual_k)
    )

    return render_template(
        "fertilizer-result.html",
        plan=plan,
        lang=lang,
        ui=UI_TEXT[lang],
        model_info=get_model_info(lang),
        show_feedback_prompt=feedback_prompt_visible,
        feedback_status=request.args.get("feedback") == "submitted",
        feedback_prediction_type="fertilizer",
        feedback_prediction_name=plan["crop_name"],
        feedback_prediction_count=saved_result.get("prediction_count"),
        current_result_url=url_for("fertilizer_result_page", lang=lang),
        title="\u0916\u0924 \u0928\u093f\u0915\u093e\u0932" if lang == "mr" else "Fertilizer Result"
    )

# ==============================================================================================
# ------------------------------------ FLASK APP ------------------------------------------------

app = Flask(__name__)
app.secret_key = os.environ.get("HARVESTIFY_SECRET_KEY") or secrets.token_hex(32)


@app.route('/')
def home():
    lang = get_lang()
    return render_template(
        'index.html',
        lang=lang,
        ui=UI_TEXT[lang],
        title="Harvestify"
    )


@app.route('/crop-recommend')
def crop_recommend():
    lang = get_lang()
    return render_template(
        'crop.html',
        lang=lang,
        ui=UI_TEXT[lang],
        crop_translations=CROP_TRANSLATIONS,
        season_translations=SEASON_TRANSLATIONS,
        water_source_options=WATER_SOURCE_OPTIONS,
        water_availability_options=WATER_AVAILABILITY_OPTIONS,
        title="पीक शिफारस" if lang == "mr" else "Crop Recommendation"
    )


@app.route('/crop-result')
def crop_result_page():
    lang = get_lang()
    saved_result = session.get("last_crop_result")

    if not saved_result:
        return redirect(url_for("crop_recommend", lang=lang))

    return render_saved_crop_result(saved_result, lang)


@app.route('/fertilizer')
def fertilizer_recommendation():
    lang = get_lang()
    fertilizer_reference_df = get_fertilizer_reference_df()
    return render_template(
        'fertilizer.html',
        lang=lang,
        ui=UI_TEXT[lang],
        crop_options=fertilizer_reference_df["Crop"].tolist(),
        crop_translations=CROP_TRANSLATIONS,
        title="खत शिफारस" if lang == "mr" else "Fertilizer Recommendation"
    )


@app.route('/fertilizer-result')
def fertilizer_result_page():
    lang = get_lang()
    saved_result = session.get("last_fertilizer_result")

    if not saved_result:
        return redirect(url_for("fertilizer_recommendation", lang=lang))

    return render_fertilizer_result_page(saved_result, lang)


@app.route('/prediction-feedback', methods=['POST'])
def prediction_feedback():
    lang = get_lang()
    prediction_type = str(request.form.get("prediction_type", "")).strip().lower()
    prediction_name = str(request.form.get("prediction_name", "")).strip()
    usefulness = str(request.form.get("usefulness", "")).strip().lower()
    chosen_crop = str(request.form.get("chosen_crop", "")).strip()
    season_result = str(request.form.get("season_result", "")).strip()
    prediction_count = str(request.form.get("prediction_count", "")).strip()
    return_path = str(request.form.get("return_path", "")).strip()

    if usefulness not in {"useful", "not_useful", ""}:
        usefulness = ""

    append_prediction_feedback_entry({
        "prediction_count": prediction_count,
        "prediction_type": prediction_type,
        "prediction_name": prediction_name,
        "usefulness": usefulness,
        "chosen_crop": chosen_crop,
        "season_result": season_result,
    })

    if prediction_type == "crop" and "last_crop_result" in session:
        saved_result = dict(session["last_crop_result"])
        saved_result["feedback_submitted"] = True
        saved_result["show_feedback_prompt"] = False
        session["last_crop_result"] = saved_result
    elif prediction_type == "fertilizer" and "last_fertilizer_result" in session:
        saved_result = dict(session["last_fertilizer_result"])
        saved_result["feedback_submitted"] = True
        saved_result["show_feedback_prompt"] = False
        session["last_fertilizer_result"] = saved_result

    return redirect(build_feedback_redirect_path(return_path, lang))


@app.route('/disease')
def disease():
    lang = get_lang()
    return render_template(
        'disease.html',
        lang=lang,
        ui=UI_TEXT[lang],
        title="Disease Detection"
    )

# ==============================================================================================
# ------------------------------------ SERVICE WORKER -------------------------------------------

@app.route('/service-worker.js')
def service_worker():
    response = send_from_directory('static/js', 'service-worker.js')
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Service-Worker-Allowed"] = "/"
    return response

# ==============================================================================================
# ------------------------------------ WEATHER API FOR UI ---------------------------------------

@app.route('/get-weather-data', methods=['POST'])
def get_weather_data():
    lang = get_lang()

    try:
        data = request.get_json(silent=True) or {}

        latitude = str(data.get("latitude", "")).strip()
        longitude = str(data.get("longitude", "")).strip()
        manual_location = str(data.get("manual_location", "")).strip()
        location_method = str(data.get("location_method", "current")).strip()

        weather_info = resolve_location_and_weather(
            lang=lang,
            latitude=latitude,
            longitude=longitude,
            manual_location=manual_location,
            location_method=location_method
        )

        return jsonify({
            "success": True,
            "city": weather_info["city"],
            "temperature": weather_info["temperature"],
            "humidity": weather_info["humidity"],
            "rainfall": weather_info["rainfall"],
            "latitude": weather_info["latitude"],
            "longitude": weather_info["longitude"]
        })

    except Exception as e:
        print("WEATHER API ERROR:", e)
        return jsonify({
            "success": False,
            "message": str(e)
        }), 400

# ==============================================================================================
# ------------------------------------ PREDICTIONS ----------------------------------------------

@app.route('/crop-predict', methods=['POST', 'GET'])
def crop_prediction():
    lang = get_lang()

    if request.method == 'GET':
        return redirect(url_for("crop_recommend", lang=lang))

        return render_template(
            "crop-result.html",
            lang=lang,
            ui=UI_TEXT[lang],
            prediction=translate_crop_name(saved_result["prediction_raw"], lang),
            prediction_raw=saved_result["prediction_raw"],
            top3_predictions=[
                {
                    "crop": translate_crop_name(item["crop"], lang),
                    "confidence": item["confidence"],
                    "actual_confidence": item.get("actual_confidence", item["confidence"]),
                    "season_bonus": item.get("season_bonus", 0),
                    "season_adjusted_score": item.get("season_adjusted_score", item.get("actual_confidence", item["confidence"]))
                }
                for item in saved_result["top3_predictions"]
            ],
            top3_predictions_raw=saved_result["top3_predictions"],
            explanation=pd.DataFrame(saved_result["explanation"]),
            explanations=[
                translate_explanation_line(line, saved_result["prediction_raw"], lang)
                for line in saved_result["explanations"]
            ],
            overall_summary=generate_overall_summary(
                saved_result["prediction_raw"],
                saved_result["top3_predictions"],
                pd.DataFrame(saved_result["explanation"]),
                lang
            ),
            land_area=saved_result["land_area"],
            land_unit_display=("एकर" if saved_result["land_unit"] == "acre" else "हेक्टर") if lang == "mr"
                              else ("Acre" if saved_result["land_unit"] == "acre" else "Hectare"),
            estimated_yield=saved_result["estimated_yield"],
            estimated_profit=saved_result["estimated_profit"],
            yield_profit_summary=saved_result["yield_profit_summary_mr"] if lang == "mr"
                                else saved_result["yield_profit_summary_en"],
            live_modal_price=saved_result["live_modal_price"],
            market_name=saved_result["market_name"],
            price_date=saved_result["price_date"],
            weather_city=saved_result.get("weather_city"),
            weather_temperature=saved_result.get("weather_temperature"),
            weather_humidity=saved_result.get("weather_humidity"),
            weather_rainfall=saved_result.get("weather_rainfall"),
            weather_latitude=saved_result.get("weather_latitude"),
            weather_longitude=saved_result.get("weather_longitude"),
            current_season=saved_result.get("current_season"),
            current_season_raw=saved_result.get("current_season"),
            sowing_guidance=saved_result.get("sowing_guidance"),
            title="Crop Result"
        )

    try:
        validated_input = validate_crop_form_input(request.form)
        N = validated_input["N"]
        P = validated_input["P"]
        K = validated_input["K"]
        ph = validated_input["ph"]
        land_area = validated_input["land_area"]
        land_unit = validated_input["land_unit"]
        water_source = validated_input["water_source"]
        water_availability = validated_input["water_availability"]
        user_rainfall = validated_input["user_rainfall"]
        latitude = validated_input["latitude"]
        longitude = validated_input["longitude"]
        manual_location = validated_input["manual_location"]
        location_method = validated_input["location_method"]
        prefetched_weather_city = validated_input["weather_city"]
        prefetched_temperature = validated_input["weather_temperature"]
        prefetched_humidity = validated_input["weather_humidity"]

        has_prefetched_weather = (
            is_valid_coordinate_pair(latitude, longitude)
            and prefetched_temperature is not None
            and prefetched_humidity is not None
            and user_rainfall is not None
        )

        if has_prefetched_weather:
            city = prefetched_weather_city or manual_location or "Selected location"
            temperature = prefetched_temperature
            humidity = prefetched_humidity
            rainfall = user_rainfall
            final_latitude = round(float(latitude), 6)
            final_longitude = round(float(longitude), 6)
        else:
            weather_info = resolve_location_and_weather(
                lang=lang,
                latitude=latitude,
                longitude=longitude,
                manual_location=manual_location,
                location_method=location_method
            )

            city = weather_info["city"]
            temperature = weather_info["temperature"]
            humidity = weather_info["humidity"]
            api_rainfall = weather_info["rainfall"]
            final_latitude = weather_info["latitude"]
            final_longitude = weather_info["longitude"]
            rainfall = api_rainfall if api_rainfall is not None else user_rainfall

        if rainfall is None:
            raise ValueError(UI_TEXT[lang]["weather_fetch_error"])

        current_season = get_current_crop_season()

        input_df = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall, current_season, water_source, water_availability]],
            columns=[
                "N",
                "P",
                "K",
                "temperature",
                "humidity",
                "ph",
                "rainfall",
                "season",
                "water_source",
                "water_availability"
            ]
        )

        print("\n========== FINAL MODEL INPUT ==========")
        print("Latitude :", final_latitude)
        print("Longitude:", final_longitude)
        print("City     :", city)
        print("Temperature used:", temperature)
        print("Humidity used   :", humidity)
        print("Rainfall used   :", rainfall)
        print(input_df)
        print("=======================================\n")

        crop_xai_runtime = get_crop_xai_runtime()
        merged_df, shap_df, explanations, prediction, top3_predictions = crop_xai_runtime["explain_crop_prediction"](input_df)

        current_season, top3_predictions = rerank_top3_predictions_by_season(
            top3_predictions,
            season=current_season
        )

        final_prediction = top3_predictions[0]["crop"]
        if final_prediction != prediction:
            merged_df, shap_df, explanations, prediction, top3_predictions = crop_xai_runtime["explain_specific_crop_prediction"](
                input_df,
                final_prediction,
                top3_predictions=top3_predictions
            )

        print("\n========== MODEL OUTPUT ==========")
        print("Prediction:", prediction)
        print("Season:", current_season)
        print("Season Re-ranked Top 3:", top3_predictions)
        print("==================================\n")

        market_data = get_msamb_live_price(prediction, city)
        live_modal_price_num = market_data["modal_price"] if market_data else None

        prediction_display, top3_display, explanation_display, explanations_display = localize_crop_output(
            prediction, top3_predictions, merged_df, explanations, lang
        )

        overall_summary = generate_overall_summary(
            prediction,
            top3_predictions,
            merged_df,
            lang
        )

        estimated_yield, estimated_profit_num = estimate_yield_and_profit(
            prediction,
            land_area,
            land_unit
        )

        if live_modal_price_num is not None:
            estimated_profit_num = round(estimated_yield * live_modal_price_num * 0.65, 2)

        if lang == "mr":
            land_unit_display = "एकर" if land_unit == "acre" else "हेक्टर"
        else:
            land_unit_display = "Acre" if land_unit == "acre" else "Hectare"

        if market_data:
            market_name = market_data.get("market", "")
            price_date = market_data.get("date", "")
            live_price_display = f"₹{live_modal_price_num}"
        else:
            market_name = "-"
            price_date = "-"
            live_price_display = "N/A"

        yield_profit_summary_en = (
            f"Based on the given land area, the estimated yield of {prediction} can be around "
            f"{estimated_yield} {UI_TEXT['en']['yield_unit']}. Based on the current market price, "
            f"the estimated profit may be around ₹{estimated_profit_num}. Actual yield and profit may vary "
            f"depending on weather, farming practices, and market conditions."
        )

        prediction_mr = translate_crop_name(prediction, "mr")
        yield_profit_summary_mr = (
            f"दिलेल्या जमिनीच्या क्षेत्रफळानुसार {prediction_mr} या पिकाचे अंदाजे उत्पादन "
            f"{estimated_yield} {UI_TEXT['mr']['yield_unit']} इतके होऊ शकते. "
            f"सध्याच्या बाजारभावानुसार अंदाजे नफा ₹{estimated_profit_num} इतका मिळू शकतो. "
            f"प्रत्यक्ष उत्पादन आणि नफा हवामान, शेती पद्धती आणि बाजारभावानुसार बदलू शकतो."
        )

        estimated_profit_display = f"₹{estimated_profit_num}"


        print("Guidance crop name received:", prediction)
        sowing_guidance = get_crop_sowing_guidance(prediction, lang)
        feedback_prompt = register_prediction_feedback_prompt()

        session["last_crop_result"] = {
            "prediction_raw": prediction,
            "top3_predictions": top3_predictions,
            "explanation": merged_df.to_dict(orient="records"),
            "explanations": explanations,
            "land_area": land_area,
            "land_unit": land_unit,
            "estimated_yield": estimated_yield,
            "estimated_profit": estimated_profit_display,
            "yield_profit_summary_en": yield_profit_summary_en,
            "yield_profit_summary_mr": yield_profit_summary_mr,
            "live_modal_price": live_price_display,
            "market_name": market_name,
            "price_date": price_date,
            "weather_city": city,
            "weather_temperature": temperature,
            "weather_humidity": humidity,
            "weather_rainfall": rainfall,
            "weather_latitude": final_latitude,
            "weather_longitude": final_longitude,
            "current_season": current_season,
            "sowing_guidance": sowing_guidance,
            "prediction_count": feedback_prompt["prediction_count"],
            "show_feedback_prompt": feedback_prompt["show_feedback_prompt"],
            "feedback_submitted": False
        }

        return render_template(
            "crop-result.html",
            lang=lang,
            ui=UI_TEXT[lang],
            prediction=prediction_display,
            prediction_raw=prediction,
            top3_predictions=top3_display,
            top3_predictions_raw=top3_predictions,
            explanation=explanation_display,
            explanations=explanations_display,
            overall_summary=overall_summary,
            land_area=land_area,
            land_unit_display=land_unit_display,
            estimated_yield=estimated_yield,
            estimated_profit=estimated_profit_display,
            yield_profit_summary=yield_profit_summary_mr if lang == "mr" else yield_profit_summary_en,
            live_modal_price=live_price_display,
            market_name=market_name,
            price_date=price_date,
            weather_city=city,
            weather_temperature=temperature,
            weather_humidity=humidity,
            weather_rainfall=rainfall,
            weather_latitude=final_latitude,
            weather_longitude=final_longitude,
            current_season=translate_season_name(current_season, lang),
            current_season_raw=current_season,
            sowing_guidance=sowing_guidance,
            model_info=get_model_info(lang),
            show_feedback_prompt=feedback_prompt["show_feedback_prompt"],
            feedback_status=False,
            feedback_prediction_type="crop",
            feedback_prediction_name=prediction,
            feedback_prediction_count=feedback_prompt["prediction_count"],
            current_result_url=url_for("crop_result_page", lang=lang),
            title="पीक निकाल" if lang == "mr" else "Crop Result"
        )

    except ValidationError as e:
        print("VALIDATION ERROR:", e)
        session.pop("last_crop_result", None)
        return render_input_error(lang, str(e), retry_endpoint="crop_recommend")
    except Exception as e:
        print("ERROR:", e)
        session.pop("last_crop_result", None)
        return render_input_error(lang, str(e), retry_endpoint="crop_recommend")

# ==============================================================================================
# Fertilizer prediction

@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    try:
        lang = get_lang()

        df = get_fertilizer_reference_df()
        validated_input = validate_fertilizer_form_input(request.form, df)
        crop_name = validated_input["crop_name"]
        N = validated_input["N"]
        P = validated_input["P"]
        K = validated_input["K"]

        nr = df[df['Crop'] == crop_name]['N'].iloc[0]
        pr = df[df['Crop'] == crop_name]['P'].iloc[0]
        kr = df[df['Crop'] == crop_name]['K'].iloc[0]

        key = get_fertilizer_recommendation_key(
            reference_n=nr,
            reference_p=pr,
            reference_k=kr,
            actual_n=N,
            actual_p=P,
            actual_k=K
        )
        feedback_prompt = register_prediction_feedback_prompt()

        session["last_fertilizer_result"] = {
            "crop_name": crop_name,
            "recommendation_key": key,
            "reference_n": int(nr),
            "reference_p": int(pr),
            "reference_k": int(kr),
            "actual_n": int(N),
            "actual_p": int(P),
            "actual_k": int(K),
            "prediction_count": feedback_prompt["prediction_count"],
            "show_feedback_prompt": feedback_prompt["show_feedback_prompt"],
            "feedback_submitted": False
        }

        return render_fertilizer_result_page(session["last_fertilizer_result"], lang)

        if lang == "mr":
            recommendation = Markup(str(fertilizer_dic_mr[key]))
        else:
            recommendation = Markup(str(fertilizer_dic[key]))

        return render_template(
            'fertilizer-result.html',
            recommendation=recommendation,
            lang=lang,
            ui=UI_TEXT[lang],
            title="खत निकाल" if lang == "mr" else "Fertilizer Result"
        )

    except ValidationError as e:
        session.pop("last_fertilizer_result", None)
        return render_input_error(get_lang(), str(e), retry_endpoint="fertilizer_recommendation")
    except Exception as e:
        session.pop("last_fertilizer_result", None)
        return render_input_error(get_lang(), str(e), retry_endpoint="fertilizer_recommendation")

# ==============================================================================================
# Disease prediction

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    try:
        if request.method == 'POST':
            file = request.files.get('file')
            img = validate_disease_upload(file)
            prediction = predict_image(img)
            prediction = Markup(str(disease_dic[prediction]))

            return render_template(
                'disease-result.html',
                prediction=prediction,
                lang=get_lang(),
                ui=UI_TEXT[get_lang()],
                title="Disease Result"
            )

        return render_template(
            'disease.html',
            lang=get_lang(),
            ui=UI_TEXT[get_lang()],
            title="Disease Detection"
        )

    except ValidationError as e:
        return render_input_error(get_lang(), str(e), retry_endpoint="disease")
    except Exception as e:
        print("ERROR:", e)
        return render_input_error(get_lang(), str(e), retry_endpoint="disease")

# ==============================================================================================

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    debug_enabled = os.environ.get("HARVESTIFY_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    app.run(host="0.0.0.0", port=port, debug=debug_enabled)
