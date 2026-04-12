from collections import OrderedDict

import pandas as pd


NUMERIC_FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
CATEGORICAL_FEATURE_COLUMNS = ["season", "water_source", "water_availability"]
FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
FEATURE_DISPLAY_NAMES = [
    "Nitrogen",
    "Phosphorous",
    "Potassium",
    "Temperature",
    "Humidity",
    "pH",
    "Rainfall",
    "Season",
    "Water Source",
    "Water Availability",
]

TARGET_COLUMN = "label"
REGION_COLUMN = "region"
REGION_VALUE = "Maharashtra"

SEASON_OPTIONS = ["Kharif", "Rabi", "Summer"]
WATER_SOURCE_OPTIONS = ["rainfed", "borewell", "canal", "drip"]
WATER_AVAILABILITY_OPTIONS = ["low", "medium", "high"]

SEASON_ENCODING = OrderedDict((value, index) for index, value in enumerate(SEASON_OPTIONS))
WATER_SOURCE_ENCODING = OrderedDict((value, index) for index, value in enumerate(WATER_SOURCE_OPTIONS))
WATER_AVAILABILITY_ENCODING = OrderedDict(
    (value, index) for index, value in enumerate(WATER_AVAILABILITY_OPTIONS)
)

EXISTING_MAHARASHTRA_CROPS = [
    "sugarcane",
    "rice",
    "soybean",
    "cotton",
    "jowar",
    "bajra",
    "wheat",
    "maize",
    "groundnut",
    "tur",
    "gram",
    "onion",
    "tomato",
    "chili",
    "turmeric",
]

NEW_MAHARASHTRA_CROPS = [
    "banana",
    "orange",
    "grapes",
    "pomegranate",
    "ginger",
    "potato",
    "brinjal",
    "cabbage",
    "sunflower",
    "sesame",
    "blackgram",
    "mungbean",
    "ragi",
    "cowpea",
    "lentil",
]

CANONICAL_CROP_LABELS = EXISTING_MAHARASHTRA_CROPS + NEW_MAHARASHTRA_CROPS

CROP_LABEL_ALIASES = {
    "sugar cane": "sugarcane",
    "sugarcane": "sugarcane",
    "rice": "rice",
    "soybean": "soybean",
    "cotton": "cotton",
    "jowar": "jowar",
    "sorghum": "jowar",
    "bajra": "bajra",
    "millet": "bajra",
    "pearl millet": "bajra",
    "pearl_millet": "bajra",
    "wheat": "wheat",
    "maize": "maize",
    "ground nut": "groundnut",
    "groundnut": "groundnut",
    "tur": "tur",
    "tur dal": "tur",
    "pigeonpea": "tur",
    "pigeonpeas": "tur",
    "pigeon peas": "tur",
    "gram": "gram",
    "gram dal": "gram",
    "chickpea": "gram",
    "chick pea": "gram",
    "onion": "onion",
    "tomato": "tomato",
    "chilli": "chili",
    "chili": "chili",
    "turmeric": "turmeric",
    "banana": "banana",
    "orange": "orange",
    "grape": "grapes",
    "grapes": "grapes",
    "pomegranate": "pomegranate",
    "ginger": "ginger",
    "potato": "potato",
    "brinjal": "brinjal",
    "eggplant": "brinjal",
    "aubergine": "brinjal",
    "cabbage": "cabbage",
    "sunflower": "sunflower",
    "sesame": "sesame",
    "sesame (til)": "sesame",
    "til": "sesame",
    "black gram": "blackgram",
    "blackgram": "blackgram",
    "urad": "blackgram",
    "mung bean": "mungbean",
    "mungbean": "mungbean",
    "moong": "mungbean",
    "green gram": "mungbean",
    "ragi": "ragi",
    "finger millet": "ragi",
    "finger millet (ragi)": "ragi",
    "nachni": "ragi",
    "cowpea": "cowpea",
    "lobia": "cowpea",
    "lentil": "lentil",
    "masoor": "lentil",
}

CROP_CONTEXT_PROFILES = {
    "sugarcane": {
        "primary_season": "Summer",
        "valid_seasons": ["Summer", "Rabi"],
        "water_source": "canal",
        "water_availability": "high",
    },
    "rice": {
        "primary_season": "Kharif",
        "valid_seasons": ["Kharif", "Summer"],
        "water_source": "canal",
        "water_availability": "high",
    },
    "soybean": {
        "primary_season": "Kharif",
        "valid_seasons": ["Kharif"],
        "water_source": "rainfed",
        "water_availability": "medium",
    },
    "cotton": {
        "primary_season": "Kharif",
        "valid_seasons": ["Kharif"],
        "water_source": "borewell",
        "water_availability": "medium",
    },
    "jowar": {
        "primary_season": "Kharif",
        "valid_seasons": ["Kharif", "Rabi"],
        "water_source": "rainfed",
        "water_availability": "low",
    },
    "bajra": {
        "primary_season": "Kharif",
        "valid_seasons": ["Kharif", "Summer"],
        "water_source": "rainfed",
        "water_availability": "low",
    },
    "wheat": {
        "primary_season": "Rabi",
        "valid_seasons": ["Rabi"],
        "water_source": "borewell",
        "water_availability": "medium",
    },
    "maize": {
        "primary_season": "Kharif",
        "valid_seasons": ["Kharif", "Rabi"],
        "water_source": "canal",
        "water_availability": "medium",
    },
    "groundnut": {
        "primary_season": "Kharif",
        "valid_seasons": ["Kharif", "Summer"],
        "water_source": "rainfed",
        "water_availability": "low",
    },
    "tur": {
        "primary_season": "Kharif",
        "valid_seasons": ["Kharif"],
        "water_source": "rainfed",
        "water_availability": "low",
    },
    "gram": {
        "primary_season": "Rabi",
        "valid_seasons": ["Rabi"],
        "water_source": "rainfed",
        "water_availability": "low",
    },
    "onion": {
        "primary_season": "Rabi",
        "valid_seasons": ["Rabi", "Summer"],
        "water_source": "drip",
        "water_availability": "medium",
    },
    "tomato": {
        "primary_season": "Rabi",
        "valid_seasons": ["Rabi", "Summer"],
        "water_source": "drip",
        "water_availability": "medium",
    },
    "chili": {
        "primary_season": "Kharif",
        "valid_seasons": ["Kharif", "Rabi"],
        "water_source": "drip",
        "water_availability": "medium",
    },
    "turmeric": {
        "primary_season": "Kharif",
        "valid_seasons": ["Kharif"],
        "water_source": "borewell",
        "water_availability": "medium",
    },
    "banana": {
        "primary_season": "Kharif",
        "valid_seasons": ["Kharif", "Summer"],
        "water_source": "drip",
        "water_availability": "high",
    },
    "orange": {
        "primary_season": "Kharif",
        "valid_seasons": ["Kharif", "Rabi"],
        "water_source": "drip",
        "water_availability": "medium",
    },
    "grapes": {
        "primary_season": "Rabi",
        "valid_seasons": ["Rabi", "Summer"],
        "water_source": "drip",
        "water_availability": "high",
    },
    "pomegranate": {
        "primary_season": "Kharif",
        "valid_seasons": ["Kharif", "Summer"],
        "water_source": "drip",
        "water_availability": "medium",
    },
    "ginger": {
        "primary_season": "Kharif",
        "valid_seasons": ["Kharif"],
        "water_source": "borewell",
        "water_availability": "high",
    },
    "potato": {
        "primary_season": "Rabi",
        "valid_seasons": ["Rabi"],
        "water_source": "borewell",
        "water_availability": "medium",
    },
    "brinjal": {
        "primary_season": "Rabi",
        "valid_seasons": ["Kharif", "Rabi"],
        "water_source": "drip",
        "water_availability": "medium",
    },
    "cabbage": {
        "primary_season": "Rabi",
        "valid_seasons": ["Rabi", "Summer"],
        "water_source": "borewell",
        "water_availability": "medium",
    },
    "sunflower": {
        "primary_season": "Rabi",
        "valid_seasons": ["Kharif", "Rabi"],
        "water_source": "rainfed",
        "water_availability": "low",
    },
    "sesame": {
        "primary_season": "Kharif",
        "valid_seasons": ["Kharif", "Summer"],
        "water_source": "rainfed",
        "water_availability": "low",
    },
    "blackgram": {
        "primary_season": "Kharif",
        "valid_seasons": ["Kharif", "Summer"],
        "water_source": "rainfed",
        "water_availability": "low",
    },
    "mungbean": {
        "primary_season": "Summer",
        "valid_seasons": ["Summer", "Kharif"],
        "water_source": "rainfed",
        "water_availability": "low",
    },
    "ragi": {
        "primary_season": "Kharif",
        "valid_seasons": ["Kharif"],
        "water_source": "rainfed",
        "water_availability": "medium",
    },
    "cowpea": {
        "primary_season": "Kharif",
        "valid_seasons": ["Kharif", "Summer"],
        "water_source": "rainfed",
        "water_availability": "low",
    },
    "lentil": {
        "primary_season": "Rabi",
        "valid_seasons": ["Rabi"],
        "water_source": "rainfed",
        "water_availability": "low",
    },
}


def normalize_crop_label(value):
    if pd.isna(value):
        return ""
    raw_value = str(value).strip().lower()
    return CROP_LABEL_ALIASES.get(raw_value, raw_value)


def normalize_season(value):
    if pd.isna(value):
        return ""
    raw_value = str(value).strip().lower()
    season_map = {
        "kharif": "Kharif",
        "rabi": "Rabi",
        "summer": "Summer"
    }
    return season_map.get(raw_value, str(value).strip())


def normalize_water_source(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def normalize_water_availability(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def get_crop_context_profile(label):
    canonical_label = normalize_crop_label(label)
    if canonical_label not in CROP_CONTEXT_PROFILES:
        raise ValueError("Unsupported crop label: {0}".format(label))
    return canonical_label, CROP_CONTEXT_PROFILES[canonical_label]


def ensure_context_columns(dataset):
    enriched_df = dataset.copy()
    enriched_df[TARGET_COLUMN] = enriched_df[TARGET_COLUMN].apply(normalize_crop_label)

    canonical_labels = enriched_df[TARGET_COLUMN]
    profiles = canonical_labels.map(lambda crop_name: get_crop_context_profile(crop_name)[1])

    enriched_df[REGION_COLUMN] = REGION_VALUE

    if "season" in enriched_df.columns:
        enriched_df["season"] = enriched_df["season"].apply(normalize_season)
        season_missing = enriched_df["season"].astype(str).str.strip() == ""
        enriched_df.loc[season_missing, "season"] = canonical_labels[season_missing].map(
            lambda crop_name: get_crop_context_profile(crop_name)[1]["primary_season"]
        )
    else:
        enriched_df["season"] = canonical_labels.map(
            lambda crop_name: get_crop_context_profile(crop_name)[1]["primary_season"]
        )

    if "water_source" in enriched_df.columns:
        enriched_df["water_source"] = enriched_df["water_source"].apply(normalize_water_source)
        source_missing = enriched_df["water_source"].astype(str).str.strip() == ""
        enriched_df.loc[source_missing, "water_source"] = canonical_labels[source_missing].map(
            lambda crop_name: get_crop_context_profile(crop_name)[1]["water_source"]
        )
    else:
        enriched_df["water_source"] = profiles.map(lambda profile: profile["water_source"])

    if "water_availability" in enriched_df.columns:
        enriched_df["water_availability"] = enriched_df["water_availability"].apply(
            normalize_water_availability
        )
        availability_missing = enriched_df["water_availability"].astype(str).str.strip() == ""
        enriched_df.loc[availability_missing, "water_availability"] = canonical_labels[
            availability_missing
        ].map(lambda crop_name: get_crop_context_profile(crop_name)[1]["water_availability"])
    else:
        enriched_df["water_availability"] = profiles.map(lambda profile: profile["water_availability"])

    return enriched_df


def ensure_irrigation_columns(dataset):
    return ensure_context_columns(dataset)


def encode_feature_frame(dataframe):
    feature_df = dataframe.copy()

    for column in NUMERIC_FEATURE_COLUMNS:
        feature_df[column] = pd.to_numeric(feature_df[column], errors="raise")

    feature_df["season"] = feature_df["season"].apply(normalize_season)
    feature_df["water_source"] = feature_df["water_source"].apply(normalize_water_source)
    feature_df["water_availability"] = feature_df["water_availability"].apply(
        normalize_water_availability
    )

    invalid_seasons = sorted(set(feature_df["season"]) - set(SEASON_OPTIONS))
    invalid_sources = sorted(set(feature_df["water_source"]) - set(WATER_SOURCE_OPTIONS))
    invalid_levels = sorted(set(feature_df["water_availability"]) - set(WATER_AVAILABILITY_OPTIONS))

    if invalid_seasons:
        raise ValueError("Unsupported season values: {0}".format(", ".join(invalid_seasons)))
    if invalid_sources:
        raise ValueError("Unsupported water source values: {0}".format(", ".join(invalid_sources)))
    if invalid_levels:
        raise ValueError(
            "Unsupported water availability values: {0}".format(", ".join(invalid_levels))
        )

    encoded_df = feature_df[NUMERIC_FEATURE_COLUMNS].copy()
    encoded_df["season"] = feature_df["season"].map(SEASON_ENCODING).astype(float)
    encoded_df["water_source"] = feature_df["water_source"].map(WATER_SOURCE_ENCODING).astype(float)
    encoded_df["water_availability"] = feature_df["water_availability"].map(
        WATER_AVAILABILITY_ENCODING
    ).astype(float)
    return encoded_df[FEATURE_COLUMNS]


def build_feature_reference(feature_df):
    reference = OrderedDict()

    for column in NUMERIC_FEATURE_COLUMNS:
        reference[column] = {
            "mean": round(float(feature_df[column].mean()), 6),
            "median": round(float(feature_df[column].median()), 6),
            "std": round(float(feature_df[column].std(ddof=0)), 6)
        }

    for column in CATEGORICAL_FEATURE_COLUMNS:
        reference[column] = {"mode": str(feature_df[column].mode().iloc[0])}

    return reference
