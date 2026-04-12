import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import shap
except Exception:
    shap = None

from utils.crop_features import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS as DEFAULT_FEATURE_COLUMNS,
    FEATURE_DISPLAY_NAMES as DEFAULT_FEATURE_NAMES,
    encode_feature_frame,
)


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
LEGACY_MODEL_PATH = MODELS_DIR / "xgboost_crop_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
METADATA_PATH = MODELS_DIR / "crop_model_metadata.json"


def resolve_model_path():
    if MODEL_PATH.exists():
        return MODEL_PATH
    return LEGACY_MODEL_PATH


def load_artifact(path_obj):
    return joblib.load(str(path_obj))


def load_metadata():
    if not METADATA_PATH.exists():
        return {}

    with open(str(METADATA_PATH), "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


model = load_artifact(resolve_model_path())
scaler = load_artifact(SCALER_PATH)
label_encoder = load_artifact(LABEL_ENCODER_PATH)
metadata = load_metadata()

FEATURE_COLUMNS = metadata.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
FEATURE_NAMES = metadata.get("feature_display_names", DEFAULT_FEATURE_NAMES)
FEATURE_REFERENCE = metadata.get("feature_reference", {})
USES_SCALER = bool(metadata.get("uses_scaler", True))

explainer = None
if shap is not None:
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = None


def transform_input(input_df):
    ordered_df = input_df[FEATURE_COLUMNS].copy()
    encoded_df = encode_feature_frame(ordered_df)

    if USES_SCALER and scaler is not None and hasattr(scaler, "transform"):
        return scaler.transform(encoded_df)

    return encoded_df.to_numpy(dtype=float)


def predict_probabilities_from_df(input_df):
    transformed = transform_input(input_df)
    probabilities = model.predict_proba(transformed)[0]
    return transformed, probabilities


def get_top_predictions(probabilities):
    top3_indices = np.argsort(probabilities)[::-1][:3]
    top3_probs = probabilities[top3_indices]
    top3_sum = np.sum(top3_probs)

    if top3_sum == 0:
        normalized_top3_probs = [0.0] * len(top3_indices)
    else:
        normalized_top3_probs = (top3_probs / top3_sum) * 100

    top3_predictions = []
    for rank, idx in enumerate(top3_indices):
        crop_name = label_encoder.inverse_transform([idx])[0]
        original_confidence = float(probabilities[idx]) * 100
        relative_confidence = float(normalized_top3_probs[rank])

        top3_predictions.append(
            {
                "crop": crop_name,
                "confidence": round(relative_confidence, 2),
                "actual_confidence": round(original_confidence, 2),
            }
        )

    best_index = int(top3_indices[0])
    best_prediction = label_encoder.inverse_transform([best_index])[0]
    return best_index, best_prediction, top3_predictions


def get_class_index(crop_name):
    matches = np.where(label_encoder.classes_ == crop_name)[0]
    if len(matches) == 0:
        raise ValueError("Crop class not found: {0}".format(crop_name))
    return int(matches[0])


def get_reference_value(feature_name, fallback_value):
    reference = FEATURE_REFERENCE.get(feature_name, {})
    if feature_name in CATEGORICAL_FEATURE_COLUMNS:
        if "mode" in reference:
            return str(reference["mode"])
        return str(fallback_value)
    if "mean" in reference:
        return float(reference["mean"])
    if "median" in reference:
        return float(reference["median"])
    return float(fallback_value)


def build_fallback_impacts(input_df, target_index):
    _, base_probabilities = predict_probabilities_from_df(input_df)
    base_probability = float(base_probabilities[target_index])
    impacts = []

    for feature_name in FEATURE_COLUMNS:
        perturbed_df = input_df[FEATURE_COLUMNS].copy()
        current_value = perturbed_df.iloc[0][feature_name]
        if feature_name not in CATEGORICAL_FEATURE_COLUMNS:
            perturbed_df[feature_name] = pd.to_numeric(perturbed_df[feature_name], errors="coerce").astype(float)
        perturbed_df.at[perturbed_df.index[0], feature_name] = get_reference_value(feature_name, current_value)

        _, perturbed_probabilities = predict_probabilities_from_df(perturbed_df)
        perturbed_probability = float(perturbed_probabilities[target_index])
        impacts.append(base_probability - perturbed_probability)

    return [float(value) for value in impacts]


def get_feature_impacts(input_scaled, input_df, target_index):
    if explainer is not None:
        try:
            shap_values = explainer.shap_values(input_scaled, check_additivity=False)

            try:
                values = shap_values[target_index][0]
            except Exception:
                values = shap_values[:, :, target_index][0]

            return [float(value) for value in values]
        except Exception:
            pass

    return build_fallback_impacts(input_df, target_index)


def format_feature_value(feature_name, feature_value):
    if feature_name in CATEGORICAL_FEATURE_COLUMNS:
        return str(feature_value)
    if feature_name in ["N", "P", "K"]:
        return str(int(round(float(feature_value))))
    return "{0:.2f}".format(float(feature_value))


def build_human_explanation(feature_key, feature_label, feature_value, prediction_name, impact_value):
    display_value = format_feature_value(feature_key, feature_value)

    if impact_value > 0.0001:
        return (
            "{0} positively influenced the recommendation of {1}; "
            "the observed value ({2}) is favorable for this crop.".format(
                feature_label,
                prediction_name,
                display_value,
            )
        )
    if impact_value < -0.0001:
        return (
            "{0} negatively affected the suitability of {1}; "
            "the observed value ({2}) is less aligned with this crop profile.".format(
                feature_label,
                prediction_name,
                display_value,
            )
        )
    return "{0} had a neutral effect on the prediction; the observed value was {1}.".format(
        feature_label,
        display_value,
    )


def build_explanation_output(prediction_name, top3_predictions, impact_values, input_df):
    observed_values = [input_df.iloc[0][feature_name] for feature_name in FEATURE_COLUMNS]
    shap_df = pd.DataFrame(
        {
            "Feature": FEATURE_NAMES,
            "Impact Value": impact_values,
            "Observed Value": [format_feature_value(feature_name, value) for feature_name, value in zip(FEATURE_COLUMNS, observed_values)],
        }
    )

    def effect_label(value):
        if value > 0.0001:
            return "Positive (Helps crop)"
        if value < -0.0001:
            return "Negative (Harms crop)"
        return "Neutral"

    shap_df["Effect"] = shap_df["Impact Value"].apply(effect_label)
    shap_df["Human Explanation"] = [
        build_human_explanation(feature_key, feature_label, feature_value, prediction_name, impact_value)
        for feature_key, feature_label, feature_value, impact_value in zip(
            FEATURE_COLUMNS,
            FEATURE_NAMES,
            observed_values,
            impact_values,
        )
    ]
    shap_df = shap_df.reindex(shap_df["Impact Value"].abs().sort_values(ascending=False).index).reset_index(drop=True)

    explanations = shap_df["Human Explanation"].tolist()
    merged_df = shap_df[["Feature", "Impact Value", "Effect"]].copy()
    return merged_df, shap_df, explanations, prediction_name, top3_predictions


def explain_crop_prediction(input_df):
    input_scaled, probabilities = predict_probabilities_from_df(input_df)
    best_index, best_prediction, top3_predictions = get_top_predictions(probabilities)
    impact_values = get_feature_impacts(input_scaled, input_df, best_index)
    return build_explanation_output(best_prediction, top3_predictions, impact_values, input_df)


def explain_specific_crop_prediction(input_df, crop_name, top3_predictions=None):
    input_scaled, probabilities = predict_probabilities_from_df(input_df)
    if top3_predictions is None:
        _, _, top3_predictions = get_top_predictions(probabilities)

    target_index = get_class_index(crop_name)
    impact_values = get_feature_impacts(input_scaled, input_df, target_index)
    return build_explanation_output(crop_name, top3_predictions, impact_values, input_df)
