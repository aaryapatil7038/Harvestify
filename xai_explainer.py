import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import shap
except Exception:
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None

from utils.crop_features import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS as DEFAULT_FEATURE_COLUMNS,
    FEATURE_DISPLAY_NAMES as DEFAULT_FEATURE_NAMES,
    SEASON_OPTIONS,
    WATER_AVAILABILITY_OPTIONS,
    WATER_SOURCE_OPTIONS,
    encode_feature_frame,
)


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
LEGACY_MODEL_PATH = MODELS_DIR / "xgboost_crop_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
METADATA_PATH = MODELS_DIR / "crop_model_metadata.json"

EPSILON = 0.0001
MAX_LIME_FEATURES = 6
LIME_BACKGROUND_ROWS = 400
LIME_SYNTHETIC_ROWS = 120


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


@lru_cache(maxsize=1)
def get_shap_explainer():
    if shap is None:
        return None

    try:
        return shap.TreeExplainer(model)
    except Exception:
        return None


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
            perturbed_df[feature_name] = pd.to_numeric(
                perturbed_df[feature_name],
                errors="coerce",
            ).astype(float)
        perturbed_df.at[perturbed_df.index[0], feature_name] = get_reference_value(
            feature_name,
            current_value,
        )

        _, perturbed_probabilities = predict_probabilities_from_df(perturbed_df)
        perturbed_probability = float(perturbed_probabilities[target_index])
        impacts.append(base_probability - perturbed_probability)

    return [float(value) for value in impacts]


def get_feature_impacts(input_scaled, input_df, target_index):
    explainer = get_shap_explainer()
    if explainer is not None:
        try:
            shap_values = explainer.shap_values(input_scaled, check_additivity=False)

            try:
                values = shap_values[target_index][0]
            except Exception:
                values = shap_values[:, :, target_index][0]

            return [float(value) for value in values], "shap"
        except Exception:
            pass

    return build_fallback_impacts(input_df, target_index), "fallback"


def format_feature_value(feature_name, feature_value):
    if feature_name in CATEGORICAL_FEATURE_COLUMNS:
        return str(feature_value)
    if feature_name in ["N", "P", "K"]:
        return str(int(round(float(feature_value))))
    return "{0:.2f}".format(float(feature_value))


def get_effect_label(value):
    if value > EPSILON:
        return "Positive (Helps crop)"
    if value < -EPSILON:
        return "Negative (Harms crop)"
    return "Neutral"


def build_human_explanation(
    feature_key,
    feature_label,
    feature_value,
    prediction_name,
    impact_value,
):
    display_value = format_feature_value(feature_key, feature_value)

    if impact_value > EPSILON:
        return (
            "{0} positively influenced the recommendation of {1}; "
            "the observed value ({2}) is favorable for this crop.".format(
                feature_label,
                prediction_name,
                display_value,
            )
        )
    if impact_value < -EPSILON:
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


def build_shap_output(prediction_name, top3_predictions, impact_values, input_df):
    observed_values = [input_df.iloc[0][feature_name] for feature_name in FEATURE_COLUMNS]
    shap_df = pd.DataFrame(
        {
            "Feature": FEATURE_NAMES,
            "Impact Value": impact_values,
            "Observed Value": [
                format_feature_value(feature_name, value)
                for feature_name, value in zip(FEATURE_COLUMNS, observed_values)
            ],
        }
    )

    shap_df["Effect"] = shap_df["Impact Value"].apply(get_effect_label)
    shap_df["Human Explanation"] = [
        build_human_explanation(
            feature_key,
            feature_label,
            feature_value,
            prediction_name,
            impact_value,
        )
        for feature_key, feature_label, feature_value, impact_value in zip(
            FEATURE_COLUMNS,
            FEATURE_NAMES,
            observed_values,
            impact_values,
        )
    ]
    shap_df = shap_df.reindex(
        shap_df["Impact Value"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)

    explanations = shap_df["Human Explanation"].tolist()
    merged_df = shap_df[["Feature", "Impact Value", "Effect"]].copy()
    return merged_df, shap_df, explanations, prediction_name, top3_predictions


def build_reference_background():
    rows = []

    for index in range(LIME_SYNTHETIC_ROWS):
        row = {}
        for feature_name in FEATURE_COLUMNS:
            reference = FEATURE_REFERENCE.get(feature_name, {})

            if feature_name == "season":
                row[feature_name] = SEASON_OPTIONS[index % len(SEASON_OPTIONS)]
            elif feature_name == "water_source":
                row[feature_name] = WATER_SOURCE_OPTIONS[index % len(WATER_SOURCE_OPTIONS)]
            elif feature_name == "water_availability":
                row[feature_name] = WATER_AVAILABILITY_OPTIONS[
                    index % len(WATER_AVAILABILITY_OPTIONS)
                ]
            else:
                mean_value = float(reference.get("mean", 0.0))
                std_value = float(reference.get("std", 1.0))
                spread = max(std_value, 1.0)
                offset = ((index % 5) - 2) * 0.25 * spread
                row[feature_name] = round(mean_value + offset, 4)

        rows.append(row)

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


@lru_cache(maxsize=1)
def get_lime_background_df():
    dataset_path_value = metadata.get("dataset_path")
    if dataset_path_value:
        dataset_path = Path(dataset_path_value)
        if dataset_path.exists():
            try:
                dataset = pd.read_csv(dataset_path)
                if set(FEATURE_COLUMNS).issubset(dataset.columns):
                    background_df = dataset[FEATURE_COLUMNS].dropna().copy()
                    if len(background_df) > LIME_BACKGROUND_ROWS:
                        background_df = background_df.sample(
                            n=LIME_BACKGROUND_ROWS,
                            random_state=42,
                        )
                    return background_df.reset_index(drop=True)
            except Exception:
                pass

    return build_reference_background()


@lru_cache(maxsize=1)
def get_lime_background_transformed():
    return transform_input(get_lime_background_df())


@lru_cache(maxsize=1)
def get_lime_explainer():
    if LimeTabularExplainer is None:
        return None

    try:
        background = get_lime_background_transformed()
        return LimeTabularExplainer(
            training_data=np.asarray(background, dtype=float),
            feature_names=FEATURE_NAMES,
            class_names=[str(name) for name in label_encoder.classes_],
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )
    except Exception:
        return None


def build_lime_output_from_pairs(
    prediction_name,
    input_df,
    weighted_pairs,
    method_name,
):
    observed_values = {feature_name: input_df.iloc[0][feature_name] for feature_name in FEATURE_COLUMNS}
    rows = []

    for feature_index, weight in weighted_pairs:
        feature_key = FEATURE_COLUMNS[feature_index]
        feature_label = FEATURE_NAMES[feature_index]
        feature_value = observed_values[feature_key]
        rows.append(
            {
                "Feature": feature_label,
                "Local Weight": float(weight),
                "Observed Value": format_feature_value(feature_key, feature_value),
                "Effect": get_effect_label(weight),
                "Human Explanation": build_human_explanation(
                    feature_key,
                    feature_label,
                    feature_value,
                    prediction_name,
                    weight,
                ),
            }
        )

    lime_df = pd.DataFrame(rows)
    if lime_df.empty:
        lime_df = pd.DataFrame(
            columns=[
                "Feature",
                "Local Weight",
                "Observed Value",
                "Effect",
                "Human Explanation",
            ]
        )
    else:
        lime_df = lime_df.reindex(
            lime_df["Local Weight"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)

    lime_lines = lime_df["Human Explanation"].tolist()
    lime_table = lime_df[["Feature", "Local Weight", "Effect"]].copy()
    return lime_table, lime_df, lime_lines, method_name


def build_lime_fallback(input_df, target_index, prediction_name):
    fallback_impacts = build_fallback_impacts(input_df, target_index)
    ranked_pairs = sorted(
        enumerate(fallback_impacts),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:MAX_LIME_FEATURES]
    return build_lime_output_from_pairs(
        prediction_name,
        input_df,
        ranked_pairs,
        "fallback",
    )


def get_lime_explanation(input_scaled, input_df, target_index, prediction_name):
    explainer = get_lime_explainer()
    if explainer is not None:
        try:
            explanation = explainer.explain_instance(
                np.asarray(input_scaled[0], dtype=float),
                model.predict_proba,
                labels=[int(target_index)],
                num_features=min(MAX_LIME_FEATURES, len(FEATURE_COLUMNS)),
            )
            explanation_map = explanation.as_map().get(int(target_index), [])
            if explanation_map:
                return build_lime_output_from_pairs(
                    prediction_name,
                    input_df,
                    explanation_map,
                    "lime",
                )
        except Exception:
            pass

    return build_lime_fallback(input_df, target_index, prediction_name)


def build_consensus_summary(prediction_name, shap_df, lime_df):
    if shap_df.empty and lime_df.empty:
        return (
            "The recommendation of {0} is available, but no explanation details could be generated."
        ).format(prediction_name)

    shap_signs = {}
    if not shap_df.empty:
        for _, row in shap_df.head(5).iterrows():
            shap_signs[row["Feature"]] = np.sign(float(row["Impact Value"]))

    lime_signs = {}
    if not lime_df.empty:
        for _, row in lime_df.head(5).iterrows():
            lime_signs[row["Feature"]] = np.sign(float(row["Local Weight"]))

    shared_features = [feature for feature in shap_signs if feature in lime_signs]
    agreed_positive = [
        feature for feature in shared_features if shap_signs[feature] > 0 and lime_signs[feature] > 0
    ]
    agreed_negative = [
        feature for feature in shared_features if shap_signs[feature] < 0 and lime_signs[feature] < 0
    ]
    disagreed = [
        feature for feature in shared_features if shap_signs[feature] != 0 and lime_signs[feature] != 0 and shap_signs[feature] != lime_signs[feature]
    ]

    summary_parts = [
        "SHAP gives the broader model view, while LIME explains this exact farm input for {0}.".format(
            prediction_name
        )
    ]

    if agreed_positive:
        summary_parts.append(
            "Both methods highlight {0} as supportive factors.".format(
                ", ".join(agreed_positive[:3])
            )
        )
    if agreed_negative:
        summary_parts.append(
            "Both methods flag {0} as limiting factors.".format(
                ", ".join(agreed_negative[:3])
            )
        )
    if disagreed:
        summary_parts.append(
            "They disagree on {0}, so those factors should be interpreted more carefully.".format(
                ", ".join(disagreed[:2])
            )
        )

    if len(summary_parts) == 1:
        summary_parts.append(
            "Reviewing both explanations still helps cross-check the recommendation from two different angles."
        )

    return " ".join(summary_parts)


def build_explanation_bundle(prediction_name, top3_predictions, input_scaled, input_df, target_index):
    impact_values, shap_method = get_feature_impacts(input_scaled, input_df, target_index)
    shap_table, shap_detail_table, shap_lines, _, _ = build_shap_output(
        prediction_name,
        top3_predictions,
        impact_values,
        input_df,
    )
    lime_table, lime_detail_table, lime_lines, lime_method = get_lime_explanation(
        input_scaled,
        input_df,
        target_index,
        prediction_name,
    )

    return {
        "prediction": prediction_name,
        "top3_predictions": top3_predictions,
        "shap_table": shap_table,
        "shap_detail_table": shap_detail_table,
        "shap_lines": shap_lines,
        "lime_table": lime_table,
        "lime_detail_table": lime_detail_table,
        "lime_lines": lime_lines,
        "consensus_summary": build_consensus_summary(
            prediction_name,
            shap_detail_table,
            lime_detail_table,
        ),
        "shap_method": shap_method,
        "lime_method": lime_method,
    }


def explain_crop_prediction_bundle(input_df):
    input_scaled, probabilities = predict_probabilities_from_df(input_df)
    best_index, best_prediction, top3_predictions = get_top_predictions(probabilities)
    return build_explanation_bundle(
        best_prediction,
        top3_predictions,
        input_scaled,
        input_df,
        best_index,
    )


def explain_specific_crop_prediction_bundle(input_df, crop_name, top3_predictions=None):
    input_scaled, probabilities = predict_probabilities_from_df(input_df)
    if top3_predictions is None:
        _, _, top3_predictions = get_top_predictions(probabilities)

    target_index = get_class_index(crop_name)
    return build_explanation_bundle(
        crop_name,
        top3_predictions,
        input_scaled,
        input_df,
        target_index,
    )


def explain_crop_prediction(input_df):
    bundle = explain_crop_prediction_bundle(input_df)
    return (
        bundle["shap_table"],
        bundle["shap_detail_table"],
        bundle["shap_lines"],
        bundle["prediction"],
        bundle["top3_predictions"],
    )


def explain_specific_crop_prediction(input_df, crop_name, top3_predictions=None):
    bundle = explain_specific_crop_prediction_bundle(
        input_df,
        crop_name,
        top3_predictions=top3_predictions,
    )
    return (
        bundle["shap_table"],
        bundle["shap_detail_table"],
        bundle["shap_lines"],
        bundle["prediction"],
        bundle["top3_predictions"],
    )
