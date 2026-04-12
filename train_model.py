import argparse
import json
from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

from utils.crop_features import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_DISPLAY_NAMES,
    NEW_MAHARASHTRA_CROPS,
    NUMERIC_FEATURE_COLUMNS,
    REGION_COLUMN,
    REGION_VALUE,
    TARGET_COLUMN,
    build_feature_reference,
    encode_feature_frame,
    ensure_context_columns,
)
from utils.maharashtra_dataset import ROWS_PER_CROP_DEFAULT, save_balanced_maharashtra_dataset


RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Data" / "crop_recommendation.csv"
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "model.pkl"
LEGACY_MODEL_PATH = MODELS_DIR / "xgboost_crop_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
METADATA_PATH = MODELS_DIR / "crop_model_metadata.json"
EVALUATION_JSON_PATH = MODELS_DIR / "crop_model_evaluation.json"
COMPARISON_CSV_PATH = MODELS_DIR / "crop_model_comparison.csv"
CONFUSION_MATRIX_CSV_PATH = MODELS_DIR / "crop_confusion_matrix.csv"
CLASSIFICATION_REPORT_PATH = MODELS_DIR / "crop_classification_report.txt"
DATA_QUALITY_REPORT_PATH = MODELS_DIR / "crop_data_quality_report.json"
CLASS_DISTRIBUTION_PATH = MODELS_DIR / "crop_class_distribution.csv"

COLUMN_ALIASES = {
    "temprature": "temperature",
    "crop": TARGET_COLUMN,
    "crop_name": TARGET_COLUMN,
    "ph_level": "ph",
}


def ensure_output_dir():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def standardize_dataset_columns(dataset):
    renamed_columns = {}
    for column in dataset.columns:
        normalized = str(column).strip()
        renamed_columns[column] = COLUMN_ALIASES.get(normalized, normalized)
    return dataset.rename(columns=renamed_columns)


def load_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path)
    dataset = standardize_dataset_columns(dataset)

    if TARGET_COLUMN not in dataset.columns:
        raise ValueError("Dataset must contain a '{0}' column.".format(TARGET_COLUMN))

    for context_column in ["season", "water_source", "water_availability"]:
        if context_column in dataset.columns:
            dataset[context_column] = dataset[context_column].fillna("")

    dataset = ensure_context_columns(dataset)

    required_columns = NUMERIC_FEATURE_COLUMNS + [TARGET_COLUMN, REGION_COLUMN] + CATEGORICAL_FEATURE_COLUMNS
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        raise ValueError("Dataset is missing required columns: {0}".format(", ".join(missing_columns)))

    for column in NUMERIC_FEATURE_COLUMNS:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")

    return dataset


def fill_missing_values(dataset):
    filled_dataset = dataset.copy()

    for column in NUMERIC_FEATURE_COLUMNS:
        filled_dataset[column] = filled_dataset[column].fillna(filled_dataset[column].median())

    for column in CATEGORICAL_FEATURE_COLUMNS:
        mode_value = filled_dataset[column].mode(dropna=True)
        fallback_value = mode_value.iloc[0] if not mode_value.empty else ""
        filled_dataset[column] = filled_dataset[column].replace("", pd.NA).fillna(fallback_value)

    filled_dataset[REGION_COLUMN] = filled_dataset[REGION_COLUMN].replace("", pd.NA).fillna(REGION_VALUE)
    filled_dataset[TARGET_COLUMN] = filled_dataset[TARGET_COLUMN].astype(str).str.strip().str.lower()
    return ensure_context_columns(filled_dataset)


def build_outlier_summary(dataset):
    summary = OrderedDict()
    for column in NUMERIC_FEATURE_COLUMNS:
        series = dataset[column].astype(float)
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())

        summary[column] = {
            "q1": round(q1, 6),
            "q3": round(q3, 6),
            "iqr": round(iqr, 6),
            "lower_bound": round(lower_bound, 6),
            "upper_bound": round(upper_bound, 6),
            "outlier_count": outlier_count,
            "outlier_percent": round((outlier_count / len(series)) * 100, 4),
        }

    return summary


def audit_and_clean_dataset(dataset, dataset_path):
    raw_row_count = int(len(dataset))
    duplicate_rows = int(dataset.duplicated().sum())
    missing_values_by_column = {column: int(value) for column, value in dataset.isna().sum().items()}

    cleaned_dataset = fill_missing_values(dataset)
    cleaned_dataset = cleaned_dataset.drop_duplicates().reset_index(drop=True)

    class_counts = cleaned_dataset[TARGET_COLUMN].value_counts().sort_index()
    class_distribution_df = (
        class_counts.rename_axis(TARGET_COLUMN)
        .reset_index(name="count")
        .sort_values(TARGET_COLUMN)
        .reset_index(drop=True)
    )
    class_distribution_df["percent"] = (class_distribution_df["count"] / len(cleaned_dataset) * 100).round(4)

    minority_count = int(class_counts.min())
    majority_count = int(class_counts.max())
    imbalance_ratio = round(float(majority_count / minority_count), 6) if minority_count else None

    quality_report = OrderedDict()
    quality_report["dataset_path"] = str(dataset_path)
    quality_report["region"] = REGION_VALUE
    quality_report["raw_row_count"] = raw_row_count
    quality_report["cleaned_row_count"] = int(len(cleaned_dataset))
    quality_report["rows_removed"] = int(raw_row_count - len(cleaned_dataset))
    quality_report["duplicate_rows_removed"] = duplicate_rows
    quality_report["missing_values_by_column_before_fill"] = missing_values_by_column
    quality_report["class_balance"] = {
        "num_classes": int(len(class_counts)),
        "minority_class_count": minority_count,
        "majority_class_count": majority_count,
        "imbalance_ratio": imbalance_ratio,
        "is_balanced_within_5_percent": bool(imbalance_ratio is not None and imbalance_ratio <= 1.05),
    }
    quality_report["feature_ranges"] = {
        column: {
            "min": round(float(cleaned_dataset[column].min()), 6),
            "max": round(float(cleaned_dataset[column].max()), 6),
            "mean": round(float(cleaned_dataset[column].mean()), 6),
            "std": round(float(cleaned_dataset[column].std(ddof=0)), 6),
        }
        for column in NUMERIC_FEATURE_COLUMNS
    }
    quality_report["context_distribution"] = {
        REGION_COLUMN: cleaned_dataset[REGION_COLUMN].value_counts().sort_index().to_dict(),
        "season": cleaned_dataset["season"].value_counts().sort_index().to_dict(),
        "water_source": cleaned_dataset["water_source"].value_counts().sort_index().to_dict(),
        "water_availability": cleaned_dataset["water_availability"].value_counts().sort_index().to_dict(),
    }
    quality_report["outlier_summary_iqr"] = build_outlier_summary(cleaned_dataset)
    quality_report["new_crops_added"] = [crop for crop in NEW_MAHARASHTRA_CROPS if crop in set(class_counts.index)]

    return cleaned_dataset, class_distribution_df, quality_report


def build_splits(X, y_encoded):
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, X_train_full, y_train_full


def fit_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(encode_feature_frame(X_train))
    return scaler


def prepare_features(X_frame, scaler):
    encoded_df = encode_feature_frame(X_frame)
    return scaler.transform(encoded_df)


def top_k_accuracy(model, X_eval, y_true, k):
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(X_eval)
    effective_k = min(k, probabilities.shape[1])
    top_k_indices = np.argsort(probabilities, axis=1)[:, -effective_k:]
    hits = [(target in row) for target, row in zip(y_true, top_k_indices)]
    return float(np.mean(hits))


def evaluate_model(model, X_eval, y_true, class_names):
    predictions = model.predict(X_eval)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )

    metrics = OrderedDict()
    metrics["accuracy"] = round(float(accuracy_score(y_true, predictions)), 6)
    metrics["precision_weighted"] = round(float(precision), 6)
    metrics["recall_weighted"] = round(float(recall), 6)
    metrics["f1_weighted"] = round(float(f1_score), 6)
    top_3 = top_k_accuracy(model, X_eval, y_true, 3)
    metrics["top_3_accuracy"] = None if top_3 is None else round(float(top_3), 6)
    metrics["classification_report"] = classification_report(
        y_true,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics["classification_report_text"] = classification_report(
        y_true,
        predictions,
        target_names=class_names,
        zero_division=0,
    )
    metrics["confusion_matrix"] = confusion_matrix(y_true, predictions).tolist()
    return metrics


def build_candidates(num_classes):
    candidates = [
        {
            "name": "random_forest_balanced_v1",
            "model_type": "RandomForestClassifier",
            "params": {
                "n_estimators": 300,
                "max_depth": 18,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "class_weight": "balanced_subsample",
                "random_state": RANDOM_STATE,
                "n_jobs": 1,
            },
        },
        {
            "name": "random_forest_balanced_v2",
            "model_type": "RandomForestClassifier",
            "params": {
                "n_estimators": 450,
                "max_depth": 24,
                "min_samples_split": 4,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "class_weight": "balanced_subsample",
                "random_state": RANDOM_STATE,
                "n_jobs": 1,
            },
        },
    ]

    if XGBClassifier is not None:
        candidates.extend(
            [
                {
                    "name": "xgboost_multiclass_v1",
                    "model_type": "XGBClassifier",
                    "params": {
                        "objective": "multi:softprob",
                        "num_class": num_classes,
                        "n_estimators": 500,
                        "learning_rate": 0.05,
                        "max_depth": 5,
                        "min_child_weight": 1,
                        "subsample": 0.9,
                        "colsample_bytree": 0.85,
                        "reg_lambda": 1.0,
                        "eval_metric": "mlogloss",
                        "tree_method": "hist",
                        "random_state": RANDOM_STATE,
                        "n_jobs": 1,
                        "early_stopping_rounds": 30,
                    },
                },
                {
                    "name": "xgboost_multiclass_v2",
                    "model_type": "XGBClassifier",
                    "params": {
                        "objective": "multi:softprob",
                        "num_class": num_classes,
                        "n_estimators": 650,
                        "learning_rate": 0.03,
                        "max_depth": 7,
                        "min_child_weight": 2,
                        "subsample": 0.85,
                        "colsample_bytree": 0.8,
                        "reg_lambda": 1.2,
                        "eval_metric": "mlogloss",
                        "tree_method": "hist",
                        "random_state": RANDOM_STATE,
                        "n_jobs": 1,
                        "early_stopping_rounds": 30,
                    },
                },
            ]
        )

    return candidates


def instantiate_candidate(candidate):
    if candidate["model_type"] == "RandomForestClassifier":
        return RandomForestClassifier(**candidate["params"])
    if candidate["model_type"] == "XGBClassifier":
        return XGBClassifier(**candidate["params"])
    raise ValueError("Unsupported model type: {0}".format(candidate["model_type"]))


def fit_candidate(candidate, X_train_scaled, y_train, X_val_scaled, y_val, class_names):
    model = instantiate_candidate(candidate)

    if candidate["model_type"] == "XGBClassifier":
        try:
            model.fit(
                X_train_scaled,
                y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False,
            )
        except TypeError:
            fit_params = candidate["params"].copy()
            fit_params.pop("early_stopping_rounds", None)
            model = XGBClassifier(**fit_params)
            model.fit(
                X_train_scaled,
                y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False,
            )
    else:
        model.fit(X_train_scaled, y_train)

    validation_metrics = evaluate_model(model, X_val_scaled, y_val, class_names)

    summary = OrderedDict()
    summary["model_name"] = candidate["name"]
    summary["model_type"] = candidate["model_type"]
    summary["validation_accuracy"] = validation_metrics["accuracy"]
    summary["validation_precision_weighted"] = validation_metrics["precision_weighted"]
    summary["validation_recall_weighted"] = validation_metrics["recall_weighted"]
    summary["validation_f1_weighted"] = validation_metrics["f1_weighted"]
    summary["validation_top_3_accuracy"] = validation_metrics["top_3_accuracy"]
    summary["params"] = candidate["params"]
    return model, validation_metrics, summary


def choose_best_candidate(leaderboard):
    return sorted(
        leaderboard,
        key=lambda row: (
            row["validation_f1_weighted"],
            -1 if row["validation_top_3_accuracy"] is None else row["validation_top_3_accuracy"],
            row["validation_accuracy"],
        ),
        reverse=True,
    )[0]


def fit_final_model(candidate, selected_model, X_train_full_scaled, y_train_full):
    if candidate["model_type"] == "XGBClassifier":
        final_params = candidate["params"].copy()
        final_params.pop("early_stopping_rounds", None)
        best_iteration = getattr(selected_model, "best_iteration", None)
        if best_iteration is not None and best_iteration >= 0:
            final_params["n_estimators"] = int(best_iteration) + 1
        final_model = XGBClassifier(**final_params)
        final_model.fit(X_train_full_scaled, y_train_full, verbose=False)
        return final_model

    final_model = RandomForestClassifier(**candidate["params"])
    final_model.fit(X_train_full_scaled, y_train_full)
    return final_model


def save_pickle(path_obj, value):
    joblib.dump(value, str(path_obj), compress=3)


def save_json(path_obj, payload):
    with open(str(path_obj), "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


def save_text(path_obj, text):
    with open(str(path_obj), "w", encoding="utf-8") as file_obj:
        file_obj.write(text)


def train_and_save(dataset_path=DATASET_PATH):
    ensure_output_dir()
    dataset = load_dataset(dataset_path)
    cleaned_dataset, class_distribution_df, quality_report = audit_and_clean_dataset(dataset, dataset_path)

    X = cleaned_dataset[FEATURE_COLUMNS].copy()
    y = cleaned_dataset[TARGET_COLUMN].copy()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_.tolist()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        X_train_full,
        y_train_full,
    ) = build_splits(X, y_encoded)

    scaler = fit_scaler(X_train)
    X_train_scaled = prepare_features(X_train, scaler)
    X_val_scaled = prepare_features(X_val, scaler)

    print("Dataset loaded from:", dataset_path)
    print("Dataset shape:", dataset.shape)
    print("Cleaned dataset shape:", cleaned_dataset.shape)
    print("Classes:", len(class_names))
    print("Train/Test rows:", len(X_train_full), len(X_test))
    print("Internal validation rows:", len(X_val))
    print("Imbalance ratio:", quality_report["class_balance"]["imbalance_ratio"])

    candidate_runs = []
    print("\nModel comparison started...\n")

    for candidate in build_candidates(len(class_names)):
        print("Training:", candidate["name"])
        model, validation_metrics, summary = fit_candidate(
            candidate,
            X_train_scaled,
            y_train,
            X_val_scaled,
            y_val,
            class_names,
        )
        summary["validation_report"] = validation_metrics["classification_report"]
        candidate_runs.append(
            {
                "candidate": candidate,
                "model": model,
                "summary": summary,
            }
        )
        print(
            "  validation accuracy={0:.4f}, f1={1:.4f}, top3={2}".format(
                summary["validation_accuracy"],
                summary["validation_f1_weighted"],
                summary["validation_top_3_accuracy"],
            )
        )

    leaderboard = [run["summary"] for run in candidate_runs]
    best_summary = choose_best_candidate(leaderboard)
    selected_run = next(
        run for run in candidate_runs if run["summary"]["model_name"] == best_summary["model_name"]
    )

    final_scaler = fit_scaler(X_train_full)
    X_train_full_scaled = prepare_features(X_train_full, final_scaler)
    X_test_scaled = prepare_features(X_test, final_scaler)
    final_model = fit_final_model(
        selected_run["candidate"],
        selected_run["model"],
        X_train_full_scaled,
        y_train_full,
    )
    final_test_metrics = evaluate_model(final_model, X_test_scaled, y_test, class_names)

    comparison_df = pd.DataFrame(leaderboard).sort_values(
        by=["validation_f1_weighted", "validation_accuracy"],
        ascending=False,
    ).reset_index(drop=True)
    confusion_df = pd.DataFrame(final_test_metrics["confusion_matrix"], index=class_names, columns=class_names)
    feature_reference = build_feature_reference(X_train_full)

    metadata = OrderedDict()
    metadata["dataset_path"] = str(dataset_path)
    metadata["region"] = REGION_VALUE
    metadata["feature_columns"] = FEATURE_COLUMNS
    metadata["feature_display_names"] = FEATURE_DISPLAY_NAMES
    metadata["target_column"] = TARGET_COLUMN
    metadata["class_names"] = class_names
    metadata["selected_model_name"] = selected_run["candidate"]["name"]
    metadata["selected_model_type"] = type(final_model).__name__
    metadata["uses_scaler"] = True
    metadata["feature_reference"] = feature_reference
    metadata["rows_per_crop_target"] = ROWS_PER_CROP_DEFAULT
    metadata["split_summary"] = {
        "train_rows": int(len(X_train_full)),
        "test_rows": int(len(X_test)),
        "validation_rows_for_tuning": int(len(X_val)),
        "train_test_split": "80_20_stratified",
    }

    evaluation_payload = OrderedDict()
    evaluation_payload["selected_model"] = metadata["selected_model_name"]
    evaluation_payload["selected_model_type"] = metadata["selected_model_type"]
    evaluation_payload["selection_metric"] = "validation_f1_weighted_then_validation_accuracy"
    evaluation_payload["candidate_leaderboard"] = leaderboard
    evaluation_payload["test_metrics"] = OrderedDict(
        [
            ("accuracy", final_test_metrics["accuracy"]),
            ("precision_weighted", final_test_metrics["precision_weighted"]),
            ("recall_weighted", final_test_metrics["recall_weighted"]),
            ("f1_weighted", final_test_metrics["f1_weighted"]),
            ("top_3_accuracy", final_test_metrics["top_3_accuracy"]),
        ]
    )
    evaluation_payload["classification_report"] = final_test_metrics["classification_report"]

    save_pickle(MODEL_PATH, final_model)
    save_pickle(LEGACY_MODEL_PATH, final_model)
    save_pickle(SCALER_PATH, final_scaler)
    save_pickle(LABEL_ENCODER_PATH, label_encoder)
    save_json(METADATA_PATH, metadata)
    save_json(EVALUATION_JSON_PATH, evaluation_payload)
    save_json(DATA_QUALITY_REPORT_PATH, quality_report)
    class_distribution_df.to_csv(str(CLASS_DISTRIBUTION_PATH), index=False)
    comparison_df.to_csv(str(COMPARISON_CSV_PATH), index=False)
    confusion_df.to_csv(str(CONFUSION_MATRIX_CSV_PATH))
    save_text(CLASSIFICATION_REPORT_PATH, final_test_metrics["classification_report_text"])

    print("\nSelected model:", metadata["selected_model_name"])
    print("Selected model type:", metadata["selected_model_type"])
    print("Test accuracy:", final_test_metrics["accuracy"])
    print("Test weighted F1:", final_test_metrics["f1_weighted"])
    print("Test top-3 accuracy:", final_test_metrics["top_3_accuracy"])
    print("\nArtifacts saved:")
    print("-", MODEL_PATH)
    print("-", LEGACY_MODEL_PATH)
    print("-", SCALER_PATH)
    print("-", LABEL_ENCODER_PATH)
    print("-", METADATA_PATH)
    print("-", EVALUATION_JSON_PATH)
    print("-", DATA_QUALITY_REPORT_PATH)
    print("-", CLASS_DISTRIBUTION_PATH)
    print("-", COMPARISON_CSV_PATH)
    print("-", CONFUSION_MATRIX_CSV_PATH)
    print("-", CLASSIFICATION_REPORT_PATH)

    return {
        "selected_model": metadata["selected_model_name"],
        "selected_model_type": metadata["selected_model_type"],
        "test_metrics": evaluation_payload["test_metrics"],
        "dataset_shape": cleaned_dataset.shape,
        "model_path": str(MODEL_PATH),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Harvestify Maharashtra crop recommendation model.")
    parser.add_argument("--dataset-path", default=str(DATASET_PATH), help="Path to the training CSV.")
    parser.add_argument(
        "--refresh-dataset",
        action="store_true",
        help="Regenerate the merged Maharashtra dataset before training.",
    )
    parser.add_argument(
        "--rows-per-crop",
        type=int,
        default=ROWS_PER_CROP_DEFAULT,
        help="Balanced rows per crop when refreshing the dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help="Random seed used for dataset refresh.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path).resolve()

    if args.refresh_dataset:
        save_balanced_maharashtra_dataset(
            output_path=dataset_path,
            rows_per_crop=args.rows_per_crop,
            seed=args.seed,
            existing_dataset_path=dataset_path,
        )

    results = train_and_save(dataset_path=dataset_path)
    print("\nTraining complete:", results)


if __name__ == "__main__":
    main()
