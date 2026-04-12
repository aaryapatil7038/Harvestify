from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from utils.crop_features import (
    CANONICAL_CROP_LABELS,
    CROP_CONTEXT_PROFILES,
    EXISTING_MAHARASHTRA_CROPS,
    NEW_MAHARASHTRA_CROPS,
    REGION_COLUMN,
    REGION_VALUE,
    TARGET_COLUMN,
    ensure_context_columns,
)


DATASET_OUTPUT_COLUMNS = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
    REGION_COLUMN,
    "season",
    "water_source",
    "water_availability",
    TARGET_COLUMN,
]

ROWS_PER_CROP_DEFAULT = 600
DATASET_RANDOM_SEED = 42

CROP_DATASET_PROFILES = {
    "sugarcane": {
        "base": {"N": (150, 240), "P": (45, 80), "K": (90, 150), "ph": (6.0, 7.8)},
        "season_climate": {
            "Summer": {"temperature": (26, 36), "humidity": (55, 78), "rainfall": (450, 900)},
            "Rabi": {"temperature": (22, 31), "humidity": (58, 82), "rainfall": (500, 950)},
        },
    },
    "rice": {
        "base": {"N": (80, 125), "P": (35, 65), "K": (35, 65), "ph": (5.0, 6.8)},
        "season_climate": {
            "Kharif": {"temperature": (24, 31), "humidity": (78, 92), "rainfall": (850, 1600)},
            "Summer": {"temperature": (26, 34), "humidity": (60, 82), "rainfall": (450, 800)},
        },
    },
    "soybean": {
        "base": {"N": (20, 60), "P": (35, 70), "K": (30, 60), "ph": (6.0, 7.5)},
        "season_climate": {
            "Kharif": {"temperature": (23, 30), "humidity": (65, 85), "rainfall": (650, 1100)},
        },
    },
    "cotton": {
        "base": {"N": (60, 120), "P": (30, 65), "K": (30, 65), "ph": (6.0, 8.0)},
        "season_climate": {
            "Kharif": {"temperature": (25, 34), "humidity": (50, 75), "rainfall": (500, 900)},
        },
    },
    "jowar": {
        "base": {"N": (40, 80), "P": (20, 45), "K": (20, 45), "ph": (6.2, 8.2)},
        "season_climate": {
            "Kharif": {"temperature": (24, 32), "humidity": (50, 72), "rainfall": (400, 750)},
            "Rabi": {"temperature": (18, 28), "humidity": (38, 60), "rainfall": (120, 320)},
        },
    },
    "bajra": {
        "base": {"N": (30, 70), "P": (15, 35), "K": (15, 35), "ph": (6.0, 8.5)},
        "season_climate": {
            "Kharif": {"temperature": (25, 34), "humidity": (40, 65), "rainfall": (300, 650)},
            "Summer": {"temperature": (28, 37), "humidity": (28, 50), "rainfall": (80, 220)},
        },
    },
    "wheat": {
        "base": {"N": (70, 120), "P": (30, 55), "K": (25, 50), "ph": (6.2, 7.8)},
        "season_climate": {
            "Rabi": {"temperature": (16, 26), "humidity": (45, 68), "rainfall": (120, 320)},
        },
    },
    "maize": {
        "base": {"N": (60, 110), "P": (25, 55), "K": (25, 55), "ph": (5.8, 7.6)},
        "season_climate": {
            "Kharif": {"temperature": (22, 31), "humidity": (55, 78), "rainfall": (450, 900)},
            "Rabi": {"temperature": (18, 27), "humidity": (40, 64), "rainfall": (120, 320)},
        },
    },
    "groundnut": {
        "base": {"N": (20, 50), "P": (30, 60), "K": (30, 60), "ph": (5.8, 7.4)},
        "season_climate": {
            "Kharif": {"temperature": (24, 31), "humidity": (52, 74), "rainfall": (350, 700)},
            "Summer": {"temperature": (27, 35), "humidity": (35, 58), "rainfall": (100, 240)},
        },
    },
    "tur": {
        "base": {"N": (18, 45), "P": (25, 55), "K": (20, 45), "ph": (6.0, 8.0)},
        "season_climate": {
            "Kharif": {"temperature": (23, 31), "humidity": (48, 72), "rainfall": (450, 850)},
        },
    },
    "gram": {
        "base": {"N": (15, 40), "P": (20, 50), "K": (20, 45), "ph": (6.0, 8.2)},
        "season_climate": {
            "Rabi": {"temperature": (17, 26), "humidity": (35, 58), "rainfall": (80, 220)},
        },
    },
    "onion": {
        "base": {"N": (65, 120), "P": (25, 55), "K": (35, 70), "ph": (6.0, 7.6)},
        "season_climate": {
            "Rabi": {"temperature": (18, 28), "humidity": (48, 68), "rainfall": (70, 200)},
            "Summer": {"temperature": (24, 33), "humidity": (35, 58), "rainfall": (30, 120)},
        },
    },
    "tomato": {
        "base": {"N": (70, 130), "P": (30, 60), "K": (35, 75), "ph": (6.0, 7.5)},
        "season_climate": {
            "Rabi": {"temperature": (18, 28), "humidity": (48, 72), "rainfall": (80, 220)},
            "Summer": {"temperature": (24, 33), "humidity": (35, 58), "rainfall": (30, 120)},
        },
    },
    "chili": {
        "base": {"N": (65, 120), "P": (30, 60), "K": (35, 75), "ph": (6.0, 7.5)},
        "season_climate": {
            "Kharif": {"temperature": (22, 31), "humidity": (55, 78), "rainfall": (300, 700)},
            "Rabi": {"temperature": (18, 28), "humidity": (42, 65), "rainfall": (80, 220)},
        },
    },
    "turmeric": {
        "base": {"N": (90, 150), "P": (35, 70), "K": (80, 130), "ph": (5.8, 7.4)},
        "season_climate": {
            "Kharif": {"temperature": (22, 30), "humidity": (68, 88), "rainfall": (700, 1300)},
        },
    },
    "banana": {
        "base": {"N": (140, 220), "P": (40, 70), "K": (180, 260), "ph": (6.0, 7.5)},
        "season_climate": {
            "Kharif": {"temperature": (24, 33), "humidity": (70, 88), "rainfall": (700, 1400)},
            "Summer": {"temperature": (27, 36), "humidity": (45, 65), "rainfall": (150, 450)},
        },
    },
    "orange": {
        "base": {"N": (60, 120), "P": (25, 55), "K": (50, 90), "ph": (5.8, 7.5)},
        "season_climate": {
            "Kharif": {"temperature": (23, 31), "humidity": (60, 82), "rainfall": (600, 1100)},
            "Rabi": {"temperature": (18, 28), "humidity": (45, 70), "rainfall": (60, 220)},
        },
    },
    "grapes": {
        "base": {"N": (70, 120), "P": (35, 65), "K": (80, 140), "ph": (6.0, 7.8)},
        "season_climate": {
            "Rabi": {"temperature": (17, 28), "humidity": (35, 60), "rainfall": (30, 160)},
            "Summer": {"temperature": (26, 36), "humidity": (25, 48), "rainfall": (10, 100)},
        },
    },
    "pomegranate": {
        "base": {"N": (50, 100), "P": (25, 50), "K": (60, 110), "ph": (6.0, 7.8)},
        "season_climate": {
            "Kharif": {"temperature": (23, 32), "humidity": (50, 75), "rainfall": (350, 750)},
            "Summer": {"temperature": (27, 38), "humidity": (25, 50), "rainfall": (20, 120)},
        },
    },
    "ginger": {
        "base": {"N": (70, 140), "P": (40, 75), "K": (80, 140), "ph": (5.5, 6.8)},
        "season_climate": {
            "Kharif": {"temperature": (22, 30), "humidity": (72, 90), "rainfall": (900, 1600)},
        },
    },
    "potato": {
        "base": {"N": (90, 150), "P": (45, 80), "K": (80, 130), "ph": (5.2, 6.8)},
        "season_climate": {
            "Rabi": {"temperature": (15, 24), "humidity": (55, 75), "rainfall": (60, 220)},
        },
    },
    "brinjal": {
        "base": {"N": (80, 140), "P": (35, 65), "K": (50, 90), "ph": (5.8, 7.2)},
        "season_climate": {
            "Kharif": {"temperature": (23, 31), "humidity": (60, 82), "rainfall": (350, 750)},
            "Rabi": {"temperature": (18, 28), "humidity": (45, 68), "rainfall": (80, 220)},
        },
    },
    "cabbage": {
        "base": {"N": (90, 150), "P": (40, 75), "K": (60, 110), "ph": (6.0, 7.5)},
        "season_climate": {
            "Rabi": {"temperature": (15, 25), "humidity": (55, 78), "rainfall": (60, 200)},
            "Summer": {"temperature": (22, 30), "humidity": (45, 70), "rainfall": (50, 180)},
        },
    },
    "sunflower": {
        "base": {"N": (50, 90), "P": (25, 50), "K": (25, 55), "ph": (6.0, 8.0)},
        "season_climate": {
            "Kharif": {"temperature": (23, 31), "humidity": (45, 68), "rainfall": (300, 650)},
            "Rabi": {"temperature": (18, 27), "humidity": (35, 55), "rainfall": (60, 180)},
        },
    },
    "sesame": {
        "base": {"N": (20, 45), "P": (15, 35), "K": (15, 35), "ph": (5.8, 7.8)},
        "season_climate": {
            "Kharif": {"temperature": (24, 33), "humidity": (40, 65), "rainfall": (250, 550)},
            "Summer": {"temperature": (28, 37), "humidity": (25, 50), "rainfall": (40, 150)},
        },
    },
    "blackgram": {
        "base": {"N": (18, 45), "P": (20, 45), "K": (20, 40), "ph": (6.0, 7.8)},
        "season_climate": {
            "Kharif": {"temperature": (24, 32), "humidity": (55, 78), "rainfall": (350, 750)},
            "Summer": {"temperature": (29, 37), "humidity": (28, 50), "rainfall": (40, 140)},
        },
    },
    "mungbean": {
        "base": {"N": (18, 40), "P": (20, 45), "K": (20, 40), "ph": (6.0, 7.5)},
        "season_climate": {
            "Summer": {"temperature": (29, 38), "humidity": (25, 45), "rainfall": (20, 100)},
            "Kharif": {"temperature": (24, 31), "humidity": (55, 75), "rainfall": (300, 650)},
        },
    },
    "ragi": {
        "base": {"N": (35, 70), "P": (20, 45), "K": (20, 45), "ph": (5.5, 7.5)},
        "season_climate": {
            "Kharif": {"temperature": (20, 28), "humidity": (65, 85), "rainfall": (650, 1200)},
        },
    },
    "cowpea": {
        "base": {"N": (20, 50), "P": (20, 45), "K": (20, 40), "ph": (5.8, 7.5)},
        "season_climate": {
            "Kharif": {"temperature": (24, 32), "humidity": (55, 78), "rainfall": (350, 750)},
            "Summer": {"temperature": (28, 36), "humidity": (30, 52), "rainfall": (50, 160)},
        },
    },
    "lentil": {
        "base": {"N": (15, 35), "P": (20, 45), "K": (20, 40), "ph": (6.0, 7.8)},
        "season_climate": {
            "Rabi": {"temperature": (16, 25), "humidity": (35, 58), "rainfall": (60, 180)},
        },
    },
}


def clipped_normal(rng, bounds, decimals=2, integer=False):
    lower, upper = bounds
    mean = (lower + upper) / 2.0
    std = max((upper - lower) / 6.0, 0.01)
    value = float(np.clip(rng.normal(mean, std), lower, upper))
    if integer:
        return int(round(value))
    return round(value, decimals)


def build_crop_row(rng, crop_name, season):
    profile = CROP_DATASET_PROFILES[crop_name]
    context = CROP_CONTEXT_PROFILES[crop_name]
    climate = profile["season_climate"][season]

    return {
        "N": clipped_normal(rng, profile["base"]["N"], integer=True),
        "P": clipped_normal(rng, profile["base"]["P"], integer=True),
        "K": clipped_normal(rng, profile["base"]["K"], integer=True),
        "temperature": clipped_normal(rng, climate["temperature"]),
        "humidity": clipped_normal(rng, climate["humidity"]),
        "ph": clipped_normal(rng, profile["base"]["ph"]),
        "rainfall": clipped_normal(rng, climate["rainfall"]),
        REGION_COLUMN: REGION_VALUE,
        "season": season,
        "water_source": context["water_source"],
        "water_availability": context["water_availability"],
        TARGET_COLUMN: crop_name,
    }


def build_season_allocation(valid_seasons, rows_per_crop):
    base_count = rows_per_crop // len(valid_seasons)
    remainder = rows_per_crop % len(valid_seasons)
    season_counts = {}
    for index, season in enumerate(valid_seasons):
        season_counts[season] = base_count + (1 if index < remainder else 0)
    return season_counts


def standardize_existing_dataset(existing_dataset):
    dataset = existing_dataset.copy()
    dataset = ensure_context_columns(dataset)
    dataset = dataset[DATASET_OUTPUT_COLUMNS].copy()
    dataset[TARGET_COLUMN] = dataset[TARGET_COLUMN].astype(str).str.strip().str.lower()

    for column in ["N", "P", "K"]:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce").round().astype("Int64")

    for column in ["temperature", "humidity", "ph", "rainfall"]:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce").round(2)

    dataset = dataset.dropna().drop_duplicates().reset_index(drop=True)
    return dataset


def rebalance_existing_crop_rows(existing_dataset, rows_per_crop, seed):
    balanced_frames = []
    for crop_name in EXISTING_MAHARASHTRA_CROPS:
        crop_rows = existing_dataset[existing_dataset[TARGET_COLUMN] == crop_name].copy()
        if crop_rows.empty:
            continue

        if len(crop_rows) >= rows_per_crop:
            crop_rows = crop_rows.sample(n=rows_per_crop, random_state=seed, replace=False)
        else:
            deficit = rows_per_crop - len(crop_rows)
            extras = crop_rows.sample(n=deficit, random_state=seed, replace=True).copy()
            balanced_frames.extend([crop_rows, extras])
            continue

        balanced_frames.append(crop_rows)

    if not balanced_frames:
        return pd.DataFrame(columns=DATASET_OUTPUT_COLUMNS)

    balanced_dataset = pd.concat(balanced_frames, ignore_index=True)
    balanced_dataset = balanced_dataset[DATASET_OUTPUT_COLUMNS].drop_duplicates().reset_index(drop=True)
    return balanced_dataset


def generate_crop_subset(crop_names, rows_per_crop=ROWS_PER_CROP_DEFAULT, seed=DATASET_RANDOM_SEED):
    rng = np.random.default_rng(seed)
    records = []
    signatures = set()

    for crop_name in crop_names:
        season_counts = build_season_allocation(
            CROP_CONTEXT_PROFILES[crop_name]["valid_seasons"],
            rows_per_crop,
        )

        for season, target_count in season_counts.items():
            generated_count = 0
            attempts = 0

            while generated_count < target_count:
                attempts += 1
                if attempts > target_count * 60:
                    raise RuntimeError(
                        "Could not generate enough unique rows for {0} / {1}".format(
                            crop_name,
                            season,
                        )
                    )

                record = build_crop_row(rng, crop_name, season)
                signature = tuple(record[column] for column in DATASET_OUTPUT_COLUMNS)
                if signature in signatures:
                    continue

                signatures.add(signature)
                records.append(record)
                generated_count += 1

    dataset = pd.DataFrame(records, columns=DATASET_OUTPUT_COLUMNS)
    return dataset.sort_values([TARGET_COLUMN, "season"]).reset_index(drop=True)


def merge_existing_and_new_dataset(existing_dataset, rows_per_crop=ROWS_PER_CROP_DEFAULT, seed=DATASET_RANDOM_SEED):
    standardized_existing = standardize_existing_dataset(existing_dataset)
    balanced_existing = rebalance_existing_crop_rows(standardized_existing, rows_per_crop, seed)
    new_crop_dataset = generate_crop_subset(
        NEW_MAHARASHTRA_CROPS,
        rows_per_crop=rows_per_crop,
        seed=seed + 101,
    )

    merged_dataset = pd.concat([balanced_existing, new_crop_dataset], ignore_index=True)
    merged_dataset = merged_dataset[DATASET_OUTPUT_COLUMNS].drop_duplicates().dropna().reset_index(drop=True)
    merged_dataset = merged_dataset.sort_values([TARGET_COLUMN, "season"]).reset_index(drop=True)
    return merged_dataset


def generate_balanced_maharashtra_dataset(rows_per_crop=ROWS_PER_CROP_DEFAULT, seed=DATASET_RANDOM_SEED):
    return generate_crop_subset(CANONICAL_CROP_LABELS, rows_per_crop=rows_per_crop, seed=seed)


def validate_generated_dataset(dataset):
    class_distribution = Counter(dataset[TARGET_COLUMN].tolist())
    validation = {
        "row_count": int(len(dataset)),
        "missing_values": int(dataset.isna().sum().sum()),
        "duplicate_rows": int(dataset.duplicated().sum()),
        "class_distribution": class_distribution,
        "season_distribution": Counter(dataset["season"].tolist()),
        "expected_crops": len(CANONICAL_CROP_LABELS),
        "actual_crops": int(dataset[TARGET_COLUMN].nunique()),
        "new_crops_added": [crop for crop in NEW_MAHARASHTRA_CROPS if crop in class_distribution],
    }
    return validation


def save_balanced_maharashtra_dataset(
    output_path,
    rows_per_crop=ROWS_PER_CROP_DEFAULT,
    seed=DATASET_RANDOM_SEED,
    existing_dataset_path=None,
):
    output_path = Path(output_path)
    if existing_dataset_path is None:
        existing_dataset_path = output_path if output_path.exists() else None
    else:
        existing_dataset_path = Path(existing_dataset_path)

    if existing_dataset_path is not None and Path(existing_dataset_path).exists():
        existing_dataset = pd.read_csv(existing_dataset_path)
        dataset = merge_existing_and_new_dataset(
            existing_dataset,
            rows_per_crop=rows_per_crop,
            seed=seed,
        )
    else:
        dataset = generate_balanced_maharashtra_dataset(rows_per_crop=rows_per_crop, seed=seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return dataset, validate_generated_dataset(dataset)
