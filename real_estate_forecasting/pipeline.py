"""End-to-end demand forecasting pipeline for real estate pricing."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CITY_OPTIONS = ["Metro", "Suburb", "Rural"]


@dataclass
class PipelineConfig:
    n_samples: int = 1200
    test_size: float = 0.2
    random_state: int = 42
    output_dir: Path = Path("artifacts")
    plots_dir: Path = Path("reports")


def generate_descriptions(features: pd.DataFrame) -> list[str]:
    descriptions: list[str] = []
    for _, row in features.iterrows():
        descriptors = []
        if row["sqft"] >= 2500:
            descriptors.append("spacious")
        if row["bedrooms"] >= 4:
            descriptors.append("family-friendly")
        if row["school_rating"] >= 8:
            descriptors.append("top-school")
        if row["crime_rate"] <= 0.25:
            descriptors.append("safe")
        if row["distance_to_center"] <= 10:
            descriptors.append("central")
        if not descriptors:
            descriptors.append("comfortable")

        city_descriptor = row["city"].lower()
        description = (
            f"{', '.join(descriptors)} {city_descriptor} home with "
            f"{int(row['bedrooms'])} beds and {int(row['bathrooms'])} baths"
        )
        descriptions.append(description)
    return descriptions


def generate_synthetic_dataset(n_samples: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    bedrooms = rng.integers(1, 6, size=n_samples)
    bathrooms = rng.integers(1, 5, size=n_samples)
    sqft = rng.integers(600, 4200, size=n_samples)
    year_built = rng.integers(1960, 2024, size=n_samples)
    city = rng.choice(CITY_OPTIONS, size=n_samples, p=[0.5, 0.35, 0.15])
    distance_to_center = rng.uniform(1, 35, size=n_samples).round(1)
    school_rating = rng.integers(1, 11, size=n_samples)
    crime_rate = rng.uniform(0.05, 0.9, size=n_samples).round(2)

    df = pd.DataFrame(
        {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft": sqft,
            "year_built": year_built,
            "city": city,
            "distance_to_center": distance_to_center,
            "school_rating": school_rating,
            "crime_rate": crime_rate,
        }
    )

    city_premium = df["city"].map({"Metro": 75000, "Suburb": 35000, "Rural": -5000})
    age_penalty = (2025 - df["year_built"]) * -450

    base_price = (
        55000
        + df["bedrooms"] * 32000
        + df["bathrooms"] * 21000
        + df["sqft"] * 140
        + df["school_rating"] * 8200
        - df["crime_rate"] * 42000
        - df["distance_to_center"] * 1600
        + city_premium
        + age_penalty
    )

    noise = rng.normal(0, 25000, size=n_samples)
    price = np.maximum(base_price + noise, 75000).round(2)
    df["price"] = price

    desirability = (
        0.45 * df["school_rating"]
        - 0.35 * df["crime_rate"] * 10
        + 0.2 * (df["sqft"] / 1000)
        - 0.15 * (df["distance_to_center"] / 10)
        + df["city"].map({"Metro": 1.2, "Suburb": 0.6, "Rural": -0.3})
    )
    price_pressure = (df["price"] - df["price"].median()) / df["price"].std()
    demand_score = desirability - 0.4 * price_pressure
    demand_prob = 1 / (1 + np.exp(-demand_score))
    df["demand"] = rng.binomial(1, demand_prob)

    df["description"] = generate_descriptions(df)

    for column in [
        "bedrooms",
        "bathrooms",
        "sqft",
        "year_built",
        "city",
        "distance_to_center",
        "school_rating",
        "crime_rate",
        "description",
    ]:
        mask = rng.random(n_samples) < 0.03
        df.loc[mask, column] = np.nan

    return df


def build_preprocessor(text_feature: str, categorical_features: list[str], numeric_features: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=600, ngram_range=(1, 2)), text_feature),
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
        ]
    )
    return preprocessor


def summarize_eda(df: pd.DataFrame, output_path: Path) -> dict:
    numeric_columns = df.select_dtypes(include=["number"]).columns
    eda_summary = {
        "shape": df.shape,
        "missing_values": df.isna().sum().to_dict(),
        "numeric_summary": df[numeric_columns].describe().to_dict(),
        "city_distribution": df["city"].value_counts(dropna=False).to_dict(),
        "demand_distribution": df["demand"].value_counts(dropna=False).to_dict(),
    }
    output_path.write_text(json.dumps(eda_summary, indent=2))
    return eda_summary


def train_models(df: pd.DataFrame, config: PipelineConfig) -> dict:
    df = df.copy()
    df["description"] = df["description"].fillna("")

    features = [
        "bedrooms",
        "bathrooms",
        "sqft",
        "year_built",
        "city",
        "distance_to_center",
        "school_rating",
        "crime_rate",
        "description",
    ]
    text_feature = "description"
    categorical_features = ["city"]
    numeric_features = [
        "bedrooms",
        "bathrooms",
        "sqft",
        "year_built",
        "distance_to_center",
        "school_rating",
        "crime_rate",
    ]

    X = df[features]
    y_class = df["demand"]
    y_reg = df["price"]

    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X,
        y_class,
        y_reg,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y_class,
    )

    preprocessor = build_preprocessor(text_feature, categorical_features, numeric_features)

    classifier = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
            ),
        ]
    )

    regressor = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=1.5)),
        ]
    )

    classifier.fit(X_train, y_class_train)
    regressor.fit(X_train, y_reg_train)

    y_class_pred = classifier.predict(X_test)
    y_reg_pred = regressor.predict(X_test)

    metrics = {
        "classification": {
            "accuracy": float(accuracy_score(y_class_test, y_class_pred)),
            "f1": float(f1_score(y_class_test, y_class_pred)),
        },
        "regression": {
            "rmse": float(np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))),
        },
    }

    plots_dir = config.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_class_test, y_class_pred, ax=ax, colorbar=False)
    ax.set_title("Demand Classification Confusion Matrix")
    fig.tight_layout()
    fig.savefig(plots_dir / "demand_confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_reg_test, y_reg_pred, alpha=0.6, edgecolor="none")
    min_val = min(y_reg_test.min(), y_reg_pred.min())
    max_val = max(y_reg_test.max(), y_reg_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1)
    ax.set_title("Actual vs Predicted Prices")
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    fig.tight_layout()
    fig.savefig(plots_dir / "price_predictions.png", dpi=160)
    plt.close(fig)

    return metrics


def run_pipeline(config: PipelineConfig) -> dict:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_dataset(config.n_samples, config.random_state)
    df.to_csv(output_dir / "synthetic_real_estate.csv", index=False)

    eda_summary = summarize_eda(df, output_dir / "eda_summary.json")
    metrics = train_models(df, config)

    metrics_payload = {
        "config": {
            "n_samples": config.n_samples,
            "test_size": config.test_size,
            "random_state": config.random_state,
        },
        "metrics": metrics,
        "eda_overview": {
            "shape": eda_summary["shape"],
            "missing_values": eda_summary["missing_values"],
        },
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    return metrics_payload


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Run the real estate demand forecasting pipeline.")
    parser.add_argument("--samples", type=int, default=1200, help="Number of synthetic samples to generate")
    parser.add_argument("--test-size", type=float, default=0.2, help="Share of data for testing")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Output directory")
    parser.add_argument("--plots-dir", type=Path, default=Path("reports"), help="Plots output directory")
    args = parser.parse_args()
    return PipelineConfig(
        n_samples=args.samples,
        test_size=args.test_size,
        random_state=args.random_state,
        output_dir=args.output_dir,
        plots_dir=args.plots_dir,
    )


def main() -> int:
    config = parse_args()
    results = run_pipeline(config)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
