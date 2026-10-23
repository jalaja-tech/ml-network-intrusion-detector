"""A small, executable machine-learning network intrusion detection system."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LOGGER = logging.getLogger("ids")
RANDOM_STATE = 42
SAMPLE_FEATURES = [
    "duration",
    "protocol_type",
    "src_bytes",
    "dst_bytes",
    "wrong_fragment",
    "count",
    "serror_rate",
    "same_srv_rate",
]
LABEL_COLUMNS = ("label", "class", "target", "attack_cat", "attack_category")
NORMAL_LABELS = {
    "0",
    "benign",
    "false",
    "legitimate",
    "normal",
    "non-attack",
    "non_attack",
}


def configure_logging(verbose: bool = False) -> None:
    """Configure concise, structured terminal logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


class NetworkIntrusionDetector:
    """Train, evaluate, persist, and run inference with a binary IDS model."""

    def __init__(self, model_path: str | Path = "ids_model.joblib") -> None:
        self.model_path = Path(model_path)
        self.pipeline: Optional[Pipeline] = None
        self.feature_columns: list[str] = []

    @staticmethod
    def generate_synthetic_data(rows: int = 1_000) -> pd.DataFrame:
        """Create reproducible traffic data with correlated anomaly indicators."""
        rng = np.random.default_rng(RANDOM_STATE)
        labels = rng.binomial(1, 0.28, rows)
        protocols = np.where(
            labels == 1,
            rng.choice(["tcp", "udp", "icmp"], rows, p=[0.58, 0.24, 0.18]),
            rng.choice(["tcp", "udp", "icmp"], rows, p=[0.72, 0.20, 0.08]),
        )

        duration = np.where(
            labels == 1,
            rng.exponential(2.5, rows),
            rng.exponential(8.0, rows),
        )
        src_bytes = np.where(
            labels == 1,
            rng.lognormal(7.0, 1.35, rows),
            rng.lognormal(8.0, 1.15, rows),
        )
        dst_bytes = np.where(
            labels == 1,
            rng.lognormal(5.2, 1.2, rows),
            rng.lognormal(7.4, 1.3, rows),
        )
        wrong_fragment = np.where(
            labels == 1, rng.binomial(1, 0.18, rows), rng.binomial(1, 0.01, rows)
        )
        count = np.where(
            labels == 1,
            rng.poisson(32, rows),
            rng.poisson(12, rows),
        )
        serror_rate = np.clip(
            np.where(labels == 1, rng.beta(3.0, 2.0, rows), rng.beta(0.7, 8.0, rows)),
            0,
            1,
        )
        same_srv_rate = np.clip(
            np.where(labels == 1, rng.beta(2.0, 4.0, rows), rng.beta(7.0, 1.5, rows)),
            0,
            1,
        )

        data = pd.DataFrame(
            {
                "duration": duration,
                "protocol_type": protocols,
                "src_bytes": src_bytes.astype(int),
                "dst_bytes": dst_bytes.astype(int),
                "wrong_fragment": wrong_fragment,
                "count": count,
                "serror_rate": serror_rate,
                "same_srv_rate": same_srv_rate,
                "label": labels,
            }
        )
        # Missing values exercise the same preprocessing path used for real data.
        for column in ("duration", "src_bytes", "protocol_type"):
            missing_rows = rng.choice(rows, size=max(1, rows // 100), replace=False)
            data.loc[missing_rows, column] = np.nan
        return data

    @classmethod
    def load_data(cls, csv_path: Optional[str]) -> pd.DataFrame:
        """Load a CSV when available; otherwise create a synthetic training set."""
        if not csv_path:
            LOGGER.warning("No CSV supplied; generating 1,000 synthetic traffic logs.")
            return cls.generate_synthetic_data()

        path = Path(csv_path).expanduser()
        if not path.is_file():
            LOGGER.warning(
                "CSV file '%s' was not found; generating 1,000 synthetic traffic logs.",
                path,
            )
            return cls.generate_synthetic_data()

        LOGGER.info("Loading dataset from %s", path)
        try:
            data = pd.read_csv(path, low_memory=False)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise ValueError(f"Could not read CSV file '{path}': {exc}") from exc

        if data.empty:
            raise ValueError(f"CSV file '{path}' contains no data rows.")
        return data

    @staticmethod
    def _find_label_column(data: pd.DataFrame) -> str:
        """Find a target column using common IDS dataset conventions."""
        by_normalized_name = {
            re.sub(r"[^a-z0-9]", "", str(column).lower()): str(column)
            for column in data.columns
        }
        for candidate in LABEL_COLUMNS:
            normalized = re.sub(r"[^a-z0-9]", "", candidate)
            if normalized in by_normalized_name:
                return by_normalized_name[normalized]
        raise ValueError(
            "No target column found. Name the binary target 'label', 'class', "
            "'target', or 'attack_cat'."
        )

    @staticmethod
    def _encode_target(values: pd.Series) -> pd.Series:
        """Map common benign labels to 0 and attack/anomaly labels to 1."""
        if values.isna().any():
            raise ValueError("Target column contains missing values; provide complete labels.")

        if pd.api.types.is_bool_dtype(values.dtype):
            encoded = values.astype(int)
        elif pd.api.types.is_numeric_dtype(values.dtype):
            unique_values = set(pd.unique(values))
            if unique_values.issubset({0, 1}):
                encoded = values.astype(int)
            else:
                # IDS multiclass labels conventionally reserve 0 for normal traffic.
                encoded = (values != 0).astype(int)
        else:
            normalized = values.astype(str).str.strip().str.lower().str.rstrip(".")
            encoded = (~normalized.isin(NORMAL_LABELS)).astype(int)

        if encoded.nunique() != 2:
            raise ValueError(
                "The target must contain both normal (0) and intrusion (1) examples."
            )
        return encoded

    def prepare_data(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, str]:
        """Separate features and target, normalizing column names and labels."""
        data = data.copy()
        data.columns = [str(column).strip() for column in data.columns]
        if len(set(data.columns)) != len(data.columns):
            raise ValueError("Dataset contains duplicate column names.")

        label_column = self._find_label_column(data)
        target = self._encode_target(data[label_column])
        features = data.drop(columns=[label_column])
        if features.shape[1] == 0:
            raise ValueError("Dataset has no feature columns.")

        # Drop all-empty columns; imputers cannot infer a useful value from them.
        all_missing = features.columns[features.isna().all()].tolist()
        if all_missing:
            LOGGER.warning("Dropping entirely empty feature columns: %s", all_missing)
            features = features.drop(columns=all_missing)
        if features.shape[1] == 0:
            raise ValueError("Dataset has no usable feature columns.")

        # Convert numeric-looking object columns where possible, preserving categories.
        for column in features.select_dtypes(include=["object", "string"]).columns:
            converted = pd.to_numeric(features[column], errors="coerce")
            if converted.notna().sum() == features[column].notna().sum():
                features[column] = converted

        self.feature_columns = list(features.columns)
        return features, target, label_column

    @staticmethod
    def _make_one_hot_encoder() -> OneHotEncoder:
        """Support both current and older supported scikit-learn versions."""
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    def build_pipeline(self, features: pd.DataFrame) -> Pipeline:
        """Build imputing/scaling and categorical encoding into one model pipeline."""
        numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_columns = [
            column for column in features.columns if column not in numeric_columns
        ]
        transformers: list[tuple[str, Pipeline, list[str]]] = []

        if numeric_columns:
            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("numeric", numeric_pipeline, numeric_columns))
        if categorical_columns:
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", self._make_one_hot_encoder()),
                ]
            )
            transformers.append(("categorical", categorical_pipeline, categorical_columns))

        preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")
        classifier = RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        return Pipeline(
            steps=[("preprocessor", preprocessing), ("classifier", classifier)]
        )

    def train(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare and stratify data, then train the classification pipeline."""
        features, target, label_column = self.prepare_data(data)
        if target.value_counts().min() < 2:
            raise ValueError("At least two examples of each target class are required.")

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=target,
        )
        self.pipeline = self.build_pipeline(x_train)
        LOGGER.info(
            "Training RandomForestClassifier on %d rows and %d features (target: %s).",
            len(x_train),
            len(self.feature_columns),
            label_column,
        )
        self.pipeline.fit(x_train, y_train)
        return x_test, y_test, x_train, y_train

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Log classification metrics and the detailed per-class report."""
        if self.pipeline is None:
            raise RuntimeError("Train the detector before evaluation.")
        predictions = self.pipeline.predict(x_test)

        LOGGER.info("=" * 72)
        LOGGER.info("MODEL EVALUATION")
        LOGGER.info("=" * 72)
        LOGGER.info("Accuracy:  %.4f", accuracy_score(y_test, predictions))
        LOGGER.info(
            "Precision: %.4f",
            precision_score(y_test, predictions, zero_division=0),
        )
        LOGGER.info("Recall:    %.4f", recall_score(y_test, predictions, zero_division=0))
        LOGGER.info("F1-Score:  %.4f", f1_score(y_test, predictions, zero_division=0))
        LOGGER.info("Confusion Matrix (rows=actual, columns=predicted; labels=[0, 1]):\n%s",
                    confusion_matrix(y_test, predictions, labels=[0, 1]))
        LOGGER.info(
            "Classification Report:\n%s",
            classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["NORMAL", "ANOMALY"],
                zero_division=0,
            ),
        )
        LOGGER.info("=" * 72)

    def save(self) -> None:
        """Persist the complete preprocessing and model pipeline."""
        if self.pipeline is None:
            raise RuntimeError("Train the detector before saving.")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"pipeline": self.pipeline, "feature_columns": self.feature_columns},
            self.model_path,
        )
        LOGGER.info("Saved trained pipeline to %s", self.model_path.resolve())

    def load(self) -> None:
        """Load a previously persisted pipeline."""
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        saved = joblib.load(self.model_path)
        if not isinstance(saved, dict) or "pipeline" not in saved:
            raise ValueError(f"Model file '{self.model_path}' has an unsupported format.")
        self.pipeline = saved["pipeline"]
        self.feature_columns = list(saved["feature_columns"])
        LOGGER.info("Loaded trained pipeline from %s", self.model_path.resolve())

    def predict(self, samples: pd.DataFrame) -> list[tuple[int, float]]:
        """Return binary predictions and intrusion probabilities for connection logs."""
        if self.pipeline is None:
            raise RuntimeError("Train or load the detector before inference.")
        aligned = samples.reindex(columns=self.feature_columns)
        predictions = self.pipeline.predict(aligned)
        probabilities = self.pipeline.predict_proba(aligned)[:, 1]
        return [
            (int(prediction), float(probability))
            for prediction, probability in zip(predictions, probabilities)
        ]

    def run_sample_inference(self) -> None:
        """Inspect benign, suspicious, and sparse edge-case connection examples."""
        samples = pd.DataFrame(
            [
                {
                    "duration": 12.4,
                    "protocol_type": "tcp",
                    "src_bytes": 2_400,
                    "dst_bytes": 18_000,
                    "wrong_fragment": 0,
                    "count": 8,
                    "serror_rate": 0.01,
                    "same_srv_rate": 0.92,
                },
                {
                    "duration": 0.02,
                    "protocol_type": "icmp",
                    "src_bytes": 950_000,
                    "dst_bytes": 0,
                    "wrong_fragment": 3,
                    "count": 95,
                    "serror_rate": 0.98,
                    "same_srv_rate": 0.04,
                },
                {
                    "duration": np.nan,
                    "protocol_type": "unknown",
                    "src_bytes": 0,
                    "dst_bytes": np.nan,
                    "wrong_fragment": 0,
                    "count": 1,
                    "serror_rate": np.nan,
                    "same_srv_rate": 0.5,
                },
            ]
        )
        sample_names = ("Benign connection", "Anomalous connection", "Edge case")

        LOGGER.info("=" * 72)
        LOGGER.info("SAMPLE TRAFFIC INSPECTION")
        LOGGER.info("=" * 72)
        for name, (prediction, probability) in zip(sample_names, self.predict(samples)):
            status = "ANOMALY DETECTED" if prediction == 1 else "NORMAL"
            LOGGER.info(
                "%s | Predicted Status: %s | Threat Probability Score: %.2f%%",
                name,
                status,
                probability * 100,
            )
        LOGGER.info("=" * 72)


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and run a Scikit-learn network intrusion detection system."
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to a labeled IDS CSV; synthetic data is generated if omitted or missing.",
    )
    parser.add_argument(
        "--model-path",
        default="ids_model.joblib",
        help="Path for the persisted model pipeline (default: ids_model.joblib).",
    )
    parser.add_argument(
        "--no-inference",
        action="store_true",
        help="Skip the three sample traffic inspections after training.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(args)


def main(args: Optional[Sequence[str]] = None) -> int:
    options = parse_args(args)
    configure_logging(options.verbose)

    try:
        detector = NetworkIntrusionDetector(model_path=options.model_path)
        dataset = detector.load_data(options.csv)
        x_test, y_test, _, _ = detector.train(dataset)
        detector.evaluate(x_test, y_test)
        detector.save()
        if not options.no_inference:
            detector.run_sample_inference()
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())