"""
Predictor that loads a trained Supreme Court logistic regression model,
reads one case from JSON, and predicts the case result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack


MODEL_PATH = Path("scotus_lr_model.joblib")
INPUT_PATH = Path("case_input.json")

def validate_index_vocab(
    vocab: Any,
    name: str,
) -> None:
    """
    Verifies that a vocabulary maps strings to unique, contiguous
    integer indices beginning at zero.
    """
    if not isinstance(vocab, dict):
        raise TypeError(
            f"'{name}' must be a dictionary, not "
            f"{type(vocab).__name__}."
        )

    if not vocab:
        raise ValueError(
            f"'{name}' cannot be empty."
        )

    for key, value in vocab.items():
        if not isinstance(key, str):
            raise TypeError(
                f"Every key in '{name}' must be a string."
            )

        if not isinstance(value, int):
            raise TypeError(
                f"Every value in '{name}' must be an integer."
            )

    indices = list(vocab.values())

    if len(indices) != len(set(indices)):
        raise ValueError(
            f"'{name}' contains duplicate numeric indices."
        )

    expected_indices = list(range(len(vocab)))

    if sorted(indices) != expected_indices:
        raise ValueError(
            f"'{name}' indices must be contiguous and start at zero. "
            f"Expected {expected_indices}, found {sorted(indices)}."
        )


def load_model_bundle(path: Path) -> dict[str, Any]:
    """
    Loads the classifier, excerpt vectorizer, justice vocabulary,
    and result vocabulary.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find model file: {path.resolve()}"
        )

    bundle = joblib.load(path)

    if not isinstance(bundle, dict):
        raise TypeError(
            "The loaded joblib file is not a dictionary bundle."
        )

    required_keys = {
        "clf",
        "excerpt_vectorizer",
        "justice_vocab",
        "results_vocab",
    }

    missing_keys = required_keys - set(bundle.keys())

    if missing_keys:
        raise KeyError(
            f"The model bundle is missing: {sorted(missing_keys)}"
        )

    validate_index_vocab(
        bundle["justice_vocab"],
        "justice_vocab",
    )

    validate_index_vocab(
        bundle["results_vocab"],
        "results_vocab",
    )

    return bundle


def load_case(path: Path) -> dict[str, Any]:
    """
    Loads one case from a JSON file.

    Expected fields:
        excerpt: string
        justices: list of strings

    The title/name field is optional because this model does not use it
    as a feature.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find case input file: {path.resolve()}"
        )

    with path.open("r", encoding="utf-8") as file:
        case = json.load(file)

    if not isinstance(case, dict):
        raise TypeError(
            "The case input JSON must contain one JSON object."
        )

    title = str(
        case.get("title", case.get("name", ""))
    ).strip()

    excerpt = str(case.get("excerpt", "")).strip()
    justices = case.get("justices", [])

    if not excerpt:
        raise ValueError(
            "The case must contain a non-empty 'excerpt' field."
        )

    if not isinstance(justices, list):
        raise TypeError(
            "The 'justices' field must be a list of justice names."
        )

    cleaned_justices = []

    for justice in justices:
        justice_name = str(justice).strip()

        if justice_name:
            cleaned_justices.append(justice_name)

    if not cleaned_justices:
        raise ValueError(
            "The case must contain at least one justice in the "
            "'justices' list."
        )

    return {
        "title": title,
        "excerpt": excerpt,
        "justices": cleaned_justices,
    }


def reverse_label_mapping(
    result_vocab: dict[str, int],
) -> dict[int, str]:
    """
    Converts:
        {'affirmed': 0, 'reversed': 1, 'vacated': 2}

    into:
        {0: 'affirmed', 1: 'reversed', 2: 'vacated'}
    """
    return {
        int(label_id): str(label)
        for label, label_id in result_vocab.items()
    }


def normalize_justice_name(name: str) -> str:
    """
    Normalizes a justice name before looking it up in the vocabulary.

    This normalization must match the normalization used while training.
    Modify this function if the training program stored names differently.
    """
    return " ".join(name.strip().upper().split())


def normalize_justice_vocab(
    justice_vocab: dict[str, int],
) -> dict[str, int]:
    """
    Creates a normalized lookup dictionary while preserving the original
    justice feature indices.
    """
    normalized_vocab: dict[str, int] = {}

    for justice_name, column_index in justice_vocab.items():
        normalized_name = normalize_justice_name(
            str(justice_name)
        )

        normalized_vocab[normalized_name] = int(column_index)

    return normalized_vocab


def encode_justices(
    justices: list[str],
    justice_vocab: dict[str, int],
) -> csr_matrix:
    """
    Converts a list of justice names into a one-row multi-hot sparse vector.

    Example:
        justice_vocab = {
            'JOHN G. ROBERTS, JR.': 0,
            'CLARENCE THOMAS': 1,
            'SONIA SOTOMAYOR': 2,
        }

        justices = [
            'CLARENCE THOMAS',
            'SONIA SOTOMAYOR',
        ]

        encoded row:
            [0, 1, 1]
    """
    if not isinstance(justice_vocab, dict):
        raise TypeError(
            "'justice_vocab' must be a dictionary mapping justice "
            "names to feature indices."
        )

    if not justice_vocab:
        raise ValueError("The justice vocabulary is empty.")

    normalized_vocab = normalize_justice_vocab(
        justice_vocab
    )

    justice_vector = np.zeros(
        len(justice_vocab),
        dtype=np.float64,
    )

    unknown_justices = []

    for justice in justices:
        normalized_name = normalize_justice_name(justice)
        column_index = normalized_vocab.get(normalized_name)

        if column_index is None:
            unknown_justices.append(justice)
            continue

        justice_vector[column_index] = 1.0

    if unknown_justices:
        print(
            "\nWarning: The following justices were not found in the "
            "training vocabulary and will be ignored:"
        )

        for justice in unknown_justices:
            print(f"  - {justice}")

    if not justice_vector.any():
        raise ValueError(
            "None of the supplied justice names matched the justice "
            "vocabulary stored with the model."
        )

    return csr_matrix(justice_vector.reshape(1, -1))


def validate_feature_count(
    classifier: Any,
    features: csr_matrix,
) -> None:
    """
    Confirms that the newly created feature row has the same number of
    columns the classifier saw during training.
    """
    expected_count = getattr(
        classifier,
        "n_features_in_",
        None,
    )

    if expected_count is None:
        return

    actual_count = features.shape[1]

    if actual_count != expected_count:
        raise ValueError(
            "Feature-count mismatch.\n"
            f"The classifier expects {expected_count} features, but "
            f"the prediction program created {actual_count}.\n"
            "Confirm that training concatenated features in this order:\n"
            "    [excerpt TF-IDF features, justice multi-hot features]\n"
            "Also confirm that the saved vectorizer and justice "
            "vocabulary came from the same training run."
        )


def predict_case(
    case: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    """
    Converts a raw case into the exact feature layout used during training
    and predicts its result.
    """
    classifier = bundle["clf"]
    excerpt_vectorizer = bundle["excerpt_vectorizer"]
    justice_vocab = bundle["justice_vocab"]
    result_vocab = bundle["results_vocab"]

    if not isinstance(result_vocab, dict):
        raise TypeError(
            "'results_vocab' must be a dictionary mapping labels "
            "to numeric class IDs."
        )

    # Use transform(), never fit_transform(), for new cases.
    excerpt_features = excerpt_vectorizer.transform(
        [case["excerpt"]]
    )

    justice_features = encode_justices(
        case["justices"],
        justice_vocab,
    )

    if excerpt_features.shape[0] != justice_features.shape[0]:
        raise ValueError(
            "Excerpt and justice features do not have the same "
            "number of rows."
        )

    # This order must exactly match the order used during training.
    features = hstack(
        [
            excerpt_features,
            justice_features,
        ],
        format="csr",
    )

    validate_feature_count(classifier, features)

    predicted_raw = classifier.predict(features)[0]

    try:
        predicted_id = int(predicted_raw)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "The classifier returned a non-numeric class label. "
            "This program currently expects numeric labels."
        ) from error

    id_to_result = reverse_label_mapping(result_vocab)

    if predicted_id not in id_to_result:
        raise KeyError(
            f"The classifier predicted class ID {predicted_id}, "
            "but that ID is missing from results_vocab."
        )

    predicted_result = id_to_result[predicted_id]

    prediction: dict[str, Any] = {
        "prediction": predicted_result,
        "predicted_class_id": predicted_id,
    }

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(features)[0]

        probability_mapping: dict[str, float] = {}

        for class_id, probability in zip(
            classifier.classes_,
            probabilities,
        ):
            numeric_class_id = int(class_id)

            if numeric_class_id not in id_to_result:
                raise KeyError(
                    f"Classifier class ID {numeric_class_id} is "
                    "missing from results_vocab."
                )

            label = id_to_result[numeric_class_id]
            probability_mapping[label] = float(probability)

        prediction["probabilities"] = probability_mapping
        prediction["confidence"] = probability_mapping[
            predicted_result
        ]

    return prediction


def main() -> None:
    try:
        bundle = load_model_bundle(MODEL_PATH)
        case = load_case(INPUT_PATH)
        result = predict_case(case, bundle)
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"\nPrediction failed: {error}")
        return

    print("\nCase")

    if case["title"]:
        print(f"Title: {case['title']}")

    print(f"Excerpt: {case['excerpt'][:200]}")

    print("Justices:")

    for justice in case["justices"]:
        print(f"  - {justice}")

    print("\nPrediction")
    print(f"Result: {result['prediction']}")

    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.2%}")

        print("\nClass probabilities")

        sorted_probabilities = sorted(
            result["probabilities"].items(),
            key=lambda item: item[1],
            reverse=True,
        )

        for label, probability in sorted_probabilities:
            print(f"  {label:<10} {probability:.2%}")


if __name__ == "__main__":
    main()

