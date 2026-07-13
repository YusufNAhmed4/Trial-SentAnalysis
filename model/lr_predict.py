from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import hstack


MODEL_PATH = Path("case_result_model.joblib")
INPUT_PATH = Path("case_input.json")


def load_model_bundle(path: Path) -> dict[str, Any]:
    """
    Loads the classifier, vectorizers, and label mapping.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find model file: {path.resolve()}"
        )

    bundle = joblib.load(path)

    required_keys = {
        "classifier",
        "title_vectorizer",
        "excerpt_vectorizer",
        "result_vocab",
    }

    missing_keys = required_keys - bundle.keys()

    if missing_keys:
        raise KeyError(
            f"The model bundle is missing: {sorted(missing_keys)}"
        )

    return bundle


def load_case(path: Path) -> dict[str, Any]:
    """
    Loads one case from a JSON file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find case input file: {path.resolve()}"
        )

    with path.open("r", encoding="utf-8") as file:
        case = json.load(file)

    title = str(case.get("title", case.get("name", ""))).strip()
    excerpt = str(case.get("excerpt", "")).strip()

    if not title and not excerpt:
        raise ValueError(
            "The case must contain a title, an excerpt, or both."
        )

    return {
        "title": title,
        "excerpt": excerpt,
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
        label_id: label
        for label, label_id in result_vocab.items()
    }


def predict_case(
    case: dict[str, str],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    """
    Converts the raw case into the same features used during training
    and predicts the result.
    """
    classifier = bundle["classifier"]
    title_vectorizer = bundle["title_vectorizer"]
    excerpt_vectorizer = bundle["excerpt_vectorizer"]
    result_vocab = bundle["result_vocab"]

    # transform(), not fit_transform().
    # The vectorizers must retain the training vocabulary.
    title_features = title_vectorizer.transform([case["title"]])
    excerpt_features = excerpt_vectorizer.transform([case["excerpt"]])

    # This order must exactly match the order used during training.
    features = hstack(
        [
            title_features,
            excerpt_features,
        ],
        format="csr",
    )

    predicted_id = int(classifier.predict(features)[0])

    id_to_result = reverse_label_mapping(result_vocab)
    predicted_result = id_to_result[predicted_id]

    prediction = {
        "prediction": predicted_result,
        "predicted_class_id": predicted_id,
    }

    # LogisticRegression normally supports predict_proba().
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(features)[0]

        prediction["probabilities"] = {
            id_to_result[int(class_id)]: float(probability)
            for class_id, probability in zip(
                classifier.classes_,
                probabilities,
            )
        }

        prediction["confidence"] = float(
            max(probabilities)
        )

    return prediction


def main() -> None:
    bundle = load_model_bundle(MODEL_PATH)
    case = load_case(INPUT_PATH)
    result = predict_case(case, bundle)

    print("\nCase")
    print(f"Title: {case['title']}")
    print(f"Excerpt: {case['excerpt'][:200]}")

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