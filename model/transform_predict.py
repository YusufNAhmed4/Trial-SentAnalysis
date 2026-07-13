from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_PATH = Path("models/legal_bert_case_result")
MAX_LENGTH = 512


class CaseResultPredictor:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Could not find model at {model_path.resolve()}"
            )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(model_path)
            .to(self.device)
        )

        self.model.eval()

    @torch.inference_mode()
    def predict(
        self,
        title: str,
        excerpt: str,
    ) -> dict[str, Any]:
        inputs = self.tokenizer(
            title or "",
            excerpt or "",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.model(**inputs)
        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]

        predicted_id = int(probabilities.argmax().item())
        predicted_label = self.model.config.id2label[predicted_id]

        scores = {
            self.model.config.id2label[index]: float(probability)
            for index, probability in enumerate(
                probabilities.cpu().tolist()
            )
        }

        return {
            "prediction": predicted_label,
            "confidence": scores[predicted_label],
            "probabilities": scores,
        }


if __name__ == "__main__":
    predictor = CaseResultPredictor()

    result = predictor.predict(
        title="EXAMPLE PETITIONER v. EXAMPLE RESPONDENT",
        excerpt=(
            "The Court of Appeals affirmed the District Court's "
            "judgment. Petitioners contend that the statute exceeds "
            "Congress's authority under the Commerce Clause..."
        ),
    )

    print(result)