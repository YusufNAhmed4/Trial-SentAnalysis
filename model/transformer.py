"""
Transformer predictor
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = Path("output.jsonl")
OUTPUT_DIR = Path("models/legal_bert_case_result")

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

LABEL_TO_ID = {
    "affirmed": 0,
    "reversed": 1,
    "vacated": 2,
}

ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}

SEED = 42
MAX_LENGTH = 256

TRAIN_END_YEAR = 2003
VALIDATION_END_YEAR = 2005

BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
NUM_EPOCHS = 8

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

EARLY_STOPPING_PATIENCE = 2
USE_CLASS_WEIGHTS = True


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Sets seed for reproducability
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def normalize_result(result: Any) -> str | None:
    """
    Rechecks if results are "affirmed", "reversed", "vacated".
    """
    if result is None:
        return None

    normalized = str(result).strip().lower()

    # This permits values such as "reversed and remanded".
    first_word = normalized.split()[0] if normalized else ""

    if first_word in LABEL_TO_ID:
        return first_word

    return None


def load_cases(path: Path) -> list[dict[str, Any]]:
    """
    Loads all cases from path and turns them into a list of JSON-style dicts
    """
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset: {path.resolve()}")

    cases: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                case = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            result = normalize_result(case.get("result"))
            title = case.get("name") or case.get("title") or ""
            excerpt = case.get("excerpt") or ""

            try:
                year = int(case["year"])
            except (KeyError, TypeError, ValueError):
                continue

            # Empty cases do not provide useful textual input.
            if not title.strip() and not excerpt.strip():
                continue

            if result is None:
                continue

            cases.append(
                {
                    "year": year,
                    "title": str(title).strip(),
                    "excerpt": str(excerpt).strip(),
                    "result": result,
                    "label": LABEL_TO_ID[result],
                }
            )

    if not cases:
        raise ValueError("No valid cases were loaded from the dataset.")

    return cases


def temporal_split(
    cases: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Training:   year <= TRAIN_END_YEAR
    Validation: TRAIN_END_YEAR < year <= VALIDATION_END_YEAR
    Test:       year > VALIDATION_END_YEAR

    Change the boundary years if these produce splits that are too small.
    """
    training = [
        case for case in cases
        if case["year"] <= TRAIN_END_YEAR
    ]

    validation = [
        case for case in cases
        if TRAIN_END_YEAR < case["year"] <= VALIDATION_END_YEAR
    ]

    test = [
        case for case in cases
        if case["year"] > VALIDATION_END_YEAR
    ]

    for split_name, split in (
        ("train", training),
        ("validation", validation),
        ("test", test),
    ):
        if not split:
            raise ValueError(
                f"The {split_name} split is empty. "
                "Adjust the temporal boundary years."
            )

    return training, validation, test


def print_split_summary(
    name: str,
    cases: list[dict[str, Any]],
) -> None:
    
    """
    Writes a short summary detailing when the cases were split
    """

    years = [case["year"] for case in cases]
    counts = Counter(case["result"] for case in cases)

    print(
        f"{name}: {len(cases)} cases, "
        f"years {min(years)}-{max(years)}, "
        f"labels={dict(counts)}"
    )


# ---------------------------------------------------------------------------
# PyTorch dataset
# ---------------------------------------------------------------------------

class CaseDataset(Dataset):

    """
    Class describing the case list
    """

    def __init__(
        self,
        cases: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.cases = cases
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        case = self.cases[index]

        # Passing two strings lets BERT create:
        # [CLS] title [SEP] excerpt [SEP]
        encoded = self.tokenizer(
            case["title"],
            case["excerpt"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        encoded["labels"] = case["label"]
        return encoded


@dataclass
class CaseBatchCollator:
    """
    Class describing a batch maker for the cases
    """
    tokenizer: Any

    def __call__(
        self,
        examples: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        labels = torch.tensor(
            [example.pop("labels") for example in examples],
            dtype=torch.long,
        )

        batch = self.tokenizer.pad(
            examples,
            padding=True,
            return_tensors="pt",
        )

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Metrics and loss
# ---------------------------------------------------------------------------

def calculate_class_weights(
    train_cases: list[dict[str, Any]],
    device: torch.device,
) -> torch.Tensor:
    """
    Calculate class weights for Transformer
    """

    counts = Counter(case["label"] for case in train_cases)
    total = len(train_cases)
    number_of_classes = len(LABEL_TO_ID)

    weights = [
        total / (number_of_classes * counts[class_id])
        for class_id in range(number_of_classes)
    ]

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


def calculate_metrics(
    labels: list[int],
    predictions: list[int],
) -> dict[str, float]:
    """
    Calculates important stats about Transformer
    """

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_function: CrossEntropyLoss,
    device: torch.device,
) -> tuple[float, dict[str, float], list[int], list[int]]:
    """
    Runs the Transformer
    """

    model.eval()

    total_loss = 0.0
    labels_all: list[int] = []
    predictions_all: list[int] = []

    for batch in tqdm(data_loader, desc="Evaluating", leave=False):
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        labels = batch.pop("labels")
        outputs = model(**batch)

        loss = loss_function(outputs.logits, labels)
        predictions = outputs.logits.argmax(dim=-1)

        total_loss += loss.item()
        labels_all.extend(labels.cpu().tolist())
        predictions_all.extend(predictions.cpu().tolist())

    average_loss = total_loss / max(len(data_loader), 1)
    metrics = calculate_metrics(labels_all, predictions_all)

    return average_loss, metrics, labels_all, predictions_all


# ---------------------------------------------------------------------------
# Model saving
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    """
    Saves the Transformer
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with (output_dir / "training_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> None:
    """
    Trains the Transformer
    """
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    print(f"Using device: {device}")

    cases = load_cases(DATA_PATH)
    train_cases, validation_cases, test_cases = temporal_split(cases)

    print_split_summary("Train", train_cases)
    print_split_summary("Validation", validation_cases)
    print_split_summary("Test", test_cases)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    model.to(device)

    collator = CaseBatchCollator(tokenizer)

    train_dataset = CaseDataset(
        train_cases,
        tokenizer,
        MAX_LENGTH,
    )
    validation_dataset = CaseDataset(
        validation_cases,
        tokenizer,
        MAX_LENGTH,
    )
    test_dataset = CaseDataset(
        test_cases,
        tokenizer,
        MAX_LENGTH,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    if USE_CLASS_WEIGHTS:
        class_weights = calculate_class_weights(
            train_cases,
            device,
        )
        print(f"Class weights: {class_weights.tolist()}")
    else:
        class_weights = None

    loss_function = CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    optimizer_steps_per_epoch = int(
        np.ceil(
            len(train_loader) /
            GRADIENT_ACCUMULATION_STEPS
        )
    )

    total_optimizer_steps = (
        optimizer_steps_per_epoch * NUM_EPOCHS
    )

    warmup_steps = int(total_optimizer_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    best_macro_f1 = -1.0
    epochs_without_improvement = 0

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{NUM_EPOCHS}",
        )

        for batch_index, batch in enumerate(progress, start=1):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            labels = batch.pop("labels")

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                outputs = model(**batch)
                loss = loss_function(outputs.logits, labels)
                scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(scaled_loss).backward()

            running_loss += loss.item()

            should_step = (
                batch_index % GRADIENT_ACCUMULATION_STEPS == 0
                or batch_index == len(train_loader)
            )

            if should_step:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                scaler.step(optimizer)
                scaler.update()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(
                train_loss=running_loss / batch_index
            )

        validation_loss, validation_metrics, _, _ = evaluate(
            model,
            validation_loader,
            loss_function,
            device,
        )

        print(
            f"\nEpoch {epoch}: "
            f"validation_loss={validation_loss:.4f}, "
            f"accuracy={validation_metrics['accuracy']:.4f}, "
            f"macro_f1={validation_metrics['macro_f1']:.4f}, "
            f"weighted_f1={validation_metrics['weighted_f1']:.4f}"
        )

        current_macro_f1 = validation_metrics["macro_f1"]

        if current_macro_f1 > best_macro_f1:
            best_macro_f1 = current_macro_f1
            epochs_without_improvement = 0

            save_checkpoint(
                model,
                tokenizer,
                OUTPUT_DIR,
                {
                    "base_model": MODEL_NAME,
                    "label_to_id": LABEL_TO_ID,
                    "max_length": MAX_LENGTH,
                    "best_validation_macro_f1": best_macro_f1,
                    "train_end_year": TRAIN_END_YEAR,
                    "validation_end_year": VALIDATION_END_YEAR,
                },
            )

            print("Saved new best checkpoint.")
        else:
            epochs_without_improvement += 1
            print(
                "No validation improvement. "
                f"Patience: {epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

            if (
                epochs_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):
                print("Early stopping.")
                break

    # Reload the best validation checkpoint before testing.
    best_model = AutoModelForSequenceClassification.from_pretrained(
        OUTPUT_DIR
    )
    best_model.to(device)

    test_loss, test_metrics, test_labels, test_predictions = evaluate(
        best_model,
        test_loader,
        loss_function,
        device,
    )

    print("\nFinal temporal test results")
    print(f"Loss:       {test_loss:.4f}")
    print(f"Accuracy:   {test_metrics['accuracy']:.4f}")
    print(f"Macro F1:   {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1:{test_metrics['weighted_f1']:.4f}")

    print("\nConfusion matrix")
    print(
        confusion_matrix(
            test_labels,
            test_predictions,
            labels=[0, 1, 2],
        )
    )

    print("\nClassification report")
    print(
        classification_report(
            test_labels,
            test_predictions,
            labels=[0, 1, 2],
            target_names=[
                ID_TO_LABEL[0],
                ID_TO_LABEL[1],
                ID_TO_LABEL[2],
            ],
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    train()