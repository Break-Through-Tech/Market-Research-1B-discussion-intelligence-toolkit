"""Compact discourse classifier with a reusable PyTorch training loop.

Requires torch and transformers only when instantiated/trained. Context uses
tokenizer pairs; half the token budget is reserved for the child comment.
"""
from __future__ import annotations

import random
import time
from collections import Counter
from pathlib import Path

from discourse_data import save_json
from evaluation import classification_metrics


class DiscourseTransformer:
    def __init__(self, checkpoint, device=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(self.device)
        import json
        self.settings = json.loads((Path(checkpoint) / "discourse_settings.json").read_text())

    def _encode(self, texts, parents=None):
        length = self.settings["max_length"]
        if self.settings["parent_context"]:
            if parents is None:
                raise ValueError("This checkpoint requires parent texts; use predict_examples")
            budget = (length - self.tokenizer.num_special_tokens_to_add(pair=True)) // 2
            # Truncate each component separately, avoiding loss of the child to a long parent.
            def truncate(values):
                ids = self.tokenizer(list(values), add_special_tokens=False,
                                     truncation=True, max_length=budget)["input_ids"]
                return self.tokenizer.batch_decode(ids, skip_special_tokens=True)
            encoded = self.tokenizer(truncate(parents), truncate(texts), padding=True,
                                     truncation=True, max_length=length, return_tensors="pt")
        else:
            encoded = self.tokenizer(list(texts), padding=True, truncation=True,
                                     max_length=length, return_tensors="pt")
        return {key: value.to(self.device) for key, value in encoded.items()}

    def predict(self, texts, batch_size=32):
        return self._predict(texts, None, batch_size)

    def predict_examples(self, examples, batch_size=32):
        return self._predict([ex.text for ex in examples],
                             [ex.metadata.get("parent_text", "") for ex in examples], batch_size)

    def _predict(self, texts, parents, batch_size):
        self.model.eval()
        predictions = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                inputs = self._encode(texts[start:start + batch_size],
                                      None if parents is None else parents[start:start + batch_size])
                indices = self.model(**inputs).logits.argmax(dim=-1).tolist()
                predictions.extend(self.model.config.id2label[i] for i in indices)
        return predictions


def train_transformer(splits, output, *, model_name="distilbert/distilbert-base-uncased",
                      seed=42, epochs=3, batch_size=16, learning_rate=2e-5,
                      max_length=256, weighted=False, parent_context=False,
                      accumulation_steps=1, mixed_precision=True):
    """Select checkpoints on dev macro F1; never evaluate the test split here.

    An empty output directory is required to avoid overwriting earlier experiments.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if min(epochs, batch_size, accumulation_steps) < 1 or max_length < 8 or learning_rate <= 0:
        raise ValueError("Invalid training settings")
    if not splits.dev:
        raise ValueError("A nonempty dev split is required for checkpoint selection")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    labels = splits.labels
    label2id = {label: i for i, label in enumerate(labels)}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = mixed_precision and device == "cuda"
    settings = dict(model_name=model_name, seed=seed, epochs=epochs, batch_size=batch_size,
                    learning_rate=learning_rate, max_length=max_length, weighted=weighted,
                    parent_context=parent_context, accumulation_steps=accumulation_steps,
                    mixed_precision=amp)
    save_json(output / "run_config.json", settings)
    classifier = object.__new__(DiscourseTransformer)
    classifier.torch, classifier.device, classifier.settings = torch, device, settings
    classifier.tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier.model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), id2label=dict(enumerate(labels)),
        label2id=label2id).to(device)
    counts = Counter(ex.label for ex in splits.train)
    weights = (torch.tensor([len(splits.train) / (len(labels) * counts[label])
                             for label in labels], device=device) if weighted else None)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(classifier.model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loader = DataLoader(splits.train, batch_size=batch_size, shuffle=True, collate_fn=list)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    best, history = -1.0, []
    for epoch in range(epochs):
        classifier.model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        for step, batch in enumerate(loader):
            inputs = classifier._encode([ex.text for ex in batch],
                                        [ex.metadata.get("parent_text", "") for ex in batch])
            targets = torch.tensor([label2id[ex.label] for ex in batch], device=device)
            group_size = min(accumulation_steps, len(loader) - (step // accumulation_steps) * accumulation_steps)
            with torch.autocast(device_type=device, enabled=amp):
                loss = criterion(classifier.model(**inputs).logits, targets)
            total_loss += loss.item()
            scaler.scale(loss / group_size).backward()
            if (step + 1) % accumulation_steps == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(classifier.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        result = classification_metrics([ex.label for ex in splits.dev],
                                        classifier.predict_examples(splits.dev, batch_size))
        history.append({"epoch": epoch + 1, "train_loss": total_loss / len(loader),
                        "elapsed_seconds": time.perf_counter() - started,
                        "dev": result.model_dump()})
        save_json(output / "history.json", history)
        print(f"Epoch {epoch + 1}: dev macro F1={result.macro_f1:.4f}", flush=True)
        if result.macro_f1 > best:
            best = result.macro_f1
            checkpoint = output / "best"
            classifier.model.save_pretrained(checkpoint)
            classifier.tokenizer.save_pretrained(checkpoint)
            save_json(checkpoint / "discourse_settings.json", settings)
    from importlib.metadata import version
    save_json(output / "compute.json", {
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "versions": {name: version(name) for name in ("torch", "transformers", "scikit-learn")},
        "best_dev_macro_f1": best,
    })
    return output / "best"
