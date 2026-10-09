import torch
import pandas as pd


class SentimentPredictor:
    """Batch prediction wrapper that logs every text sent through predict()."""

    def __init__(self, tokenizer, model, device, id2label, batch_size=16):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.id2label = id2label
        self.batch_size = int(batch_size)
        self.call_records = []
        self.call_index = 0

    def predict_proba(self, texts):
        texts = [str(text) for text in texts]
        for text in texts:
            self.call_records.append({
                "call_index": self.call_index,
                "sentence": text,
            })
            self.call_index += 1

        outputs = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.inference_mode():
                logits = self.model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            outputs.append(probs)

        if not outputs:
            return torch.empty((0, len(self.id2label))).numpy()
        import numpy as np
        return np.concatenate(outputs, axis=0)

    def call_log_frame(self):
        frame = pd.DataFrame(self.call_records, columns=["call_index", "sentence"])
        if not frame.empty:
            probs = self.predict_proba_without_logging(frame["sentence"].tolist())
            for idx, label in sorted(self.id2label.items()):
                frame[f"probability_{label.lower()}"] = probs[:, idx]
        return frame

    def predict_proba_without_logging(self, texts):
        """Score logged texts without adding duplicate entries to the call log."""
        outputs = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512,
                return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.inference_mode():
                probs = torch.softmax(self.model(**encoded).logits, dim=-1).cpu().numpy()
            outputs.append(probs)
        import numpy as np
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0, len(self.id2label)))
