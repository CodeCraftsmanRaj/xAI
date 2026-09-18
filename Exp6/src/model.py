import torch
from torchvision.models import (
    Inception_V3_Weights,
    inception_v3,
)


class InceptionClassifier:
    def __init__(
        self,
        device: torch.device,
        weights_name: str = "DEFAULT",
        num_classes: int = 1000,
        **kwargs,
    ):
        self.device = device

        # Pretrained InceptionV3 requires aux_logits=True
        # in the installed torchvision version.
        weights = Inception_V3_Weights.DEFAULT

        self.model = inception_v3(
            weights=weights,
            aux_logits=True,
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)

        self.std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)

        self.class_names = weights.meta["categories"]

    # ------------------------------------------------------------------
    # IMAGE CONVERSION
    # ------------------------------------------------------------------

    def image_to_tensor(self, image):
        """
        Convert an RGB NumPy/PIL image into a tensor.

        Returns an unbatched tensor:
            [3, H, W]

        The existing xrai.py adds the batch dimension itself.
        """

        import numpy as np

        if hasattr(image, "convert"):
            image = np.asarray(image.convert("RGB"))

        image = np.asarray(image)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected RGB image with shape [H, W, 3], got {image.shape}"
            )

        tensor = torch.from_numpy(
            image.astype(np.float32)
        )

        # Convert [H, W, C] -> [C, H, W]
        tensor = tensor.permute(2, 0, 1)

        # Convert [0,255] -> [0,1]
        if tensor.max() > 1.0:
            tensor = tensor / 255.0

        return tensor.to(self.device)

    # ------------------------------------------------------------------
    # PREPROCESSING
    # ------------------------------------------------------------------

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        ImageNet normalization.

        Input:
            [B, 3, H, W] in [0,1]
        """

        return (x - self.mean) / self.std

    # ------------------------------------------------------------------
    # MODEL FORWARD
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)

        output = self.model(x)

        # InceptionV3 returns the main logits during evaluation.
        if hasattr(output, "logits"):
            output = output.logits

        return output

    # ------------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, x):
        """
        Predict an ImageNet class and return top-5 predictions.

        Accepts:
            - NumPy/PIL image
            - Tensor [3, H, W]
            - Tensor [1, 3, H, W]
        """

        if not isinstance(x, torch.Tensor):
            x = self.image_to_tensor(x)

        x = x.to(self.device)

        if x.ndim == 3:
            x = x.unsqueeze(0)

        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                f"Expected [B, 3, H, W], got {x.shape}"
            )

        logits = self.forward(x)

        probabilities = torch.softmax(logits, dim=1)

        confidence, class_index = probabilities.max(dim=1)

        # Top-5 predictions
        top5_confidence, top5_indices = torch.topk(
            probabilities,
            k=5,
            dim=1,
        )

        top5 = []

        for index, conf in zip(
            top5_indices[0].tolist(),
            top5_confidence[0].tolist(),
        ):
            top5.append(
                {
                    "class_index": int(index),
                    "class_name": self.class_name(int(index)),
                    "confidence": float(conf),
                }
            )

        class_index = class_index.item()
        confidence = confidence.item()

        return {
            "class_index": class_index,
            "class_name": self.class_name(class_index),
            "confidence": confidence,
            "top5": top5,
        }

    # ------------------------------------------------------------------
    # SINGLE IMAGE SCORE
    # ------------------------------------------------------------------

    def score(self, x, class_index: int) -> float:
        if not isinstance(x, torch.Tensor):
            x = self.image_to_tensor(x)

        x = x.to(self.device)

        if x.ndim == 3:
            x = x.unsqueeze(0)

        logits = self.forward(x)

        probabilities = torch.softmax(logits, dim=1)

        return probabilities[0, class_index].item()
    # ------------------------------------------------------------------
    # BATCH SCORE
    # ------------------------------------------------------------------

    def score_batch(
        self,
        x: torch.Tensor,
        class_index: int,
    ) -> torch.Tensor:

        logits = self.forward(x)

        probabilities = torch.softmax(logits, dim=1)

        return probabilities[:, class_index]

    # ------------------------------------------------------------------
    # CLASS NAME
    # ------------------------------------------------------------------

    def class_name(self, class_index: int) -> str:
        return self.class_names[class_index]