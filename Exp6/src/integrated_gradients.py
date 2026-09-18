import torch


def integrated_gradients(
    model,
    image_tensor=None,
    target_class=None,
    baseline=None,
    steps=50,
    mean=None,
    std=None,
    input_tensor=None,
):
    """
    Compute Integrated Gradients for a target ImageNet class.

    Parameters
    ----------
    model : torch.nn.Module
        InceptionV3 model.

    image_tensor : torch.Tensor
        RGB image tensor [1, 3, H, W] with values in [0, 1].

    target_class : int
        ImageNet target class index.

    baseline : torch.Tensor, optional
        Baseline image with the same shape as image_tensor.

    steps : int
        Number of integration steps.

    mean, std : torch.Tensor, optional
        ImageNet normalization tensors.

    input_tensor : torch.Tensor, optional
        Alias for image_tensor.

    Returns
    -------
    torch.Tensor
        Integrated-gradient attribution tensor.
    """

    # ------------------------------------------------------------------
    # Backward compatibility with both parameter names
    # ------------------------------------------------------------------

    if image_tensor is None:
        image_tensor = input_tensor

    if image_tensor is None:
        raise ValueError(
            "Either image_tensor or input_tensor must be provided."
        )

    device = image_tensor.device

    # ------------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------------

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    if image_tensor.ndim != 4:
        raise ValueError(
            f"Expected [B, C, H, W], got {image_tensor.shape}"
        )

    if image_tensor.shape[1] != 3:
        raise ValueError(
            "Integrated Gradients requires an RGB image "
            f"with 3 channels, got {image_tensor.shape}"
        )

    image_tensor = image_tensor.float().to(device)

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    if baseline is None:
        baseline = torch.zeros_like(image_tensor)
    else:
        if baseline.ndim == 3:
            baseline = baseline.unsqueeze(0)

        baseline = baseline.float().to(device)

        if baseline.shape != image_tensor.shape:
            raise ValueError(
                f"Baseline shape {baseline.shape} does not match "
                f"image shape {image_tensor.shape}"
            )

    # ------------------------------------------------------------------
    # ImageNet normalization
    # ------------------------------------------------------------------

    if mean is None:
        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32,
            device=device,
        ).view(1, 3, 1, 1)
    else:
        mean = mean.to(device)

    if std is None:
        std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32,
            device=device,
        ).view(1, 3, 1, 1)
    else:
        std = std.to(device)

    # ------------------------------------------------------------------
    # Integrated Gradients
    # ------------------------------------------------------------------

    difference = image_tensor - baseline

    total_gradients = torch.zeros_like(image_tensor)

    for step in range(1, steps + 1):

        alpha = step / steps

        interpolated = (
            baseline + alpha * difference
        ).detach()

        interpolated.requires_grad_(True)

        # Normalize raw [0,1] image before InceptionV3.
        normalized = (interpolated - mean) / std

        output = model(normalized)

        # Handle torchvision InceptionOutputs if returned.
        if hasattr(output, "logits"):
            output = output.logits

        target_output = output[:, target_class].sum()

        gradients = torch.autograd.grad(
            target_output,
            interpolated,
            retain_graph=False,
            create_graph=False,
        )[0]

        total_gradients += gradients

    # Average gradients along the integration path.
    average_gradients = total_gradients / steps

    # Integrated Gradients:
    # (input - baseline) * average_gradient
    attributions = difference * average_gradients

    return attributions.detach()