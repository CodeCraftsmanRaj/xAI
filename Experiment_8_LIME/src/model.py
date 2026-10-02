import torch
from torchvision.models import resnet50, ResNet50_Weights

def load_model(cfg, device):
    weights = ResNet50_Weights.DEFAULT if cfg["model"]["pretrained"] else None
    model = resnet50(weights=weights)
    model = model.to(device).eval()
    preprocess = weights.transforms() if weights is not None else ResNet50_Weights.DEFAULT.transforms()
    categories = weights.meta["categories"] if weights is not None else ResNet50_Weights.DEFAULT.meta["categories"]
    return model, preprocess, categories

@torch.inference_mode()
def predict_batch(model, images, preprocess, device):
    tensors = torch.stack([preprocess(img) for img in images]).to(device)
    probs = torch.softmax(model(tensors), dim=1)
    return probs.detach().cpu().numpy()
