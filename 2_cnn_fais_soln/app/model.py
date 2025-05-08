from PIL import Image
import torch
from torchvision import models, transforms
import numpy as np
import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained ResNet50 without classification head
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
model.eval()

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

def extract_embedding(image: Image.Image) -> np.ndarray: 
    """
    Extract a normalized 2048-dim embedding from a PIL image.
    """
    # 
    image = image.convert("RGB")
    x = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x).squeeze().cpu().numpy()
    emb /= np.linalg.norm(emb)
    return emb.astype("float32")


def extract_clip_embedding(image: Image.Image) -> np.ndarray:
    """
    Extract a normalized embedding from a PIL image using CLIP model.
    """
    image = image.convert("RGB")
    # Preprocess image for CLIP
    x = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(x)
    emb = emb.squeeze().cpu().numpy()
    emb /= np.linalg.norm(emb)
    return emb.astype("float32")


def extract_clip_text_embedding(text: str) -> np.ndarray:
    # print(f"Extracting clip text embedding for '{text}'...")
    tokens = clip.tokenize([text], truncate=True).to(device)  # tokenize returns a tensor
    with torch.no_grad():
        text_emb = clip_model.encode_text(tokens)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)  # normalize embeddings
    return text_emb.squeeze().cpu().numpy().astype("float32")