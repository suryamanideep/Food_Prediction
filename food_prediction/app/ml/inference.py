import torch
from PIL import Image
from torchvision import transforms, models
import json
import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(ROOT, "saved_models", "food_model_best.pth"))
CLASSES_PATH = os.environ.get("CLASSES_PATH", os.path.join(ROOT, "saved_models", "classes.json"))




def load_model(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # instantiate model
    model = models.efficientnet_b0(pretrained=False)
    # placeholder classifier size (will be replaced by state_dict load)
    if hasattr(model, "classifier"):
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, 101)
    else:
        model.classifier = torch.nn.Linear(model.classifier.in_features, 101)


    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()


    with open(CLASSES_PATH, "r") as f:
        classes = json.load(f)
    return model, classes, device




def get_transform():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
        [0.229,0.224,0.225])
    ])




def predict_image(model, classes, device, pil_image, topk=3):
    x = get_transform()(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1)[0]
        topk_vals, topk_idx = torch.topk(probs, k=topk)
        results = []
        for p, idx in zip(topk_vals.cpu().numpy(), topk_idx.cpu().numpy()):
            results.append({"label": classes[int(idx)], "confidence": float(p)})
    return results