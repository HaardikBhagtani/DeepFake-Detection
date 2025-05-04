import sys
import torch
import os
from torchvision import transforms
from torchvision.io import read_image
import torch.nn as nn
from timm import create_model


class EnsembleFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientnet = create_model("efficientnet_b3", pretrained=True, num_classes=0)
        self.xception = create_model("xception", pretrained=True, num_classes=0)
        self.resnet50 = create_model("resnet50", pretrained=True, num_classes=0)
        self.eval_mode()

    def eval_mode(self):
        self.efficientnet.eval()
        self.xception.eval()
        self.resnet50.eval()

    def forward(self, x):
        with torch.no_grad():
            return torch.cat([
                self.efficientnet(x),
                self.xception(x),
                self.resnet50(x)
            ], dim=1)

class TemporalEncoderBiLSTMAttn(nn.Module):
    def __init__(self, input_dim=5632, hidden_dim=256):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        return torch.sum(attn_weights * lstm_out, dim=1)

class FeatureDiffClassifier(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def model(framesPath):
    MODEL_PATH = r"model/model.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    MAX_FRAMES = 22

    extractor = EnsembleFeatureExtractor().to(DEVICE)
    encoder = TemporalEncoderBiLSTMAttn().to(DEVICE)
    classifier = FeatureDiffClassifier().to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    encoder.load_state_dict(checkpoint['encoder'])
    classifier.load_state_dict(checkpoint['classifier'])

    encoder.eval()
    classifier.eval()

    return extractor, encoder, classifier, DEVICE, TRANSFORM

def predict(framesPath):
    extractor, encoder, classifier, device, transform = model(framesPath)
    frame_files = sorted(os.listdir(framesPath))
    frames = []

    for f in frame_files:
        img = read_image(os.path.join(framesPath, f))
        img = transform(img)
        frames.append(img)

    if not frames:
        raise ValueError("No frames found in the given video directory.")

    video_tensor = torch.stack(frames).unsqueeze(0).to(device)

    with torch.no_grad():
        B, T, C, H, W = video_tensor.shape
        features = extractor(video_tensor.view(B * T, C, H, W))
        features = features.view(B, T, -1)
        embedding = encoder(features)
        output = classifier(embedding)
        probability = output.item()
        label = "Fake" if probability > 0.5 else "Real"

    return probability, label