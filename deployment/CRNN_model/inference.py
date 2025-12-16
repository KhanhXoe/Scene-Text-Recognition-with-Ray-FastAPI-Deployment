import os
import numpy as np

import cv2
from PIL import Image
from torchvision import transforms

import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from crnn_model_structure import CRNN

DET_MODEL_PATH = "/data_hdd_16t/khanhtran/4.SceneTextRecognition/src/deployment/best.pt"
REG_MODEL_PATH = "/data_hdd_16t/khanhtran/4.SceneTextRecognition/train_phase/checkpoints/crnn_best.pth"

CHARS = "-!#$%&()/0123456789:?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]abcdefghijklmnopqrstuvwxyz"
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(sorted(CHARS))}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}

HIDDEN_SIZE = 256
N_LAYERS = 3
DROPOUT_PROB = 0.2
UNFREEZE_LAYERS = 12

transform = transforms.Compose(
    [
        transforms.Resize((100, 420)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

def load_model(DET_MODEL_PATH, REG_MODEL_PATH, DEVICE):
    det_model = YOLO(DET_MODEL_PATH).to(DEVICE)
    reg_model = CRNN(
        vocab_size=len(CHARS),
        hidden_size=HIDDEN_SIZE,
        n_layers=N_LAYERS,
        dropout=DROPOUT_PROB,
        unfreeze_layers=UNFREEZE_LAYERS,
    ).to(DEVICE)
    checkpoints = torch.load(REG_MODEL_PATH, map_location=DEVICE)
    reg_model.load_state_dict(checkpoints['model_state_dict'])
    reg_model.eval()

    return det_model, reg_model


def text_detection(det_model, img_path, device):
    results = det_model(img_path, verbose=False)[0]
    return (
        results.boxes.xyxy.tolist(),
        results.boxes.cls.tolist(),
        results.names,
        results.boxes.conf.tolist(),
    )

def decode(encoded_sequences, IDX_TO_CHAR, blank_char="-"):
    decoded_sequences = []

    for seq in encoded_sequences:
        decoded_label = []
        prev_char = None

        for token in seq:
            if token != 0:
                char = IDX_TO_CHAR[token.item()]
                if char != blank_char:
                    if char != prev_char or prev_char == blank_char:
                        decoded_label.append(char)
                prev_char = char

        decoded_sequences.append("".join(decoded_label))

    return decoded_sequences


def text_recognition(reg_model, transform, img, device):
    transformed_image = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = reg_model(transformed_image).cpu()
    text = decode(logits.permute(1, 0, 2).argmax(2), IDX_TO_CHAR)
    return text


def inference(det_model, reg_model, img_path, device):
    bboxes, classes, names, confs = text_detection(det_model, img_path, device)

    img = Image.open(img_path)
    predictions = []

    for bbox, cls_idx, conf in zip(bboxes, classes, confs):
        x1, y1, x2, y2 = bbox
        name = names[int(cls_idx)]
        cropped_image = img.crop((x1, y1, x2, y2))
        transcribed_text = text_recognition(reg_model, transform, cropped_image, device)
        print(transcribed_text)
        predictions.append((bbox, name, conf, transcribed_text[0]))

    return predictions


def draw_predictions(image, predictions):
        image_array = np.array(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        annotator = Annotator(image_array, font="Arial.ttf", pil=False)
        predictions = sorted(
            predictions, key=lambda x: x[0][1]
        )

        for bbox, class_name, confidence, text in predictions:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]

            color = colors(hash(class_name) % 20, True)

            label = f"{class_name[:3]}{confidence:.1f}:{text}"

            annotator.box_label(
                [x1, y1, x2, y2], label, color=color, txt_color=(255, 255, 255)
            )
        return Image.fromarray(annotator.result())


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
det_model, reg_model = load_model(DET_MODEL_PATH, REG_MODEL_PATH, DEVICE)
det_model.eval()
reg_model.eval()
img_path = r"/data_hdd_16t/khanhtran/4.SceneTextRecognition/src/deployment/download.jpg"
predictions = inference(det_model, reg_model, img_path=img_path, device=DEVICE)
anotated_img = draw_predictions(Image.open(img_path), predictions)
anotated_img.save("annotated_output.png")
print("Saved result to annotated_output.png")