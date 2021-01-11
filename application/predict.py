import argparse
import json
import torchvision
from torchvision import models
import torch
import numpy
from PIL import Image

def load_checkpoint(file):
    checkpoint = torch.load(file)
    model = getattr(torchvision.models, checkpoint["structure"])(pretrained = True)
    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    model.epochs = checkpoint["epochs"]
    return model

def process_image(image):
    if image.width >= image.height:
        image.thumbnail((16777216, 256))
    else: 
        image.thumbnail((256, 16777216))
    image = image.crop((16, 16, 240, 240))
    image = numpy.array(image) / 255
    mean = numpy.array([0.485, 0.456, 0.406])
    std = numpy.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = image.transpose(2, 0, 1)
    return image

def predict(image_path, model, topk):
    model.eval()
    model.cpu()
    image = process_image(Image.open(image_path))
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    probabilities = torch.exp(model.forward(image))
    top_probabilities = probabilities.topk(topk)[0][0]
    top_indices = probabilities.topk(topk)[1][0]
    indices = []
    for i in range(len(model.class_to_idx.items())):
        indices.append(list(model.class_to_idx.items())[i][0])
    labels = []
    for i in range(topk):
        labels.append(indices[top_indices[i]])
    return top_probabilities, labels

parser = argparse.ArgumentParser()
parser.add_argument("path_to_image", type = str, help = "Path to image")
parser.add_argument("checkpoint", type = str, default = "classifier.pth", help = "Checkpoint")
parser.add_argument("--gpu", action="store_true", help = "Use GPU")
parser.add_argument("--topk", type = int, default = 1, help = "Top k most likely classes")
parser.add_argument("--category_names", type = str, default = "cat_to_name.json", help = "Mapping of categories to real names")
args = parser.parse_args() 

with open(args.category_names, "r") as file:
        cat_to_name = json.load(file)

model = load_checkpoint(args.checkpoint)

device = "cpu"
if args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU available")
model.to(device)
model.eval()

probabilities, classes = predict(args.path_to_image, model, args.topk)
probabilities = probabilities.detach().numpy()
names = []
for i in classes:
    names.append(cat_to_name[i])

for i in range(len(probabilities)):
    print(f"{probabilities[i]:.3f} chance of {names[i]}")