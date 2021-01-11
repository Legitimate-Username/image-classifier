import argparse
import torchvision
from torchvision import datasets, transforms, models
import torch
from torch import nn
from torch import optim
import torch.nn.functional

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type = str, help = "Path to dataset directory")
parser.add_argument("--gpu", action="store_true", help = "Use GPU")
parser.add_argument("--arch", type = str, default = "vgg19", help = "Model architecture")
parser.add_argument("--learning_rate", type = float, default = 0.003, help = "Learning rate")
parser.add_argument("--hidden_units", type = int, default = 1024, help = "Hidden units in hidden layer")
parser.add_argument("--epochs", type = int, default = 10, help = "Number of epochs")
parser.add_argument("--save_dir", type = str, default = "", help = "Save directory")
args = parser.parse_args()

save_dir = args.save_dir
if len(save_dir) > 0:
    if save_dir[0] == "/":
        save_dir = save_dir[1:]
    if save_dir[-1] != "/":
        save_dir = save_dir + "/"
        
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms = {"train": transforms.Compose([transforms.RandomRotation(30), 
                                                   transforms.RandomResizedCrop(224), 
                                                   transforms.RandomHorizontalFlip(), 
                                                   transforms.RandomVerticalFlip(), 
                                                   transforms.ToTensor(), 
                                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), 

                   "valid": transforms.Compose([transforms.Resize(256), 
                                                     transforms.CenterCrop(224), 
                                                     transforms.ToTensor(), 
                                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), 
                   
                   "test": transforms.Compose([transforms.Resize(256), 
                                                  transforms.CenterCrop(224), 
                                                  transforms.ToTensor(), 
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

image_datasets = {"train": datasets.ImageFolder(train_dir, transform = data_transforms["train"]), 
                  "valid": datasets.ImageFolder(valid_dir, transform = data_transforms["valid"]), 
                  "test": datasets.ImageFolder(test_dir, transform = data_transforms["test"])}

dataloaders = {"train": torch.utils.data.DataLoader(image_datasets["train"], batch_size = 64, shuffle = True), 
               "valid": torch.utils.data.DataLoader(image_datasets["valid"], batch_size = 64), 
               "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size = 64)}

if args.arch == "vgg19":
    model = models.vgg19(pretrained = True)
elif args.arch == "vgg16":
    model = models.vgg16(pretrained = True)
elif args.arch == "vgg13":
    model = models.vgg13(pretrained = True)
elif args.arch == "vgg11":
    model = models.vgg11(pretrained = True)
else: 
    raise ValueError("Unknown architechture", args.arch)

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(25088, args.hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(args.hidden_units, 102),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)
device = "cpu"
if args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on GPU")
if device == "cpu":
    print("Training on CPU")
model.to(device)

epochs = args.epochs
steps = 0
running_loss = 0
print_every = 20

for epoch in range(epochs): 
    model.train()
    for inputs, labels in dataloaders["train"]:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in dataloaders["test"]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    running_loss = 0
                    model.train()
            print(f"Epoch {epoch + 1} / {epochs}.. "
                  f"Train loss: {running_loss / print_every:.3f}.. "
                  f"Test loss: {test_loss / len(dataloaders['test']):.3f}.. "
                  f"Test accuracy: {accuracy / len(dataloaders['test']):.3f}")

model.class_to_idx = image_datasets["train"].class_to_idx
model.cpu()
torch.save({"structure": args.arch, 
            "epochs": epochs, 
            "classifier": model.classifier, 
            "optimizer": optimizer.state_dict(), 
            "state_dict": model.state_dict(), 
            "class_to_idx": model.class_to_idx}, 
            save_dir + "classifier.pth")
print("Saved checkpoint to " + save_dir + "classifier.pth")
