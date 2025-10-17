import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from projects.VolcanoFinder.web.models import MyFirstCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MyFirstCNN().to(device)
model.load_state_dict(torch.load(
    os.path.join(os.path.dirname(__file__), "..", "notebooks", "8224BCE.pth"),
    map_location=device
))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

image_path = 'C:/Users/spec/Documents/programming/projects/AirSim/captured_images'
image_files = [os.path.join(image_path, f) for f in os.listdir(image_path)]

for image_path in image_files:
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        prediction = (output > 0.5).float().item()

    image_name = os.path.basename(image_path)
    print(f"{image_name} -> {'🌋 Volcano' if prediction == 1.0 else '❌ Not a Volcano'}")