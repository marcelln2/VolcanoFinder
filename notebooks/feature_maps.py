import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from projects.VolcanoFinder.models import MyFirstCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyFirstCNN().to(device)
img_path = 'C:/Users/spec/Documents/programming/projects/VolcanoFinder/data/train/volcanoes/0_11.jpg'

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

img = Image.open(img_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    fmap1, fmap2 = model.get_feature_maps(input_tensor)

def show_feature_maps(feature_maps, layer_name, num_channels=6):
    fmap = feature_maps[0].cpu()
    plt.figure(figsize=(12, 4))
    for i in range(num_channels):
        plt.subplot(1, num_channels, i + 1)
        plt.imshow(fmap[i].detach().numpy(), cmap='viridis')
        plt.axis('off')
    plt.suptitle(f'{layer_name} Feature Maps')
    plt.show()

show_feature_maps(fmap1, 'Conv1')
show_feature_maps(fmap2, 'Conv2')