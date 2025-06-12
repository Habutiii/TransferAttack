import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from timm import create_model
import transferattack
from transferattack.attack import Attack
from transferattack.utils import load_pretrained_model, wrap_model
from transferattack.utils import *
import sys
import os

#set path to import /mnt/data/DeepfakeBench/DeepfakeBench directory
sys.path.append("/mnt/data/DeepfakeBench/DeepfakeBench")

from export_model import load_model




# Create a wrapper class to make our model compatible with the attack
class DFBModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        model, process, configs = load_model(model_name, {"test_batch_size": 1})
        self.model = model.eval().cuda()
        self.configs = configs
        self.process = process
        print(f"[DEBUG] Wrapped model type: {model.__class__.__name__}")
    
    def forward(self, x):

        x=x.cuda()

        output = self.model({"image": x, "label": torch.tensor([1]).cuda()}, inference=True)
        return output['cls']


# Load the attack class from transferattack
attack_class = transferattack.load_attack_class("mifgsm")
print(f"[DEBUG] Loaded attack class: {attack_class.__name__}")

def custom_load_model(self, model_name):
    # print(f"[DEBUG] Custom load_model invoked for model: {model_name}")
    model = DFBModel(model_name)
    return model


attack_class.load_model = custom_load_model

# Directly edit the function in the attacker instance
# attack_class.get_logits = custom_get_logits

attack_class.epsilon = 1 / 255.0
attack_class.targeted = True
# attack_class.alpha = 1 / 255.0
attack_class.random_start = False
attack_class.alpha = 0.01 # Set the step size for perturbation
attack_class.epoch = 10

# Wrap the obtained attack_class instance with AttackWrapper
attacker = attack_class('xception')

image_paths = ["./data/image.png"]

images = [attacker.model.process(Image.open(image_path).convert('RGB')) for image_path in image_paths]  # Load images using PIL


images = torch.stack(images).cuda()  # Combine into a single tensor and move to GPU

labels = torch.tensor([0]).cuda()  # Example label tensor, adjust as needed



perturbations = attacker(images , labels)

# Add perturbation in normalized space
perturbed_images = normalized_images + perturbations

std_tensor = torch.tensor(attacker.model.configs["std"]).view(1, 3, 1, 1).cuda()  # Standard deviation for normalization
mean_tensor = torch.tensor(attacker.model.configs["mean"]).view(1, 3, 1, 1).cuda()  # Mean for normalization

# Denormalize for visualization
denormalized_images = perturbed_images * std_tensor + mean_tensor

# Clamp values to the valid range (0-1)
denormalized_images = torch.clamp(denormalized_images, 0, 1)

transform = transforms.ToPILImage()

for i in range(denormalized_images.shape[0]):
    # Convert tensor to PIL image for visualization
    image = transform(denormalized_images[i].cpu())

    # Display the perturbed image
    plt.imshow(image)
    plt.title("Perturbed Image")
    plt.axis("off")
    plt.show()