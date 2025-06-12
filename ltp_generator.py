from transferattack.generation.ltp import LTP
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
from pathlib import Path

checkpoint_path = str((Path(__file__).parent / "weights").resolve())

print(f"Using checkpoint path: {checkpoint_path}")

# Initialize the attack
ltp_attack = LTP(
    model_name="generation",
    checkpoint_path=checkpoint_path
)

# Perform the attack
input_data = torch.randn((1, 3, 224, 224)).cuda()  # Example input tensor
labels = torch.tensor([0])  # Example label tensor
delta = ltp_attack.forward(input_data, labels)
adv_data = input_data + delta

# Display the perturbed image
perturbed_image = adv_data[0].cpu().detach()
transform = transforms.ToPILImage()
image = transform(perturbed_image)

plt.imshow(image)
plt.title("Perturbed Image")
plt.axis("off")
plt.show()