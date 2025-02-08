import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch

# Load EMNIST without transformations
dataset = torchvision.datasets.EMNIST(root="models/data", split="letters", train=True, download=True)

# Extract an image
img, label = dataset[0]

# Convert to tensor
img_tensor = transforms.ToTensor()(img)

# Apply rotation manually
img_fixed = torch.rot90(img_tensor, k=1, dims=[1, 2])  # Matches your training transform
img_fixed = torch.flip(img_fixed, dims=[1])

# Plot before & after
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(img_tensor.squeeze(), cmap="gray")
axes[0].set_title("Original EMNIST (Rotated 90° CW)")

axes[1].imshow(img_fixed.squeeze(), cmap="gray")
axes[1].set_title("After Training Rotation (Fixed)")

plt.show()