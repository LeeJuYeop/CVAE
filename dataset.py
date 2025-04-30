import torchvision
import torchvision.transforms as transforms

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from tqdm import tqdm


download_root = "./MNIST_DATASET"
batch_size = 100

mnist_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])

train_dataset = MNIST(root=download_root, 
                      transform=mnist_transforms,
                      train=True,
                      download=True)

test_dataset = MNIST(root=download_root,
                     transform=mnist_transforms,
                     train=False,
                     download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)
