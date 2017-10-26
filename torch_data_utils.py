import torchvision.datasets as dset
from torchvision import datasets, transforms
import torch

class FashionMNIST(dset.MNIST):
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]

def load_fashion_mnist(shuffle=True, batch_size = 64, num_workers=1, download = True, root_dir='./'):
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = FashionMNIST(root=root_dir, train=True, transform=data_transform, download=True)
    test_set = FashionMNIST(root=root_dir, train=False, transform=data_transform, download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=10000, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader