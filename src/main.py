import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
from tqdm import tqdm

import scipy.io
import matplotlib.pyplot as plt

from src.model import CalligraphyCNN

def data_augmentation():
    train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

    test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    return train_transforms, test_transforms

def load_dataset_from_dir(dataset_dir='dataset/'):
    train_transforms, test_transforms = data_augmentation()

    train_dataset = datasets.ImageFolder(root="{}/train".format(dataset_dir), transform=train_transforms)
    test_dataset = datasets.ImageFolder(root="{}/test".format(dataset_dir), transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("Successfully loaded dataset")

    return train_loader, test_loader

def load_dataset_train_only(dataset_dir):
    train_transforms, _ = data_augmentation()

    train_dataset = datasets.ImageFolder(root=dataset_dir, transform=train_transforms)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print("Successfully loaded dataset")

    return train_loader

def load_dataset_from_mat(mat_dir='./CV_1_801.mat', batch_size=32, augment=False):
    mat_contents = load_mat_file(mat_dir)
    print(mat_contents['train'].shape)

    if augment:
        train_transforms, test_transforms = data_augmentation()
        train_images = torch.stack([train_transforms(image) for image in mat_contents['train']])
        test_images = torch.stack([test_transforms(image) for image in mat_contents['test']])
    else:
        train_images = torch.Tensor(mat_contents['train']).unsqueeze(1)  # Add channel dimension
        train_labels = torch.Tensor(mat_contents['train_label'])
        test_images = torch.Tensor(mat_contents['test']).unsqueeze(1)  # Add channel dimension
        test_labels = torch.Tensor(mat_contents['test_label'])

    train_labels = torch.Tensor(mat_contents['train_label'])
    test_labels = torch.Tensor(mat_contents['test_label'])

    # Create TensorDatasets
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Successfully loaded dataset")

    return train_loader, test_loader

def train_model(model, criterion, optimizer, train_loader, device, num_epochs=10):
    print('Start training ...')

    model.train()
    train_loss, train_accuracy = [], []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            # Move data to GPU
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            #_, labels = torch.max(labels.data, 1)
            labels = labels.data
            
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}%')

    print('Training finished')
    return train_loss, train_accuracy

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            # Move data to GPU
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            _, labels = torch.max(labels.data, 1)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy}%')
    return accuracy


def load_mat_file(file_path):
    # Load the MATLAB file
    mat_contents = scipy.io.loadmat(file_path)
    return mat_contents

def get_overview(mat_contents):
    # Print keys and parts of data for inspection
    for key, value in mat_contents.items():
        print(f"Key: {key}")
        try:
            # Assuming the value could be large, only show a small part
            if isinstance(value, (dict, list, tuple)):
                print(f"Type: {type(value)}, Length: {len(value)}")
            else:
                print(f"Type: {type(value)}, Shape: {value.shape}")
                print(value)
        except AttributeError:
            print(f"Value: {value}")


def main(load_dir="/root/calligraphy_classifier/data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    #train_loader, test_loader = load_dataset_from_mat()
    train_loader = load_dataset_train_only(load_dir)

    model = CalligraphyCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_loss, train_accuracy = train_model(model, criterion, optimizer, train_loader, device=device, num_epochs=20)
    #test_accuracy = evaluate_model(model, test_loader, device)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    load_dir = "/root/calligraphy_classifier/data"  # path to preoprocessed dataset for train only
    main(load_dir=load_dir)
