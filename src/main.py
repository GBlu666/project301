import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

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

def load_dataset_from_dir(train_dir, test_dir, val_split=0.2):
    train_transforms, test_transforms = data_augmentation()

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)

    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("Successfully loaded dataset")

    return train_loader, val_loader, test_loader



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

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10, device='cpu'):
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = 100.0 * train_correct / train_total

        train_loss_history.append(train_loss)
        train_acc_history.append(train_accuracy)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100.0 * val_correct / val_total

        val_loss_history.append(val_loss)
        val_acc_history.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    return train_loss_history, train_acc_history, val_loss_history, val_acc_history

'''def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy}%')
    return np.array(true_labels), np.array(predicted_labels)'''


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

def plot_confusion_matrix(true_labels, predicted_labels, classes):
    cm = confusion_matrix(true_labels, predicted_labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(cm, cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.show()

def plot_training_curve(train_loss_history, train_acc_history, val_loss_history, val_acc_history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy}%')

    # Calculate evaluation metrics
    precision = precision_score(true_labels, predicted_labels, average=None)
    recall = recall_score(true_labels, predicted_labels, average=None)
    f1 = f1_score(true_labels, predicted_labels, average=None)

    # Print evaluation metrics for each class
    classes = ['caoshu', 'kaishu', 'lishu', 'zhuanshu']  # Replace with your actual class names
    for i in range(len(classes)):
        print(f'Class: {classes[i]}')
        print(f'Precision: {precision[i]:.4f}')
        print(f'Recall: {recall[i]:.4f}')
        print(f'F1-score: {f1[i]:.4f}')
        print()

    return np.array(true_labels), np.array(predicted_labels)

def main(load_dir="/root/calligraphy_classifier/data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    train_dir = "/Users/garybluedemac/Desktop/advance_topic/project/project301/project301/data"
    test_dir = "/Users/garybluedemac/Desktop/advance_topic/project/project301/project301/data_test"
    train_loader, val_loader, test_loader = load_dataset_from_dir(train_dir, test_dir)

    model = CalligraphyCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 20
    #train_loss, train_accuracy = train_model(model, criterion, optimizer, train_loader, device=device, num_epochs=20)
    
    train_loss_history, train_acc_history, val_loss_history, val_acc_history = train_model(
        model, criterion, optimizer, train_loader, val_loader, num_epochs, device
    )
    
    plot_training_curve(train_loss_history, train_acc_history, val_loss_history, val_acc_history)
    
    true_labels, predicted_labels = evaluate_model(model, test_loader, device)
    test_accuracy = evaluate_model(model, test_loader, device)
    print("test_acc: ", test_accuracy)


    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the plots
    plt.show()
    
    # Plot confusion matrix
    classes = ['caoshu', 'kaishu', 'lishu', 'zhuanshu']  # Replace with your actual class names
    plot_confusion_matrix(true_labels, predicted_labels, classes)
    
if __name__ == "__main__":
    load_dir = "/Users/garybluedemac/Desktop/advance_topic/project/project301/project301/data"  # path to preoprocessed dataset for train only
    main(load_dir=load_dir)
