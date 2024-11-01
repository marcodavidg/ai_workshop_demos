import torch  # PyTorch library
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # For data augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Dataset managment
import matplotlib.pyplot as plt  # For plotting
from tqdm import tqdm  # For progress bars

from cnn import CNN


def set_device():
    """Sets the device to GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_hyperparameters():
    """Returns a dictionary of hyperparameters."""
    return {
        "in_channels": 1,  # MNIST dataset is grayscale
        "num_classes": 10,  # 10 different digits
        "learning_rate": 3e-4,  # karpathy's constant
        "batch_size": 64,
        "num_epochs": 3
    }

def load_data(batch_size):
    """Creates test and train data loaders for the MNIST dataset."""
    train_dataset = datasets.MNIST(
        root="dataset/", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = datasets.MNIST(
        root="dataset/", train=False, transform=transforms.ToTensor(), download=True
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def show_test_data(loader, num_images=5):
    """Displays a few images from the test set."""
    data_iter = iter(loader)
    images, labels = next(data_iter)
    
    images = images[:num_images]
    labels = labels[:num_images]
    
    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title(f'Label: {labels[i].item()}')
        ax.axis('off')
    plt.show()

def initialize_model(in_channels, num_classes, learning_rate):
    """Initializes the model, loss, and optimizer."""
    model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer

def train(model, train_loader, criterion, optimizer, num_epochs):
    """Trains the model on the training data."""
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Get data to cuda if possible
            data = data.to(device)
            targets = targets.to(device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

def test(model, loader):
    """Tests the model on the test data."""
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples

def main():
    """Main function to train the model."""
    
    # Set device
    global device
    device = set_device()

    # Get hyperparameters
    hyperparams = get_hyperparameters()

    # Load data
    train_loader, test_loader = load_data(hyperparams["batch_size"])

    # Show some test data
    show_test_data(test_loader)

    # Initialize network, loss, and optimizer
    model, criterion, optimizer = initialize_model(
        hyperparams["in_channels"], hyperparams["num_classes"], hyperparams["learning_rate"]
    )

    # Train the model
    train(model, train_loader, criterion, optimizer, hyperparams["num_epochs"])

    # Check accuracy on training & test to see how good the model performs
    train_accuracy = test(model, train_loader)
    test_accuracy = test(model, test_loader)
    print(f"Accuracy on training set: {train_accuracy*100:.2f}%")
    print(f"Accuracy on test set: {test_accuracy*100:.2f}%")

    # Save the model
    model_path = 'models/cnn_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
    