import torch
import torchvision.transforms as transforms

from PIL import Image
from cnn import CNN


def load_model(model_path):
    """Loads the model from the given path."""
    model = CNN(in_channels=1, num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def preprocess_image(image_path):
    """Preprocesses the image to be compatible with the model."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)


def predict(image_path, model):
    """Predicts the class of the image at the given path."""
    image = preprocess_image(image_path)
    with torch.no_grad():
        scores = model(image)
        _, prediction = scores.max(1)
    return prediction.item()


if __name__ == '__main__':
    # Set device
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example usage
    model_path = 'cnn_model.pth'
    image_path = 'five.png'
    model = load_model(model_path)
    prediction = predict(image_path, model)
    print(f'Predicted class: {prediction}')
