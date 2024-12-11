import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


## This is the best the tested model. 
class EfficientNetEdgeNet(nn.Module):
    def __init__(self):
        super(EfficientNetEdgeNet, self).__init__()
        # Load pretrained EfficientNet (EfficientNet-B0 in this case)
        efficientnet = models.efficientnet_b0(pretrained=True)
        
        # Use all EfficientNet layers except the final classification layer
        self.features = nn.Sequential(*list(efficientnet.children())[:-1])  # Skip the classifier
        
        # Output layer for edge detection
        # Final layer uses the same number of channels as the final EfficientNet output, but outputs 1 channel for edges
        self.edge_out = nn.Conv2d(1280, 1, kernel_size=1)  # EfficientNet-B0 has 1280 channels before the classifier

    def forward(self, x):
        # Pass input through EfficientNet layers
        features = self.features(x)
        
        # Output edge map using final convolution layer
        edge_map = self.edge_out(features)
        
        # Resize to the original input size
        edge_map = F.interpolate(edge_map, size=x.size()[2:], mode='bilinear', align_corners=False)
        
        return torch.sigmoid(edge_map)
    
    def extract_embedding(self, x):
        features = self.features(x)  # Output shape: [batch_size, 1280, H, W]
        embedding = F.adaptive_avg_pool2d(features, (1, 1))  # Shape: [batch_size, 1280, 1, 1]
        embedding = embedding.view(embedding.size(0), -1)    # Flatten to: [batch_size, 1280]
        return embedding


# Load and preprocess images
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def get_image_embedding(model, image_path, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if type(image_path) == str:
        image = load_image(image_path).to(device)
    else:
        image = transform(image).unsqueeze(0).to(device)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        embedding = model.extract_embedding(image)
    return embedding.squeeze().cpu().numpy()

def display_edge_map(image_tensor, edge_map):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Convert image tensor to numpy for visualization
    image_np = image_tensor.squeeze().permute(1, 2, 0).numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # Normalize to [0, 1]
    
    ax[0].imshow(image_np)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    # Process edge map for better visualization
    edge_np = edge_map.squeeze().detach().cpu().numpy()
    edge_np = (edge_np - edge_np.min()) / (edge_np.max() - edge_np.min())  # Normalize to [0, 1]
    
    ax[1].imshow(edge_np, cmap='gray')  # Use grayscale colormap
    ax[1].set_title('Detected Edges')
    ax[1].axis('off')
    
    plt.show()


# Testing EfficientNet-based edge detection model
def test_efficientnet_edge_model(image_paths):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    # Initialize the model
    model = EfficientNetEdgeNet().to(device)
    model.eval()  # Set model to evaluation mode
    
    for image_path in image_paths:
        image = load_image(image_path).to(device)
        
        # Forward pass through the model
        with torch.no_grad():
            edge_map = model(image)
        
        # Display original image and edge map
        display_edge_map(image.cpu(), edge_map.cpu())

if __name__ == "__main__":
    image_path = '../realsense/results/color_avg_path.png'
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu' ## For raspberry pi use only CPU
    model = EfficientNetEdgeNet().to(device)
    embedding = get_image_embedding(model, image_path, device)
    #test_efficientnet_edge_model([image_path])
    print("Image Embedding Shape:", embedding.shape)
    print("Image Embedding:", embedding)
