import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import streamlit as st
import time  # For adding delay

# Define the accuracy function
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Base class for the model
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))

# Convolution block for ResNet9
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

# ResNet9 Model
class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)  # 64 x 32 x 32
        self.conv2 = conv_block(64, 128, pool=True)  # 128 x 16 x 16
        self.res1 = nn.Sequential(conv_block(128, 128),  # 128 x 16 x 16
                                  conv_block(128, 128))  # 128 x 16 x 16

        self.conv3 = conv_block(128, 256, pool=True)  # 256 x 8 x 8
        self.conv4 = conv_block(256, 512, pool=True)  # 512 x 4 x 4
        self.res2 = nn.Sequential(conv_block(512, 512),  # 512 x 4 x 4
                                  conv_block(512, 512))  # 512 x 4 x 4

        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d(1),  # 512 x 1 x 1
                                        nn.Flatten(),  # 512
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))  # num_classes (8 for blood groups)

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

@st.cache_resource
def load_model():
    # Define the number of classes (ensure it matches your training dataset)
    num_classes = 8  # 8 classes for blood group prediction
    
    # Load model architecture
    model = ResNet9(in_channels=3, num_classes=num_classes)
    
    # Load pre-trained weights
    model.load_state_dict(torch.load('FingurePrintTOBloodGroup.pth', map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode
    
    # Use CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    return model, device

def preprocess_image(image):
    # Define the preprocessing steps
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Streamlit frontend for deployment
st.set_page_config(page_title= "Fingerprint to Blood Group",page_icon="🩸", layout="wide")
st.title("Fingerprint to Blood Group Prediction")

# Sidebar Section
st.sidebar.title("About the Model")
st.sidebar.write("""
- **Model Accuracy:** 92%  
- **How it works:**  
   - The model takes a fingerprint image.
   - It preprocesses the image using resizing and transformation.
   - Using a deep learning model (ResNet9), it predicts one of the 8 blood groups.
""")
st.sidebar.write("---")

st.write("Upload a fingerprint image to predict the blood group.")

# Load the model
model, device = load_model()

# File uploader for the fingerprint image
uploaded_file = st.file_uploader("Upload a fingerprint image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file:
    # Read the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Fingerprint", width=148)  # Smaller image (148x148)
    
    # Spinner for processing
    with st.spinner("Processing the image..."):
        time.sleep(5)  # Simulate processing delay
        
        # Preprocess the image
        processed_image = preprocess_image(image).to(device)
        
        # Perform inference
        with torch.no_grad():
            output = model(processed_image)
            _, predicted_class = torch.max(output, dim=1)
    
    # Blood group mapping (8 classes)
    blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']  # 8 blood groups
    predicted_blood_group = blood_groups[predicted_class.item()]
    
    # Display the result
    st.success(f"Predicted Blood Group: **{predicted_blood_group}**")
