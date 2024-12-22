# Fingerprint to Blood Group Prediction

This project leverages deep learning to predict an individual's blood group based on their fingerprint image. The model uses ResNet9 architecture, which is deployed via a Streamlit web application for an intuitive and interactive user experience.

---

## Demo

🎯 [Live Demo: Fingerprint to Blood Group Prediction](https://fingure-print-to-blood-group.streamlit.app/)

---

## Features

- **Easy-to-Use Interface**: Upload a fingerprint image and receive a blood group prediction instantly.
- **High Accuracy**: Achieves 92% accuracy on the test dataset.
- **Supports 8 Blood Groups**:
  - `A+`, `A-`, `B+`, `B-`, `AB+`, `AB-`, `O+`, `O-`
- **Streamlit Frontend**:
  - Responsive and user-friendly.
  - Sidebar explaining the model details and accuracy.
- **Real-time Processing**:
  - Includes a simulated processing spinner for better user interaction.

---

## Screenshot

![App Screenshot](./image.png)

---

## Installation

### Prerequisites
1. Python 3.8 or above.
2. Required libraries installed (listed in `requirements.txt`).

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/fingerprint-to-blood-group.git
   cd fingerprint-to-blood-group
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the pre-trained model file `FingurePrintTOBloodGroup.pth` in the project directory.

4. Run the Streamlit application:
   ```bash
   streamlit run app_resnet.py
   ```

---

## How It Works

### Model Architecture
The model is based on the ResNet9 architecture, designed for efficient and accurate image classification tasks.

### Steps
1. **Image Upload**:
   - Users upload a fingerprint image in formats like PNG, JPG, JPEG, or BMP.

2. **Preprocessing**:
   - The image is resized to 128x128 pixels and normalized for model compatibility.

3. **Prediction**:
   - The model processes the image and predicts the blood group among 8 possible classes.

4. **Output**:
   - The result is displayed on the screen, along with the uploaded fingerprint.

---

## Files

- `app_resnet.py`: Main script for the Streamlit application.
- `FingurePrintTOBloodGroup.pth`: Pre-trained weights for the ResNet9 model.
- `requirements.txt`: Python dependencies.

---

## Dependencies

- PyTorch
- torchvision
- Streamlit
- Pillow
- Other libraries (see `requirements.txt`).

---

## Usage

### Uploading a Fingerprint
1. Open the [live app](https://fingure-print-to-blood-group.streamlit.app/).
2. Drag and drop or browse to upload a fingerprint image.
3. Wait for the processing to complete and view the predicted blood group.

---

## Contributing

Contributions are welcome! Feel free to fork this repository, make updates, and submit a pull request.

---

## Acknowledgments

- **PyTorch**: For providing the deep learning framework.
- **Streamlit**: For the seamless deployment of the web application.
