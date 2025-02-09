# **Handwritten Letter Recognition**
A deep learning-powered **handwritten letter recognition** model that allows users to draw **uppercase letters (A-Z)** on a canvas and a trained CNN model will predict them.

## **✅ Features**
✔ Users can **draw letters** on an interactive canvas  
✔ **Preprocessing** done with **OpenCV** to improve recognition  
✔ Predictions using a **trained Convolutional Neural Network (CNN)**  
✔ Model trained on the **EMNIST dataset**  

## **⚙️ Installation**

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/LNikolov01/handwritten_letter_recognition.git
cd handwritten_letter_recognition
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

## **🚀 Usage**

### **Run the Application**
```bash
python3 app.py
```
- Draw an **uppercase letter (A-Z)** in the window that appears.  
- Press **'s'** to classify the letter. (Or **'q'** to quit)  
- The program will print the **recognized letter and confidence score**.

## **📊 Example Output**
```
Predicted Letter: W
Confidence: 96.85%
```

## **📂 Project Structure**
```
handwritten_letter_recognition/
│── app.py                 # Main application script
│── test_rotations.py      # A testing script for the preprocessing
│── train_model_letter.py  # Model training script
│── draw_letter.py         # Allows users to draw letters
│── process_image.py       # Image preprocessing for the model
│── requirements.txt       # Dependencies
│── cnn_model_letters.h5   # Trained CNN model
│── README.md              # Project documentation
```

## **🧠 Model Details**
- **Architecture:** Convolutional Neural Network (CNN)  
- **Dataset:** Trained on the **EMNIST Letters dataset** (28x28 grayscale images)  
- **Number of Classes:** **26 uppercase letters (A-Z)**  

## **📊 Model Performance**

### **✅ Test Accuracy & Loss**
- **Test Accuracy:** **94.57%**  
- **Test Loss:** **0.1703**  

### **📜 Classification Report**
```
              precision    recall  f1-score   support

           A       0.93      0.96      0.95       800
           B       0.98      0.97      0.98       800
           C       0.97      0.96      0.97       800
           D       0.97      0.96      0.96       800
           E       0.97      0.98      0.97       800
           F       0.98      0.97      0.98       800
           G       0.91      0.81      0.86       800
           H       0.95      0.96      0.96       800
           I       0.78      0.73      0.76       800
           J       0.97      0.95      0.96       800
           K       0.98      0.97      0.98       800
           L       0.75      0.79      0.77       800
           M       0.99      1.00      0.99       800
           N       0.96      0.97      0.96       800
           O       0.96      0.98      0.97       800
           P       0.98      0.99      0.99       800
           Q       0.85      0.90      0.88       800
           R       0.97      0.96      0.97       800
           S       0.98      0.98      0.98       800
           T       0.97      0.98      0.97       800
           U       0.95      0.94      0.95       800
           V       0.93      0.94      0.93       800
           W       1.00      0.98      0.99       800
           X       0.97      0.98      0.98       800
           Y       0.95      0.96      0.96       800
           Z       0.99      1.00      0.99       800

    accuracy                           0.95     20800
   macro avg       0.95      0.95      0.95     20800
weighted avg       0.95      0.95      0.95     20800
```

### **📈 Training Performance Graphs**
Below are the training accuracy and loss plots of the model:

![Loss and Accuracy Plots](models/loss_accuracy_graph.png)

## **🛠 Model Configuration**
```yaml
model:
  type: "CNN"
  layers:
    - Conv2D: {filters: 64, kernel_size: [3,3], activation: "LeakyReLU", input_shape: [28, 28, 1], kernel_initializer: "he_normal"}
    - MaxPooling2D: {pool_size: [2,2]}
    - Conv2D: {filters: 128, kernel_size: [3,3], activation: "LeakyReLU", kernel_initializer: "he_normal"}
    - MaxPooling2D: {pool_size: [2,2]}
    - Conv2D: {filters: 256, kernel_size: [3,3], activation: "LeakyReLU", kernel_initializer: "he_normal"}
    - Flatten: {}
    - Dense: {units: 512, activation: "LeakyReLU", kernel_initializer: "he_normal"}
    - Dropout: {rate: 0.4}  # Increased to prevent overfitting
    - Dense: {units: 26, activation: "softmax"}
training:
  optimizer: "adam"
  loss_function: "categorical_crossentropy"
  batch_size: 64
  epochs: 30
  learning_rate_schedule: "ReduceLROnPlateau (patience=1, factor=0.5, min_lr=1e-5)"
dataset:
  name: "EMNIST Letters"
  input_shape: [28, 28, 1]
  classes: 26
  preprocessing:
    - "Rotate -90° counterclockwise"
    - "Flip horizontally"
    - "Normalize pixel values to range [0,1]"
  augmentation:
    - "RandomAffine: degrees=6, translate=(0.05, 0.05), scale=(0.98, 1.02)"
    - "RandomApply: ElasticTransform(alpha=2.0, p=0.2)"
    - "RandomApply: GaussianBlur(kernel_size=3, p=0.05)"
```

## **🔍 Challenges Faced**
During development, I faced multiple challenges and worked through them systematically:

### **Incorrect Letter Orientation**
❌ **Problem:** The EMNIST dataset stores images **rotated 90° clockwise and mirrored**  
✅ **Solution:** Applied **-90° counterclockwise rotation and horizontal flipping** to correct alignment.  

### **Real-World Testing Was Inaccurate**
❌ **Problem:** The model performed well on EMNIST but failed with real drawn letters.  
✅ **Solution:** Improved **image preprocessing** with OpenCV to better match the EMNIST format and introduced **slight augmentations** to the training set.  

### **J ↔ L and O ↔ Q Misclassification**
❌ **Problem:** The model confused simillar letters such as "J" and "L" due to dataset quirks.  
✅ **Solution:** Verified EMNIST mappings, adjusted the training process, and fine-tuned the preprocessing.  

### **Optimizing Model Generalization**
❌ **Problem:** Severe overfitting occurred in the early training stages.  
✅ **Solution:** Used **dropout layers**, **batch normalization**, and **learning rate decay** to improve performance.

### **Overfitting Prevention**
❌ **Problem:** The model began overfitting after ~9 epochs.
✅ **Solution:** Adjusted dropout rate (0.4), ReduceLROnPlateau and EarlyStopping triggers, and slight batch size tuning to improve generalization.

## **Planned Future Improvements**
- ✅ Data augmentation to improve generalization
- ✅ Model training script for custom datasets
- ✅ Web-based interface with Flask
- ✅ Deployment as a web app