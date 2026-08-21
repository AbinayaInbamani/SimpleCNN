Simple CNN for Tomato Disease Classification

This project demonstrates how a Convolutional Neural Network (CNN) can be built from scratch using PyTorch to classify tomato leaf diseases.
The notebook is designed for students who are beginning to learn deep learning and computer vision. It demonstrates the fundamental CNN workflow:
Convolution → ReLU → Pooling → Flatten → Classification
The code is designed to run in Google Colab

Learning Objectives

By completing this notebook, students will learn how to:

* Load an image dataset
* Select specific classes from a dataset
* Split images into training and testing datasets
* Resize images and convert them into tensors
* Create a custom PyTorch Dataset
* Use DataLoaders
* Build a simple CNN
* Understand convolution, ReLU, and max pooling
* Train a CNN
* Evaluate a trained model
* Compare actual and predicted classes
* Save a trained model
* Reload a saved model for future use

 Dataset

The project uses the PlantVillage Dataset
The complete dataset contains:
54,305 images, 38 plant disease/healthy classes. For this activity, only tomato images are selected.

Tomato Dataset

Total tomato images:18,160

The notebook automatically identifies 10 tomato classes:

1. Bacterial Spot
2. Early Blight
3. Late Blight
4. Leaf Mold
5. Septoria Leaf Spot
6. Two-Spotted Spider Mite
7. Target Spot
8. Tomato Yellow Leaf Curl Virus
9. Tomato Mosaic Virus
10. Healthy


Dataset Split

The tomato images are divided into:

| Dataset  | Number of Images |
| -------- | ---------------: |
| Training |           14,528 |
| Testing  |            3,632 |
| Total    |           18,160 |

The code uses:

80% Training → 20% Testing

A stratified split is used so that the proportion of each tomato class is maintained in both datasets.

# Image Preprocessing

Each image is:

1. Opened as an RGB image
2. Resized to 64 × 64 pixels
3. Converted into a PyTorch tensor
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
```

# CNN Architecture

The CNN contains two convolutional layers.


Input Tomato Leaf Image
64 × 64 × 3
        ↓
3 × 3 Convolution
3 → 8 feature maps
        ↓
ReLU
        ↓
2 × 2 Max Pooling
        ↓
32 × 32 × 8
        ↓
3 × 3 Convolution
8 → 16 feature maps
        ↓
ReLU
        ↓
2 × 2 Max Pooling
        ↓
16 × 16 × 16
        ↓
Flatten
        ↓
4096 Features
        ↓
Fully Connected Layer
        ↓
10 Tomato Classes
```



# What Happens Inside the CNN?

## 1. Convolution

The convolutional layers scan the image using small filters.

```python
nn.Conv2d(
    3,
    8,
    kernel_size=3,
    padding=1
)
```

The first convolution converts:

**3 RGB channels → 8 feature maps**

The CNN begins learning visual patterns such as:

* Edges
* Colors
* Textures
* Spots
* Leaf patterns



## 2. ReLU

ReLU introduces non-linearity into the CNN.

```python
torch.relu()
```

It allows the network to learn more complicated visual patterns.



## 3. Max Pooling

Max pooling reduces the spatial dimensions of the feature maps.

```python
nn.MaxPool2d(2, 2)
```

For example:

```text
64 × 64
   ↓
32 × 32
   ↓
16 × 16
```

This reduces computation while retaining important features.

---

## 4. Flatten

After convolution and pooling, the feature maps have the size:

```text
16 × 16 × 16
```

They are converted into:

```text
4096 features
```

using:

```python
nn.Flatten()
```



## 5. Classification

The flattened features are passed to a fully connected layer.

```python
nn.Linear(
    16 * 16 * 16,
    num_classes
)
```

The output contains scores for the **10 tomato classes**.

The class with the highest score becomes the predicted class.



# Training

The model uses:

### Loss Function

```python
nn.CrossEntropyLoss()
```

Cross-entropy loss measures how different the predicted class is from the correct class.

### Optimizer

```python
optim.Adam(
    model.parameters(),
    lr=0.001
)
```

Optimizer:

**Adam**

Learning rate:

**0.001**

### Batch Size

```text
64 images
```

### Number of Epochs

```text
3 epochs
```



# Training Results

The example notebook run produced:

| Epoch |   Loss | Training Accuracy |
| ----- | -----: | ----------------: |
| 1     | 1.3126 |            56.78% |
| 2     | 0.7700 |            75.19% |
| 3     | 0.5993 |            80.53% |

The improvement in training accuracy shows that the CNN is learning patterns from the tomato leaf images.



# Test Result

After training, the model is evaluated using images that were not used during training.

The example run achieved:

**Test Accuracy: 80.45%**

Results may vary slightly between runs.



# Visualizing Predictions

The notebook displays test images together with:

```text
Actual: Early Blight
Predicted: Early Blight
```

This allows students to visually examine where the CNN makes correct and incorrect predictions.

Twelve sample test images are displayed at a time.



# Saving the Model

After training, the learned model parameters are saved using:

```python
torch.save(
    model.state_dict(),
    "tomato_simple_cnn.pth"
)
```

The saved model is:

```text
tomato_simple_cnn.pth
```



# Loading the Saved Model

The model can later be restored without training it again.

```python
model = SimpleCNN(
    num_classes=len(tomato_classes)
)

model.load_state_dict(
    torch.load(
        "tomato_simple_cnn.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()
```

This demonstrates an important machine-learning workflow:

```text
Train
  ↓
Save Model
  ↓
Close Program
  ↓
Load Model
  ↓
Use Model for Prediction
```



# Complete Workflow

```text
PlantVillage Dataset
        ↓
Select Tomato Images
        ↓
Assign Class Labels
        ↓
80% Training / 20% Testing
        ↓
Resize Images to 64 × 64
        ↓
Convert to Tensors
        ↓
Create DataLoaders
        ↓
Simple CNN
        ↓
Convolution
        ↓
ReLU
        ↓
Max Pooling
        ↓
Convolution
        ↓
ReLU
        ↓
Max Pooling
        ↓
Flatten
        ↓
Fully Connected Layer
        ↓
Disease Classification
        ↓
Train Model
        ↓
Evaluate Test Images
        ↓
Visualize Predictions
        ↓
Save Model
        ↓
Reload Model
```



# Running the Notebook

The easiest way to run the notebook is with **Google Colab**.

1. Open the notebook from GitHub.

2. Select **Open in Colab**.

3. In Google Colab, select:

   **Runtime → Change runtime type → GPU**

4. Run the cells from top to bottom.

The notebook automatically downloads the PlantVillage dataset.



# Technologies Used

* Python
* PyTorch
* Torchvision
* Scikit-learn
* Pillow
* Matplotlib
* Google Colab
* GitHub



# Main Takeaway

This activity demonstrates the basic CNN pipeline:

**Image → Convolution → ReLU → Pooling → Learned Features → Flatten → Classification**

The purpose of this notebook is not to create the most advanced tomato disease classifier. Instead, it provides a simple CNN architecture that allows students to understand **how a convolutional neural network learns from images**.
