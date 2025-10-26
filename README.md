# Signature Forgery Detection System

A machine learning-based system for detecting forged signatures using image processing and deep learning techniques. This project helps in verifying the authenticity of handwritten signatures by analyzing key geometric and statistical features.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Results](#results)
- [Future Improvements](#future-improvements)

## 🔍 Overview

This project implements an automated signature verification system that can distinguish between genuine and forged signatures. The system uses:

- **Image Processing**: Preprocessing signatures to extract meaningful features
- **Feature Extraction**: Calculating 9 geometric and statistical features from signatures
- **Deep Learning**: Using a Multi-Layer Perceptron (MLP) neural network for classification
- **TensorFlow**: Implementing the neural network model with TensorFlow 1.x

### Key Capabilities

- ✓ Preprocesses signature images (RGB → Grayscale → Binary)
- ✓ Extracts 9 distinctive features from each signature
- ✓ Trains neural network models for each person's signature
- ✓ Classifies signatures as genuine or forged
- ✓ Handles individual signature verification

## ✨ Features

The system extracts the following features from signature images:

1. **Ratio** - Ratio of signature pixels to total image area
2. **Centroid Y** - Vertical position of signature center
3. **Centroid X** - Horizontal position of signature center
4. **Eccentricity** - Measure of how elongated the signature is
5. **Solidity** - Ratio of signature pixels to convex hull pixels
6. **Skew X** - Horizontal skewness of pixel distribution
7. **Skew Y** - Vertical skewness of pixel distribution
8. **Kurtosis X** - Peak flatness of horizontal pixel distribution
9. **Kurtosis Y** - Peak flatness of vertical pixel distribution

## 📁 Dataset

**Note**: The original signature images are not included in this repository but are available via Google Drive.

**Drive Link**: https://drive.google.com/drive/folders/1KcAvPwbwMEPS6yembqIJgoZG8Tm7m8ya?usp=sharing

### Dataset Structure

- **39 individuals** (Person IDs: 001-039)
- **For each person**:
  - 10 genuine signatures (training: 7, testing: 3)
  - 10 forged signatures (training: 7, testing: 3)
- **Total samples**: 780 signatures (390 genuine + 390 forged)

### Processed Features

The project includes pre-generated feature files:
- `Training/` - Training CSV files for each person (training_001.csv to training_039.csv)
- `Testing/` - Testing CSV files for each person (testing_001.csv to testing_039.csv)
- Each CSV contains 14 samples (7 genuine + 7 forged) with 9 features plus classification label

## 🚀 Installation

### Prerequisites

- Python 3.7+
- TensorFlow 1.x (or TensorFlow 2.x with compatibility mode)
- NumPy
- Pandas
- Matplotlib
- SciPy
- scikit-image

### Dependencies

Install the required packages:

```bash
pip install numpy pandas matplotlib scipy scikit-image tensorflow keras
```

Or for TensorFlow 2.x compatibility:

```bash
pip install tensorflow==2.x
```

**Note**: The code uses TensorFlow 1.x syntax with `tf.disable_v2_behavior()` for compatibility.

## 📂 Project Structure

```
signature_forgery_detection/
│
├── Code_sign.py              # Main Python implementation
├── Main_Code.ipynb           # Jupyter notebook version (Google Colab)
├── README.md                 # This file
│
├── Training/                 # Training feature files
│   ├── training_001.csv
│   ├── training_002.csv
│   └── ... (training_039.csv)
│
├── Testing/                  # Testing feature files
│   ├── testing_001.csv
│   ├── testing_002.csv
│   └── ... (testing_039.csv)
│
├── TestFeatures/            # Test feature extraction
│   └── testcsv.csv
│
└── TestFeatures/            # Temporary test files
```

## 🔧 How It Works

### 1. Image Preprocessing

```python
def preproc(path):
    # Convert RGB to grayscale
    grey = rgbgrey(img)
    
    # Convert to binary using Otsu's threshold
    binimg = greybin(grey)
    
    # Crop to signature boundaries
    signimg = binimg[r.min():r.max(), c.min():c.max()]
```

**Steps**:
- Reads RGB image
- Converts to grayscale
- Applies Gaussian filter (blur_radius=0.8) for noise reduction
- Uses Otsu's threshold for binarization
- Crops to signature bounds

### 2. Feature Extraction

The system extracts 9 features:

- **Ratio**: Pixel density in cropped signature
- **Centroid**: Normalized center coordinates (x, y)
- **Eccentricity & Solidity**: Using scikit-image regionprops
- **Skewness & Kurtosis**: Statistical moments of pixel projections

### 3. Model Architecture

**Neural Network Structure**:
- Input: 9 features
- Hidden Layer 1: 7 neurons (tanh activation)
- Hidden Layer 2: 10 neurons
- Hidden Layer 3: 30 neurons
- Output: 2 classes (genuine/forged)

**Training Parameters**:
- Learning Rate: 0.001
- Epochs: 1000 (or until loss < 0.0001)
- Optimizer: Adam
- Loss Function: Mean Squared Difference
- Activation: Softmax for output

### 4. Classification

The model outputs a probability distribution over 2 classes:
- Class 0: Forged signature
- Class 1: Genuine signature

## 💻 Usage

### Option 1: Using Python Script (`Code_sign.py`)

**Important**: Update the file paths in `Code_sign.py` before running:

```python
genuine_image_paths = "path/to/genuine/signatures"
forged_image_paths = "path/to/forged/signatures"
```

1. Generate features from images:
```bash
python Code_sign.py
```

2. The script will:
   - Extract features from all training/testing images
   - Create CSV files in Training/ and Testing/ folders
   - Prompt for person ID and test image path
   - Classify the signature

### Option 2: Using Jupyter Notebook (`Main_Code.ipynb`)

1. Open in Google Colab
2. Mount Google Drive
3. Update paths for your signature images
4. Run cells sequentially

### Interactive Testing

When running the script, you'll be prompted:

```
Enter person's id : 001
Enter path of signature image : path/to/signature.png
```

**Output**:
- "Genuine Image" - Signature is authentic
- "Forged Image" - Signature is forged

## 🔬 Technical Details

### Image Processing Pipeline

```python
RGB Image → Grayscale → Gaussian Filter → Binary (Otsu) → Crop → Features
```

### Feature Extraction Functions

- `rgbgrey()`: Manual RGB to grayscale conversion
- `greybin()`: Binarization with noise removal
- `Ratio()`: Signature pixel density
- `Centroid()`: Center of mass (normalized)
- `EccentricitySolidity()`: Shape metrics
- `SkewKurtosis()`: Statistical distributions

### Neural Network

**Architecture**: 4-layer MLP
- Layer 1: Linear → tanh (feature transformation)
- Layer 2: Linear
- Layer 3: Linear (deep representation)
- Output: Linear → tanh → softmax

**Key Functions**:
- `multilayer_perceptron()`: Network definition
- `readCSV()`: Data loading and preprocessing
- `evaluate()`: Training and testing
- `trainAndTest()`: Cross-validation

## 📊 Results

### Model Performance

The system achieves different accuracy levels based on:
- Person-specific signatures
- Quality of input images
- Feature extraction quality

### Typical Performance

- Training Accuracy: ~95-98%
- Testing Accuracy: ~85-92%
- Varies by signature complexity

### Factors Affecting Performance

1. **Image Quality**: Higher resolution = better features
2. **Signature Complexity**: More distinctive signatures = better detection
3. **Forgery Skill**: Skilled forgeries are harder to detect
4. **Model Parameters**: Learning rate, epochs, network architecture

## 🎯 Future Improvements

### Potential Enhancements

1. **Deep Learning Models**
   - Implement CNNs for raw image analysis
   - Use Siamese networks for signature comparison
   - Transfer learning from pre-trained models

2. **Feature Engineering**
   - Add texture features (LBP, Gabor filters)
   - Incorporate stroke-level analysis
   - Dynamic time warping for temporal features

3. **Data Augmentation**
   - Rotation, scaling, noise addition
   - Synthetic forgery generation
   - Balanced dataset creation

4. **User Interface**
   - Web-based upload and verification
   - Real-time visualization of features
   - Batch processing capabilities

5. **Model Improvements**
   - Hyperparameter tuning
   - Ensemble methods
   - Attention mechanisms
   - Regularization techniques

## 🐛 Known Issues

1. **TensorFlow Version**: Code uses TensorFlow 1.x syntax
2. **Hard-coded Paths**: File paths need to be updated
3. **Dataset Dependency**: Original images not in repository
4. **Limited to 39 Persons**: Expand dataset for production

## 📝 Code Structure Summary

### Main Functions

| Function | Purpose |
|----------|---------|
| `rgbgrey()` | RGB to grayscale conversion |
| `greybin()` | Grayscale to binary with noise removal |
| `preproc()` | Complete preprocessing pipeline |
| `Ratio()` | Extract signature pixel ratio |
| `Centroid()` | Calculate centroid coordinates |
| `EccentricitySolidity()` | Extract shape features |
| `SkewKurtosis()` | Calculate statistical features |
| `getFeatures()` | Extract all features |
| `makeCSV()` | Generate feature CSV files |
| `testing()` | Extract features for test image |
| `readCSV()` | Load training/testing data |
| `multilayer_perceptron()` | Define neural network |
| `evaluate()` | Train and evaluate model |
| `trainAndTest()` | Cross-validation testing |

## ⚠️ Important Notes

1. **Dataset Required**: Download signature images from the provided Google Drive link before running feature extraction
2. **Update Paths**: Modify file paths in the code to match your system
3. **TensorFlow Version**: Ensure compatibility with TensorFlow 1.x or use compatibility mode
4. **Test Features**: The system works best with clean, high-contrast signature images
5. **Individual Models**: Each person requires a separate trained model for best accuracy

