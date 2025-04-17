## Motor Imagery BCI Application

### Overview
This application is a Brain-Computer Interface (BCI) system designed to classify motor imagery tasks (left hand, right hand, and neutral state) using EEG data. The system connects to Explore Pro device via the Lab Streaming Layer (LSL) protocol, processes the EEG signals, and classifies the data using machine learning models, including a Transformer-based neural network.

The application supports real-time classification. It includes tools for data collection, preprocessing, model training, and evaluation.

### Features
- **Real-Time EEG Data Acquisition:** Connects to an EEG device via LSL for real-time data streaming.
- **Motor Imagery Classification:** Classifies EEG data into three states: left hand movement, right hand movement, and neutral state.
- **Transformer Model:** Implements a Transformer-based neural network for EEG signal classification.
- **Multiple Classifiers:** Supports Linear Discriminant Analysis (LDA), Support Vector Machines (SVM), Random Forest (RF), and the custom Transformer model.
- **Data Preprocessing:** Includes bandpass filtering (mu and beta bands) and feature extraction (band power).
- **Model Training and Evaluation:** Allows training and evaluation of multiple classifiers with visualization of results.
- **Real-Time BCI Operation:** Runs in continuous mode for real-time classification and control.

### Requirements
#### Hardware
- An Explore Pro device.
- A computer with sufficient processing power (GPU recommended for Transformer model training).

#### Software
- Python 3.10 or higher
- Required Python packages (install via pip):
  
  ```bash
  pip install pylsl numpy scipy scikit-learn torch matplotlib seaborn pandas
  ```

Ensure your EEG device is set up and streaming data via LSL.

### Usage
#### 1. Collect Calibration Data
To train the classifiers, you need to collect calibration data. Run the following command and follow the on-screen instructions:

```bash
python main.py
```

Choose option **1** to train and compare all classifiers. The system will prompt you to perform left hand, right hand, and neutral state tasks.

#### 2. Train Classifiers
The application will automatically preprocess the data, extract features, and train multiple classifiers (LDA, SVM, Random Forest, and Transformer). Training progress and results will be displayed in the console.

#### 3. Run Real-Time BCI
After training, you can run the BCI in real-time mode:

```bash
python main.py
```

Choose option **2** to run the BCI with the best-performing classifier or option **3** to select a specific classifier.

### File Structure
```
motor-imagery-bci/
├── models/                  # Saved classifier models
├── dataset/                 # Saved calibration datasets
├── main.py                  # Main application script
├── README.md                # This file
```

### Classifiers
The application supports the following classifiers:
- **Linear Discriminant Analysis (LDA)**
- **Support Vector Machine (SVM)**
- **Random Forest (RF)**
- **Transformer Model**

### Customizing the Transformer Model
The Transformer model is defined in the `EEGTransformer` class. You can customize the following parameters:

- `input_dim`: Input feature dimension.
- `num_classes`: Number of output classes (default: 3).
- `d_model`: Dimension of the model (default: 64).
- `nhead`: Number of attention heads (default: 8).
- `num_layers`: Number of Transformer encoder layers (default: 4).
- `dropout`: Dropout rate (default: 0.2).

Example:

```python
model = EEGTransformer(input_dim=64, num_classes=3, d_model=128, nhead=8, num_layers=6, dropout=0.3)
```

### Data Preprocessing
The EEG data is preprocessed as follows:
- **Bandpass Filtering:** Applied to extract mu (8-12 Hz) and beta (16-24 Hz) bands.
- **Feature Extraction:** Band power is calculated for each frequency band.
- **Dataset Preparation:** Features and labels are saved in a CSV file for training.

### Real-Time Operation
In real-time mode, the application:
1. Continuously collects EEG data from the LSL stream.
2. Applies preprocessing and feature extraction.
3. Classifies the data using the selected model.
4. Displays the detected state (left hand, right hand, or neutral) and certainty level.

### Troubleshooting
#### 1. LSL Stream Not Found
- Ensure your device is properly connected and streaming data.
- Verify the stream name in the `MotorImageryBCI` class initialization.

#### 2. Poor Classification Accuracy
- Collect more calibration data.
- Adjust the frequency bands (`mu_band` and `beta_band`) in the `MotorImageryBCI` class.
- Fine-tune the Transformer model parameters.

#### 3. Performance Issues
- Use a GPU for training the Transformer model.
- Reduce the window size or overlap in the `MotorImageryBCI` class.

---

## Future Improvements
While the current system is functional, several enhancements can improve its accuracy, real-time performance, and adaptability:

1. **Advanced Deep Learning Models**
   - Replace traditional classifiers (LDA, SVM, RF) with more powerful architectures like **EEGNet, ConvLSTM, or Graph Neural Networks (GNNs)**.
   - Optimize the Transformer model by using **EEG-specialized architectures** such as **TS-Transformer** or **EEGFormer** for better time-series representation.

2. **Enhanced Feature Extraction**
   - Incorporate **Common Spatial Patterns (CSP)** to improve the separation of motor imagery classes.
   - Use **Riemannian geometry-based approaches** to extract more robust spatial features.

3. **Data Augmentation for EEG**
   - Implement **synthetic EEG data generation** techniques like **time-domain warping, frequency shifting, and GAN-based augmentation** to improve model generalization.

4. **Online & Transfer Learning**
   - Introduce **adaptive learning** to fine-tune the model based on user-specific EEG patterns.
   - Use **few-shot learning** or **domain adaptation techniques** to improve performance across multiple users.