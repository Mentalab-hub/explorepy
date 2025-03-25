import pylsl
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import os
import pickle
import pandas as pd
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class EEGTransformer(nn.Module):
    def __init__(
        self, input_dim, num_classes=3, d_model=64, nhead=8, num_layers=4,
        dropout=0.2
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)

        self.pos_embedding = nn.Parameter(torch.randn(1, 1, d_model))

        self.channel_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, 
            batch_first=True
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters for better convergence"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, x):
        # x shape: [batch_size, feature_dim]
        # Reshape to [batch_size, 1, feature_dim] to treat as sequence of
        # length 1
        x = x.unsqueeze(1)

        # Project to transformer dimension
        x = self.input_projection(x)

        # Add positional embedding
        x = x + self.pos_embedding

        # Apply channel attention
        attn_output, _ = self.channel_attention(x, x, x)
        x = x + attn_output  # Residual connection

        # Pass through transformer
        x = self.transformer_encoder(x)

        # Apply temporal attention
        attn_output, _ = self.temporal_attention(x, x, x)
        x = x + attn_output  # Residual connection

        # Use the output corresponding to the 'sequence'
        x = x.squeeze(1)

        # Classification
        x = self.classifier(x)

        return x


class PyTorchClassifier:
    def __init__(
        self, model, criterion, optimizer, num_epochs=100, batch_size=32, 
        device="cpu"
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.model.to(device)

    def fit(self, X, y):
        # Convert to PyTorch dataset
        dataset = EEGDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, 
                                shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                logging.info(
                    f"Epoch {epoch+1}/{self.num_epochs}, Loss: {running_loss/len(dataloader):.4f}"
                )

        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()


class EnhancedPyTorchClassifier(PyTorchClassifier):
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        num_epochs=100,
        batch_size=32,
        device="cpu",
        patience=10,
    ):
        super().__init__(model, criterion, optimizer, num_epochs, batch_size,
                         device)
        self.patience = patience

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        train_dataset = EEGDataset(X_train, y_train)
        val_dataset = EEGDataset(X_val, y_val)

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        best_val_loss = float("inf")
        no_improve_epochs = 0
        best_model_state = None

        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(train_dataloader)

            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs, labels = inputs.to(self.device),
                    labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss = val_loss / len(val_dataloader)
            val_accuracy = correct / total

            scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0:
                logging.info(
                    f"Epoch {epoch+1}/{self.num_epochs}, "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_accuracy:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_epochs = 0
                best_model_state = self.model.state_dict().copy()
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= self.patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self


class EEGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MotorImageryBCI:
    def __init__(self, stream_name="Explore_AAAG_ExG", scaling_factor=100000):
        """Initialize the BCI system with neutral state support

        Args:
            stream_name (str): Name of the LSL stream to connect to
            scaling_factor (float): Factor to scale raw EEG values
        """
        self.stream_name = stream_name
        self.scaling_factor = scaling_factor
        self.fs = None
        self.classifier = None
        self.eeg_channels = None
        self.mu_band = (8, 12)
        self.beta_band = (16, 24)
        self.window_duration = 2.0
        self.overlap = 0.5
        self.trained_classifiers = {}
        self.num_classes = 3

        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        logging.info(f"Using device: {self.device}")

        self.classifier_options = {
            "lda": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", LinearDiscriminantAnalysis()),
                ]
            ),
            "svm": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", SVC(kernel="linear", probability=True)),
                ]
            ),
            "rf": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", RandomForestClassifier(n_estimators=100)),
                ]
            ),
        }

    def add_transformer_classifier(self, X_train):
        """Add enhanced transformer-based classifier to the options"""
        input_dim = X_train.shape[1]
        model = EEGTransformer(input_dim=input_dim,
                               num_classes=self.num_classes)
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(model.parameters(), lr=0.001,
                                weight_decay=0.01)

        self.classifier_options["transformer"] = EnhancedPyTorchClassifier(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=100,
            batch_size=16,
            device=self.device,
            patience=15,
        )

    def connect_to_stream(self):
        """Connect to the LSL stream"""
        logging.info(f"Looking for {self.stream_name} stream...")
        streams = pylsl.resolve_streams()

        target_stream = None
        for stream in streams:
            if stream.name() == self.stream_name:
                target_stream = stream
                break

        if target_stream is None:
            logging.error(f"{self.stream_name} stream not found!")
            if streams:
                logging.info("Available streams:")
                for i, s in enumerate(streams):
                    logging.info(
                        f"  {i+1}. {s.name()} ({s.type()}) - {s.channel_count()} channels"
                    )
            return None

        inlet = pylsl.StreamInlet(target_stream)
        self.fs = target_stream.nominal_srate()
        logging.info(f"Connected to {self.stream_name} stream at {self.fs} Hz")

        return inlet

    def collect_calibration_data(self, cue_duration=4, rest_duration=2,
                                 n_trials=1):
        """Collect calibration data including neutral state

        Args:
            cue_duration (float): Duration of each imagery cue in seconds
            rest_duration (float): Duration of rest between trials in seconds
            n_trials (int): Number of trials for each condition

        Returns:
            tuple: Left hand data, right hand data, neutral state data,
            sampling frequency
        """
        inlet = self.connect_to_stream()
        if inlet is None:
            return None, None, None, None

        left_hand_data = []
        right_hand_data = []
        neutral_data = []

        logging.info("\n=== MOTOR IMAGERY CALIBRATION ===")
        logging.info(
            "You will be prompted to imagine moving your left or right hand,"
            )
        logging.info("or to remain in a neutral relaxed state.")
        logging.info(
            "Don't actually move - just vividly imagine the movement when prompted."
        )
        time.sleep(3)

        for trial in range(n_trials):
            logging.info(f"\nTrial {trial+1}/{n_trials}")

            logging.info(
                "NEUTRAL STATE - Stay relaxed, don't imagine any movement"
                )
            start_time = time.time()
            trial_data = []

            while time.time() - start_time < cue_duration:
                sample, _ = inlet.pull_sample(timeout=0.0)
                if sample:
                    # Apply correct scaling
                    # sample = [val/self.scaling_factor for val in sample]
                    trial_data.append(sample)

            if trial_data:
                neutral_data.append(np.array(trial_data))

            logging.info("REST - Preparing for next task...")
            time.sleep(rest_duration)

            logging.info(
                "IMAGINE LEFT HAND MOVEMENT - Making a fist and releasing"
                )
            start_time = time.time()
            trial_data = []

            while time.time() - start_time < cue_duration:
                sample, _ = inlet.pull_sample(timeout=0.0)
                if sample:
                    # sample = [val/self.scaling_factor for val in sample]
                    trial_data.append(sample)

            if trial_data:
                left_hand_data.append(np.array(trial_data))

            logging.info("REST - Preparing for next task...")
            time.sleep(rest_duration)

            logging.info(
                "IMAGINE RIGHT HAND MOVEMENT - Making a fist and releasing"
                )
            start_time = time.time()
            trial_data = []

            while time.time() - start_time < cue_duration:
                sample, _ = inlet.pull_sample(timeout=0.0)
                if sample:
                    # sample = [val/self.scaling_factor for val in sample]
                    trial_data.append(sample)

            if trial_data:
                right_hand_data.append(np.array(trial_data))

            logging.info("REST - Trial complete")
            time.sleep(rest_duration)

        return left_hand_data, right_hand_data, neutral_data, self.fs

    def preprocess_data(self, data, band):
        """Preprocess data with bandpass filter

        Args:
            data (ndarray): Raw EEG data
            band (tuple): Frequency band (low, high)

        Returns:
            ndarray: Filtered data
        """
        if self.fs is None or self.fs <= 0:
            logging.warning(
                "Warning: Invalid sampling frequency. Using default of 250 Hz."
            )
            self.fs = 250.0

        nyquist = self.fs / 2
        if band[0] <= 0 or band[1] >= nyquist:
            logging.warning(
                f"Warning: Invalid frequency band {band} for fs={self.fs}. Adjusting..."
            )
            band = (max(0.1, band[0]), min(nyquist - 0.1, band[1]))

        low = band[0] / nyquist
        high = band[1] / nyquist

        b, a = signal.butter(4, [low, high], "bandpass")
        filtered = signal.filtfilt(b, a, data, axis=0)

        return filtered

    def extract_features(self, mu_data, beta_data, window_size=1.0):
        """Extract features for motor imagery

        Args:
            mu_data (ndarray): Mu-band filtered data
            beta_data (ndarray): Beta-band filtered data
            window_size (float): Window size in seconds

        Returns:
            ndarray: Feature vectors
        """
        window_samples = int(window_size * self.fs)
        n_windows = max(1, mu_data.shape[0] // window_samples)

        features = []

        for win in range(n_windows):
            start_idx = win * window_samples
            end_idx = start_idx + window_samples

            if end_idx > mu_data.shape[0]:
                break

            # band power
            mu_power = np.mean(mu_data[start_idx:end_idx, :] ** 2, axis=0)
            beta_power = np.mean(beta_data[start_idx:end_idx, :] ** 2, axis=0)

            window_features = np.concatenate([mu_power, beta_power])
            features.append(window_features)

        return np.array(features)

    def prepare_dataset(
        self, left_hand_data, right_hand_data, neutral_data, save_dir="dataset"
    ):
        """Prepare dataset with three classes for classifier training and save
        it to disk in CSV format.
        Args:
            left_hand_data (list): List of arrays containing
            left hand trial datset
            right_hand_data (list): List of arrays containing
            right hand trial dataset
            neutral_data (list): List of arrays containing
            neutral state trial dataset
            save_dir (str): Directory to save the dataset.
            Default is "dataset".

        Returns:
            tuple: X (features), y (labels)
        """
        self.eeg_channels = [1, 2, 3, 4, 5, 6, 7, 8]
        left_features = []
        right_features = []
        neutral_features = []

        # Process left hand data
        for trial_data in left_hand_data:
            # eeg_data = trial_data[:, self.eeg_channels]
            eeg_data = trial_data
            mu_filtered = self.preprocess_data(eeg_data, self.mu_band)
            beta_filtered = self.preprocess_data(eeg_data, self.beta_band)
            features = self.extract_features(mu_filtered, beta_filtered)
            if len(features) > 0:
                left_features.extend(features)

        # Process right hand data
        for trial_data in right_hand_data:
            # eeg_data = trial_data[:, self.eeg_channels]
            eeg_data = trial_data
            mu_filtered = self.preprocess_data(eeg_data, self.mu_band)
            beta_filtered = self.preprocess_data(eeg_data, self.beta_band)
            features = self.extract_features(mu_filtered, beta_filtered)
            if len(features) > 0:
                right_features.extend(features)

        # Process neutral state data
        for trial_data in neutral_data:
            # eeg_data = trial_data[:, self.eeg_channels]
            eeg_data = trial_data
            mu_filtered = self.preprocess_data(eeg_data, self.mu_band)
            beta_filtered = self.preprocess_data(eeg_data, self.beta_band)
            features = self.extract_features(mu_filtered, beta_filtered)
            if len(features) > 0:
                neutral_features.extend(features)

        # Create feature vectors and labels
        # Labels: 0 = left, 1 = right, 2 = neutral
        X = np.vstack([left_features, right_features, neutral_features])
        y = np.hstack(
            [
                np.zeros(len(left_features)),  # Left hand: 0
                np.ones(len(right_features)),  # Right hand: 1
                np.full(len(neutral_features), 2),  # Neutral: 2
            ]
        )

        os.makedirs(save_dir, exist_ok=True)
        feature_columns = [f"feature_{i}" for i in range(X.shape[1])]
        df_features = pd.DataFrame(X, columns=feature_columns)
        df_labels = pd.DataFrame(y, columns=["label"])

        df_dataset = pd.concat([df_features, df_labels], axis=1)

        csv_path = os.path.join(save_dir, "dataset.csv")
        df_dataset.to_csv(csv_path, index=False)

        logging.info(f"Dataset saved to {csv_path}")

        return X, y

    def train_and_evaluate_classifiers(self, X, y, test_size=0.3):
        """Train and evaluate multiple classifiers

        Args:
            X (ndarray): Feature matrix
            y (ndarray): Labels
            test_size (float): Proportion of data to use for testing

        Returns:
            dict: Dictionary of trained classifiers
            dict: Performance metrics for each classifier
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        self.add_transformer_classifier(X_train)

        results = {}

        for name, clf in self.classifier_options.items():
            logging.info(f"\nTraining {name} classifier...")
            start_time = time.time()

            clf.fit(X_train, y_train)

            self.trained_classifiers[name] = clf

            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            training_time = time.time() - start_time

            results[name] = {
                "accuracy": accuracy,
                "confusion_matrix": conf_matrix,
                "training_time": training_time,
            }

            logging.info(f"{name} Accuracy: {accuracy:.4f}")
            logging.info(f"Training time: {training_time:.2f} seconds")

        self.save_classifiers()

        return results

    def compare_classifiers(self, results):
        """Compare different classifiers with visualizations

        Args:
            results (dict): Performance metrics for each classifier
        """
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        accuracies = [results[name]["accuracy"] for name in results]
        axs[0].bar(results.keys(), accuracies)
        axs[0].set_title("Classifier Accuracy Comparison")
        axs[0].set_ylim(0, 1)
        axs[0].set_ylabel("Accuracy")
        axs[0].set_xticklabels(results.keys(), rotation=45)

        train_times = [results[name]["training_time"] for name in results]
        axs[1].bar(results.keys(), train_times)
        axs[1].set_title("Training Time Comparison")
        axs[1].set_ylabel("Time (seconds)")
        axs[1].set_xticklabels(results.keys(), rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, "classifier_comparison.png"))
        plt.show()

        plt.figure(figsize=(15, 4))
        for i, name in enumerate(results.keys()):
            plt.subplot(1, len(results), i + 1)
            sns.heatmap(
                results[name]["confusion_matrix"],
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Left", "Right", "Neutral"],
                yticklabels=["Left", "Right", "Neutral"],
            )
            plt.title(f"{name} Confusion Matrix")

        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, "confusion_matrices.png"))
        plt.show()

        results_df = pd.DataFrame(
            {
                "Classifier": list(results.keys()),
                "Accuracy": [results[name]["accuracy"] for name in results],
                "Training_Time": [results[name]["training_time"]
                                  for name in results],
            }
        )
        results_df.to_csv(
            os.path.join(self.model_dir, "classifier_results.csv"), index=False
        )

        logging.info("\n=== CLASSIFIER COMPARISON SUMMARY ===")
        for name in results:
            logging.info(
                f"{name}: Accuracy={results[name]['accuracy']:.4f}, Time={results[name]['training_time']:.2f}s"
            )

    def save_classifiers(self):
        """Save trained classifiers to disk"""
        for name, clf in self.trained_classifiers.items():
            filename = os.path.join(self.model_dir, f"{name}_classifier.pkl")
            with open(filename, "wb") as f:
                pickle.dump(clf, f)

        logging.info(f"Classifiers saved to {self.model_dir}")

    def load_classifier(self, name):
        """Load a trained classifier from disk"""
        filename = os.path.join(self.model_dir, f"{name}_classifier.pkl")
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                self.classifier = pickle.load(f)
            logging.info(f"Loaded {name} classifier from {filename}")
            return True
        else:
            logging.error(f"Classifier file {filename} not found")
            return False

    def run_bci(self, classifier_name="best"):
        """Run the BCI in continuous mode with neutral state detection

        Args:
            classifier_name (str): Name of classifier to use 
            ('best' uses highest accuracy)
        """
        if classifier_name == "best" and hasattr(self, "classifier_results"):
            accuracies = {
                name: self.classifier_results[name]["accuracy"]
                for name in self.classifier_results
            }
            classifier_name = max(accuracies, key=accuracies.get)
            logging.info(f"Using best classifier: {classifier_name}")

        if classifier_name in self.trained_classifiers:
            self.classifier = self.trained_classifiers[classifier_name]
        else:
            logging.error(
                f"Classifier '{classifier_name}' not found. Please train first."
            )
            return

        inlet = self.connect_to_stream()
        if inlet is None:
            return

        logging.info("\nBCI ACTIVE - Continuously classifying brain activity")
        logging.info("Press Ctrl+C to exit")
        logging.info(
            "\nTimestamp | Detected State | Certainty | Control Status")
        logging.info("-" * 60)

        try:
            data_buffer = []
            last_update_time = time.time()

            while True:
                sample, timestamp = inlet.pull_sample(timeout=0.0)
                if sample:
                    # sample = [val/self.scaling_factor for val in sample]
                    data_buffer.append(sample)

                buffer_duration = len(data_buffer) / self.fs if self.fs > 0 else 0
                current_time = time.time()

                if (
                    buffer_duration >= self.window_duration
                    and (current_time - last_update_time) >= self.overlap
                ):

                    data_array = np.array(data_buffer)

                    # eeg_data = data_array[:, self.eeg_channels]
                    eeg_data = data_array

                    mu_filtered = self.preprocess_data(eeg_data, self.mu_band)
                    beta_filtered = self.preprocess_data(eeg_data,
                                                         self.beta_band)

                    features = self.extract_features(mu_filtered,
                                                     beta_filtered)

                    if len(features) > 0:
                        prediction = self.classifier.predict_proba(features)

                        avg_prob = np.mean(prediction, axis=0)

                        predicted_class = np.argmax(avg_prob)

                        if predicted_class == 0:
                            state = "LEFT HAND"
                        elif predicted_class == 1:
                            state = "RIGHT HAND"
                        else:  # predicted_class == 2
                            state = "NEUTRAL"

                        certainty = avg_prob[predicted_class]

                        threshold = 0.6  # Adjust as needed
                        if certainty > threshold and state != "NEUTRAL":
                            control_active = True
                        else:
                            control_active = False
                        logging.info(
                            f"{timestamp} | {state:10} | {certainty:.2f} | {'ACTIVE' if control_active else 'INACTIVE'}"
                        )

                    overlap_samples = int(self.overlap * self.fs)
                    if len(data_buffer) > overlap_samples:
                        data_buffer = data_buffer[-overlap_samples:]

                    last_update_time = current_time

                time.sleep(0.005)

        except KeyboardInterrupt:
            logging.info("\nBCI stopped by user")

    def train_all_classifiers(self):
        """Train and compare all classifiers with three classes"""
        logging.info("=== STARTING COMPREHENSIVE CLASSIFIER TRAINING ===")

        left_data, right_data, neutral_data, _ = self.collect_calibration_data()
        if left_data is None or right_data is None or neutral_data is None:
            logging.error("Calibration failed!")
            return False

        X, y = self.prepare_dataset(left_data, right_data, neutral_data)

        self.classifier_results = self.train_and_evaluate_classifiers(X, y)

        self.compare_classifiers(self.classifier_results)

        return True


def main():
    logging.info("=== MOTOR IMAGERY BCI WITH THREE CLASSES ===")

    bci = MotorImageryBCI()

    logging.info("\nOptions:")
    logging.info("1. Train and compare all classifiers")
    logging.info("2. Run BCI with best classifier")
    logging.info("3. Run BCI with specific classifier")
    choice = input("\nEnter your choice (1-3): ")
    if choice == "1":
        bci.train_all_classifiers()
        run_bci = input(
            "\nDo you want to run the BCI with the best classifier? (y/n): "
        ).lower()
        if run_bci == "y":
            bci.run_bci("best")
    elif choice == "2":
        if os.path.exists(os.path.join(bci.model_dir, "classifier_results.csv")):
            results_df = pd.read_csv(
                os.path.join(bci.model_dir, "classifier_results.csv")
            )
            best_classifier = results_df.loc[
                results_df["Accuracy"].idxmax(), "Classifier"
            ]
            if bci.load_classifier(best_classifier):
                bci.run_bci(best_classifier)
            else:
                logging.error(
                    "Could not load best classifier, training new ones...")
                bci.train_all_classifiers()
                bci.run_bci("best")
        else:
            logging.error(
                "No pre-trained classifiers found, training new ones...")
            bci.train_all_classifiers()
            bci.run_bci("best")
    elif choice == "3":
        classifier_files = [
            f for f in os.listdir(bci.model_dir)
            if f.endswith("_classifier.pkl")
        ]
        if classifier_files:
            available_classifiers = [
                f.replace("_classifier.pkl", "") for f in classifier_files
            ]
            logging.info("\nAvailable classifiers:")
            for i, clf in enumerate(available_classifiers):
                logging.info(f"{i+1}. {clf}")
            idx = int(input("\nSelect classifier (number): ")) - 1
            if 0 <= idx < len(available_classifiers):
                clf_name = available_classifiers[idx]
                if bci.load_classifier(clf_name):
                    bci.run_bci(clf_name)
            else:
                logging.error("Invalid selection")
        else:
            logging.error(
                "No pre-trained classifiers found, training new ones...")
            bci.train_all_classifiers()
            bci.run_bci("best")
    else:
        logging.error("Invalid choice")


if __name__ == "__main__":
    main()
