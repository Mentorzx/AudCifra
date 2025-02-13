import os
import pickle
from typing import Optional

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class ChordClassifier:
    """
    ChordClassifier loads a pre-trained model if available or trains a simple classifier using
    synthetic data generated from chord templates. The classifier uses a k-nearest neighbors algorithm.
    """

    def __init__(self, model_path: Optional[str]):
        """
        Initializes the ChordClassifier. If a valid model_path is provided, the classifier will load the model.
        Otherwise, it trains a default model using synthetic data.

        Parameters:
            model_path (str, optional): The file path to a pre-trained model (pickle file).
        """
        if model_path and os.path.exists(model_path):
            self.model = self.load_model(model_path)
            self.labels = self.model.classes_
        else:
            self.model, self.labels = self.train_default_model()

    def train_default_model(self) -> tuple:
        """
        Trains a default k-nearest neighbors classifier using synthetic data generated from chord templates.
        Synthetic samples are created by adding slight Gaussian noise to each chord template.

        Returns:
            tuple: A tuple containing the trained classifier and a list of chord labels.
        """
        from .chord_templates import get_chord_templates

        templates = get_chord_templates(False, True)
        labels = []
        X = []
        for chord, template in templates.items():
            for _ in range(100):
                noise = np.random.normal(0, 0.05, size=template.shape)
                sample = template + noise
                sample = sample / (np.linalg.norm(sample) + 1e-6)
                X.append(sample)
                labels.append(chord)
        X = np.array(X)
        labels = np.array(labels)
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X, labels)
        return model, model.classes_

    def load_model(self, model_path: str):
        """
        Loads a pre-trained model from a pickle file.

        Parameters:
            model_path (str): The file path to the model pickle file.

        Returns:
            object: The loaded model.
        """
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model

    def predict(self, chroma_vector: np.ndarray) -> tuple:
        """
        Predicts the chord label from a normalized chroma vector using the classifier.

        Parameters:
            chroma_vector (np.ndarray): A normalized chroma vector.

        Returns:
            tuple: A tuple containing the predicted chord label (str) and a confidence score (float).
        """
        features = chroma_vector.reshape(1, -1)
        probabilities = self.model.predict_proba(features)
        max_index = np.argmax(probabilities, axis=1)[0]
        confidence = float(probabilities[0, max_index])
        predicted_label = self.labels[max_index] if self.labels is not None else "N.C."
        return predicted_label, confidence
