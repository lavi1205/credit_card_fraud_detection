import unittest
from unittest.mock import patch
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from src.models.train_model import train_and_evaluate

class TestTrainAndEvaluate(unittest.TestCase):

    @patch('src.models.train_model.logger')
    def test_train_and_evaluate_classification(self, mock_logger):
        # Generate synthetic classification data
        xTrain, yTrain = make_classification(n_samples=100, n_features=20, random_state=42)
        xTest, yTest = make_classification(n_samples=20, n_features=20, random_state=42)

        # Initialize a Decision Tree classifier
        model = DecisionTreeClassifier()

        # Train and evaluate the model
        metrics = train_and_evaluate(model, xTrain, yTrain, xTest, yTest, 'Decision Tree')

        # Check if the metrics are returned correctly
        self.assertIn('Accuracy', metrics)
        self.assertIn('Precision', metrics)
        self.assertIn('Recall', metrics)
        self.assertIn('F1-Score', metrics)
        self.assertIn('MCC', metrics)

    @patch('src.models.train_model.logger')
    def test_train_and_evaluate_regression(self, mock_logger):
        # Generate synthetic regression data
        xTrain, yTrain = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)
        xTest, yTest = make_regression(n_samples=20, n_features=20, noise=0.1, random_state=42)

        # Initialize a Linear Regression model
        model = LinearRegression()

        # Train and evaluate the model
        metrics = train_and_evaluate(model, xTrain, yTrain, xTest, yTest, 'Linear Regression', is_classification=False)

        # Check if the metrics are returned correctly
        self.assertIn('MSE', metrics)
        self.assertIn('MAE', metrics)
        self.assertIn('R2 Score', metrics)

    @patch('src.models.train_model.logger')
    def test_train_and_evaluate_with_transform(self, mock_logger):
        # Generate synthetic classification data
        xTrain, yTrain = make_classification(n_samples=100, n_features=20, random_state=42)
        xTest, yTest = make_classification(n_samples=20, n_features=20, random_state=42)

        # Initialize a Logistic Regression model
        model = LogisticRegression(max_iter=10000)

        # Initialize a StandardScaler
        transform = StandardScaler()

        # Train and evaluate the model with transformation
        metrics = train_and_evaluate(model, xTrain, yTrain, xTest, yTest, 'Logistic Regression', transform=transform)

        # Check if the metrics are returned correctly
        self.assertIn('Accuracy', metrics)
        self.assertIn('Precision', metrics)
        self.assertIn('Recall', metrics)
        self.assertIn('F1-Score', metrics)
        self.assertIn('MCC', metrics)

if __name__ == '__main__':
    unittest.main()
