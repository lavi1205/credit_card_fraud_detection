import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import os
import sys

base_directory = os.path.dirname(os.path.abspath(__file__))
data_directory = os.path.join(base_directory, '../../')
sys.path.insert(0, data_directory)
from src.data.data_process import read_data, extract_fraud_valid
from src.visualizations.visualize import plot_results
from src.ultils.feature import *

class TestCreditCardData(unittest.TestCase):

    @patch('src.data.data_process.pd.read_csv')
    def test_read_data(self, mock_read_csv):
        # Create a mock dataframe
        mock_data = pd.DataFrame({
            'Time': [1, 2, 3],
            'V1': [0.1, 0.2, 0.3],
            'V2': [0.4, 0.5, 0.6],
            'Amount': [100, 200, 300],
            'Class': [0, 1, 0]
        })
        mock_read_csv.return_value = mock_data

        # Call the function
        result = read_data()

        # Assertions
        mock_read_csv.assert_called_once()
        pd.testing.assert_frame_equal(result, mock_data)

    def test_extract_fraud_valid(self):
        # Create a mock dataframe with a realistic structure but fewer rows
        mock_data = pd.DataFrame({
            'Time': [1, 2, 3, 4],
            'V1': [0.1, 0.2, 0.3, 0.4],
            'V2': [0.4, 0.5, 0.6, 0.7],
            'V3': [0.1, 0.2, 0.3, 0.4],
            'V4': [0.5, 0.6, 0.7, 0.8],
            'V5': [0.1, 0.2, 0.3, 0.4],
            'V6': [0.5, 0.6, 0.7, 0.8],
            'V7': [0.1, 0.2, 0.3, 0.4],
            'V8': [0.5, 0.6, 0.7, 0.8],
            'V9': [0.1, 0.2, 0.3, 0.4],
            'V10': [0.5, 0.6, 0.7, 0.8],
            'V11': [0.1, 0.2, 0.3, 0.4],
            'V12': [0.5, 0.6, 0.7, 0.8],
            'V13': [0.1, 0.2, 0.3, 0.4],
            'V14': [0.5, 0.6, 0.7, 0.8],
            'V15': [0.1, 0.2, 0.3, 0.4],
            'V16': [0.5, 0.6, 0.7, 0.8],
            'V17': [0.1, 0.2, 0.3, 0.4],
            'V18': [0.5, 0.6, 0.7, 0.8],
            'V19': [0.1, 0.2, 0.3, 0.4],
            'V20': [0.5, 0.6, 0.7, 0.8],
            'V21': [0.1, 0.2, 0.3, 0.4],
            'V22': [0.5, 0.6, 0.7, 0.8],
            'V23': [0.1, 0.2, 0.3, 0.4],
            'V24': [0.5, 0.6, 0.7, 0.8],
            'V25': [0.1, 0.2, 0.3, 0.4],
            'V26': [0.5, 0.6, 0.7, 0.8],
            'V27': [0.1, 0.2, 0.3, 0.4],
            'V28': [0.5, 0.6, 0.7, 0.8],
            'Amount': [100, 200, 300, 400],
            'Class': [0, 1, 0, 1]
        })

        # Expected outputs
        expected_xData = np.array([
            [1, 0.1, 0.4, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 100],
            [2, 0.2, 0.5, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 200],
            [3, 0.3, 0.6, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 300],
            [4, 0.4, 0.7, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 400]
        ])
        expected_yData = np.array([0, 1, 0, 1])
        expected_fraud = mock_data[mock_data['Class'] == 1]

        with patch('src.data.data_process.read_data', return_value=mock_data):
            # Call the function
            xData, yData, fraud = extract_fraud_valid(mock_data)

            # Assertions
            np.testing.assert_array_equal(xData, expected_xData)
            np.testing.assert_array_equal(yData, expected_yData)
            pd.testing.assert_frame_equal(fraud, expected_fraud)

if __name__ == '__main__':
    unittest.main()
