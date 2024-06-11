# Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using machine learning techniques. The code is organized into several modules, each responsible for different aspects of the project.

## Directory Structure

The project has the following directory structure:


### Descriptions of Each Directory and Module

- **data/**: Contains modules related to data processing.
  - `__init__.py`: Initializes the data module.
  - `data_process.py`: Contains functions for processing raw data into a format suitable for analysis and modeling.

- **models/**: Contains modules related to model training and evaluation.
  - `__init__.py`: Initializes the models module.
  - `train_model.py`: Contains functions and classes for training machine learning models on the processed data.

- **tests/**: Contains test cases to ensure the correctness of the code.
  - `__init__.py`: Initializes the tests module.
  - `test_data_process.py`: Contains test functions for the data processing module.

- **utils/**: Contains utility functions used across the project.
  - `__init__.py`: Initializes the utils module.
  - `feature.py`: Contains functions for feature engineering and other utility operations.

- **visualizations/**: Contains modules related to data visualization.
  - `__init__.py`: Initializes the visualizations module.
  - `visualize.py`: Contains functions for visualizing data and model results.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/lavi1205/credit_card_fraud_detection.git
    cd credit-card-fraud-detection
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. **Data Processing**: Prepare your data by running the data processing script.
    ```bash
    python src/data/data_process.py
    ```

2. **Model Training**: Train your model using the processed data.
    ```bash
    python src/models/train_model.py
    ```

3. **Visualization**: Generate visualizations to understand the data and model performance.
    ```bash
    python src/visualizations/visualize.py
    ```

4. **Testing**: Run tests to ensure everything is working correctly.
    ```bash
    python -m unittest discover src/tests
    ```

## Project Description

The goal of this project is to build a robust model for detecting fraudulent credit card transactions. The project involves data processing, feature engineering, model training, and evaluation. Visualizations are used to understand the data and model performance better.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Dataset Source](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Any other acknowledgments here.

---

Feel free to contribute to this project by opening issues or submitting pull requests.
