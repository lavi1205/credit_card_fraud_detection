import os
import pandas as pd

# Define the root directory based on the current file's location
root_directory = os.path.dirname(os.path.abspath(__file__)) 
# Define the path to the data file
data_directory = os.path.join(root_directory, '../../data/raw/creditcard.csv')

def read_data():
    """
    Reads the credit card data from a CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the credit card transaction data.
    """
    df = pd.read_csv(data_directory)
    return df

def extract_fraud_valid(data_frame):
    """
    Extracts fraud and valid transaction data, computes the outlier fraction, and provides statistical summaries.

    Parameters:
        data_frame (pd.DataFrame): DataFrame containing the credit card transaction data.

    Returns:
        tuple: A tuple containing the following:
            - xData (np.ndarray): Features from the dataset, excluding the 'Class' column.
            - yData (np.ndarray): Labels from the dataset, the 'Class' column.
            - fraud (pd.DataFrame): DataFrame containing only fraudulent transactions.
    """
    # Read the data
    data = read_data()
    # Separate the fraudulent and valid transactions
    fraud = data_frame[data_frame['Class'] == 1]
    valid = data_frame[data_frame['Class'] == 0]
    # Calculate the outlier fraction
    outlierFraction = len(fraud) / float(len(valid))
    print(outlierFraction)
    # Print the number of fraud and valid cases
    print('Fraud Cases: {}'.format(len(data_frame[data_frame['Class'] == 1])))
    print('Valid Transactions: {}'.format(len(data_frame[data_frame['Class'] == 0])))
    # Print amount details of the fraudulent transactions
    print('Amount details of the fraudulent transaction')
    print(fraud.Amount.describe())
    print(valid.Amount.describe())
    # Divide the data into features (X) and labels (Y)
    X = data.drop(['Class'], axis=1)
    Y = data["Class"]
    print(X.shape)
    print(Y.shape)
    # Get the values for processing (as numpy arrays)
    xData = X.values
    yData = Y.values
    # Return the feature data, label data, and fraud data
    result = (xData, yData, fraud)
    return result

def main():
    extract_fraud_valid(data_frame=read_data())

if __name__ == "__main__":
    main()
