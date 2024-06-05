import os
import pandas as pd

root_directory = os.path.dirname(os.path.abspath(__file__)) 
data_directory = os.path.join(root_directory, '../../data/raw/creditcard.csv')

    
def read_data():
    df = pd.read_csv(data_directory)
    return df


def extract_fraud_valid(data_frame):
    data = read_data()
    fraud = data_frame[data_frame['Class'] == 1]
    valid = data_frame[data_frame['Class'] == 0]
    outlierFraction = len(fraud)/float(len(valid))
    print(outlierFraction)
    # print('Fraud Cases: {}'.format(len(data_frame[data_frame['Class'] == 1])))
    # print('Valid Transactions: {}'.format(len(data_frame[data_frame['Class'] == 0])))
    # print('Amount details of the fraudulent transaction')
    # print(fraud.Amount.describe())
    # print(valid.Amount.describe())
    # dividing the X and the Y from the dataset
    X = data.drop(['Class'], axis = 1)
    Y = data["Class"]
    # print(X.shape)
    # print(Y.shape)
    # getting just the values for the sake of processing 
    # (its a numpy array with no columns)
    xData = X.values
    yData = Y.values
    result = (xData,yData,fraud)
    # print(result[0])
    # print(result[1])
    # print(result[2])
    return result


print(extract_fraud_valid(data_frame=read_data()))
