import os
import pandas as pd

root_directory = os.path.dirname(os.path.abspath(__file__)) 
data_directory = os.path.join(root_directory, '../data/raw/creditcard.csv')

def read_data(data_directory):
    df = pd.read_csv(data_directory)
    return df


def extract_fraud_valid(data_frame):
    fraud = data_frame[data_frame['Class'] == 1]
    print(fraud)
    valid = data_frame[data_frame['Class'] == 0]
    print(valid)
    outlierFraction = len(fraud)/float(len(valid))
    print(outlierFraction)
    print('Fraud Cases: {}'.format(len(data_frame[data_frame['Class'] == 1])))
    print('Valid Transactions: {}'.format(len(data_frame[data_frame['Class'] == 0])))

print(extract_fraud_valid(data_frame=read_data(data_directory=data_directory)))
