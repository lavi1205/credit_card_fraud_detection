from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import logging
import matplotlib.pyplot as plt
import os, sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

base_directory = os.path.dirname(os.path.abspath(__file__)) 
data_directory = os.path.join(base_directory, '../../')
sys.path.insert(0, data_directory)
from src.data.data_process import read_data, extract_fraud_valid
from src.visualizations.visualize import plot_results
from src.ultils.feature import *

def read_and_preprocess_data():
    logger.info("Reading data...")
    values = extract_fraud_valid(data_frame=read_data())
    xData, yData = values[0], values[1]
    logger.info("Splitting data into training and testing sets...")
    return train_test_split(xData, yData, test_size=0.2, random_state=42)

def main():
        # Main Script
        xTrain, xTest, yTrain, yTest = read_and_preprocess_data()

        # Initialize results dictionary
        results_classification = {}
        results_regression = {}

        # Random Forest Model and Grid Search
        rfc_model = RandomForestClassifier()
        results_classification['Random Forest'] = train_and_evaluate(rfc_model, xTrain, yTrain, xTest, yTest, 'Random Forest')

        # param_grid = {
        #     'n_estimators': [25, 50, 100, 150],
        #     'max_features': ['sqrt', 'log2', None],
        #     'max_depth': [3, 6, 9],
        #     'max_leaf_nodes': [3, 6, 9]
        # }
        # grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, verbose=3)

        # progress_flag = {'running': True}
        # progress_thread = threading.Thread(target=display_progress, args=(progress_flag,))
        # progress_thread.start()

        # logger.info("Starting Grid Search")
        # grid_search.fit(xTrain, yTrain)
        # logger.info("Grid Search Completed")
        # progress_flag['running'] = False
        # progress_thread.join()

        # logger.info(f"Best Estimator: {grid_search.best_estimator_}")
        # logger.info(f"Best Parameters: {grid_search.best_params_}")

        # Polynomial Regression
        poly = PolynomialFeatures(degree=2)
        results_regression['Polynomial Regression'] = train_and_evaluate(LinearRegression(), xTrain, yTrain, xTest, yTest, 'Polynomial Regression', is_classification=False, transform=poly)

        # Linear Regression
        results_regression['Linear Regression'] = train_and_evaluate(LinearRegression(), xTrain, yTrain, xTest, yTest, 'Linear Regression', is_classification=False)

        # Logistic Regression
        results_classification['Logistic Regression'] = train_and_evaluate(LogisticRegression(max_iter=10000), xTrain, yTrain, xTest, yTest, 'Logistic Regression')

        # Decision Tree
        results_classification['Decision Tree'] = train_and_evaluate(DecisionTreeClassifier(), xTrain, yTrain, xTest, yTest, 'Decision Tree')

        # K-Nearest Neighbors
        scaler = StandardScaler()
        results_classification['K-Nearest Neighbors'] = train_and_evaluate(KNeighborsClassifier(), xTrain, yTrain, xTest, yTest, 'K-Nearest Neighbors', transform=scaler)

        # Naive Bayes
        results_classification['Naive Bayes'] = train_and_evaluate(GaussianNB(), xTrain, yTrain, xTest, yTest, 'Naive Bayes')

        # Visualize results
        plot_results(results_classification, ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC'], 'Classification Models Performance Comparison', 'Classification.png')
        plot_results(results_regression, ['MSE', 'MAE', 'R2 Score'], 'Regression Models Performance Comparison', 'Regression.png')

        # Log results
        log_results(results_classification, "Classification")
        log_results(results_regression, "Regression")


if __name__ == '__main__':
    main()