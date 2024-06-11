from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import time, threading, logging


logger = logging.getLogger(__name__)

def evaluate_classification_model(model, xTest, yTest):
    """
    Evaluates a classification model using various metrics.

    Parameters:
        model: Trained classification model
        xTest: Features of the test set
        yTest: True labels of the test set

    Returns:
        dict: A dictionary containing accuracy, precision, recall, F1-score, and MCC
    """
    yPred = model.predict(xTest)
    return {
        'Accuracy': accuracy_score(yTest, yPred),
        'Precision': precision_score(yTest, yPred),
        'Recall': recall_score(yTest, yPred),
        'F1-Score': f1_score(yTest, yPred),
        'MCC': matthews_corrcoef(yTest, yPred)
    }

def evaluate_regression_model(model, xTest, yTest):
    """
    Evaluates a regression model using various metrics.

    Parameters:
        model: Trained regression model
        xTest: Features of the test set
        yTest: True values of the test set

    Returns:
        dict: A dictionary containing MSE, MAE, and R2 score
    """
    yPred = model.predict(xTest)
    return {
        'MSE': mean_squared_error(yTest, yPred),
        'MAE': mean_absolute_error(yTest, yPred),
        'R2 Score': r2_score(yTest, yPred)
    }

def train_and_evaluate(model, xTrain, yTrain, xTest, yTest, model_name, is_classification=True, transform=None):
    """
    Trains a model and evaluates it on a test set.

    Parameters:
        model: The machine learning model to train
        xTrain: Features of the training set
        yTrain: Labels or values of the training set
        xTest: Features of the test set
        yTest: Labels or values of the test set
        model_name: Name of the model (for logging purposes)
        is_classification: Boolean indicating if the model is a classification model (default is True)
        transform: Optional transformer for preprocessing (default is None)

    Returns:
        dict: A dictionary containing the evaluation metrics
    """
    logger.info(f"Training {model_name} model...")
    if transform:
        xTrain = transform.fit_transform(xTrain)
        xTest = transform.transform(xTest)
    # best_model = hyperparameter_tuning(model, param_distributions, xTrain, yTrain)
    model.fit(xTrain, yTrain)
    if is_classification:
        return evaluate_classification_model(model, xTest, yTest)
    else:
        return evaluate_regression_model(model, xTest, yTest)

def hyperparameter_tuning(model, param_distributions, xTrain, yTrain):
    randomized_search = RandomizedSearchCV(model, param_distributions=param_distributions, n_iter=50, cv=5, verbose=3, random_state=42, n_jobs=-1)
    
    progress_flag = {'running': True}
    progress_thread = threading.Thread(target=display_progress, args=(progress_flag,))
    progress_thread.start()
    
    logger.info("Starting Randomized Search")
    randomized_search.fit(xTrain, yTrain)
    logger.info("Randomized Search Completed")
    progress_flag['running'] = False
    progress_thread.join()
    
    logger.info(f"Best Estimator: {randomized_search.best_estimator_}")
    logger.info(f"Best Parameters: {randomized_search.best_params_}")
    
    return randomized_search.best_estimator_

def display_progress(progress_flag):
    """
    Displays a progress message periodically during a long-running process.

    Parameters:
        progress_flag: A dictionary with a 'running' key to control the progress display

    Returns:
        None
    """
    while progress_flag['running']:
        print("Grid search in progress...")
        time.sleep(10)

def log_results(results, title):
    """
    Logs the results of model comparisons.

    Parameters:
        results: A dictionary containing model names and their evaluation metrics
        title: Title for the logging section

    Returns:
        None
    """
    logger.info(f"{title} Model comparison:")
    for model_name, metrics in results.items():
        logger.info(f"{model_name}:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
