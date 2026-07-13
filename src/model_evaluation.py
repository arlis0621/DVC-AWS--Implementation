
import numpy as np
import pandas as pd
import yaml
from dvclive.live import Live



import pickle
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging

log_dir='logs'

os.makedirs(log_dir, exist_ok=True)

logger=logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')
console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path=os.path.join(log_dir, 'model_evaluation.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(congig_path:str)->dict:
    """Load parameters from a YAML file."""
    try:
        with open(congig_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', congig_path)
        return params
    except FileNotFoundError as e:
        logger.error('Configuration file not found: %s', e)
        raise
    except yaml.YAMLError as e:
        logger.error('Error parsing the YAML file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading parameters: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise
    

def load_model(model_path: str):
    """Load a trained model from a file."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Failed to load the model: %s', e)
        raise

def evaluate_model(model, X_test: np.ndarray, y_test) -> dict:
    """Evaluate the model and return performance metrics."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Get probabilities
        y_prob = model.predict_proba(X_test)
        
        # If binary classification, take probabilities of the positive class only
        if y_prob.shape[1] == 2:
            roc_auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        }

        logger.debug('Model evaluated with metrics: %s', metrics)
        return metrics
    except Exception as e:
        logger.error('Failed to evaluate the model: %s', e)
        raise
    
def save_metrics(metrics: dict, output_path: str):
    """Save evaluation metrics to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.debug('Metrics saved to %s', output_path)
    except Exception as e:
        logger.error('Failed to save metrics: %s', e)
        raise

def main():
    """Main function to load model, evaluate it, and save metrics."""
    try:
        params = load_params('params.yaml')
        # model_path = 'models/trained_model.pkl'
        # test_data_path = 'data/test_data.csv'
        metrics_output_path = 'reports/metrics.json'

        # Load the trained model
        model = load_model('./models/model.pkl')
        test_data=load_data('./data/processed/test_tfidf.csv')
        
        
        

        # Load the test data
        # test_data = pd.read_csv(test_data_path)
        X_test = test_data.iloc[:, :-1].values  # Assuming the last column is the target
        y_test = test_data.iloc[:, -1].values

        # Evaluate the model
        metrics = evaluate_model(model, X_test, y_test)
        #experiment tracking using dvclive
        
        with Live(save_dvc_exp=True) as live:
            live.log_metric("accuracy", metrics['accuracy'])
            live.log_metric("precision", metrics['precision'])
            live.log_metric("recall", metrics['recall'])
            live.log_params(params)

        # Save the evaluation metrics
        save_metrics(metrics, metrics_output_path)

    except Exception as e:
        logger.error('An error occurred in the main function: %s', e)    
        
        
if __name__ == "__main__":
    main()
    
