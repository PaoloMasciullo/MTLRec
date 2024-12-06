import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from src import F1Score, AccuracyScore, ConfusionMatrix
from src.evaluation.evaluate_model import ModelEvaluator
from src.evaluation.metrics import AvgAucScore, MrrScore, NdcgScore, RMSE, MSE, R2Score
from src.models.simple_mlp.simple_mlp import MLP
from src.preprocess.data_preprocessor import DataProcessor
from src.training.train_model import Trainer
import json

from src.utils.data import SingleTaskDataset
from src.utils.torch_utils import seed_everything

if __name__ == '__main__':
    ''' 
    Usage: python main.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='MLP_ebnerd_demo_x1', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    args = vars(parser.parse_args())

    experiment_id = args['expid']
    config = args['config']

    # Set seed for reproducibility
    SEED = 51
    seed_everything(SEED)

    # Configuration
    dataset_config = json.load(open(config + experiment_id + '/dataset_config.json', 'r'))
    model_config = json.load(open(config + experiment_id + '/model_config.json', 'r'))

    train_file = dataset_config['train_file']
    valid_file = dataset_config['valid_file']
    test_file = dataset_config['test_file']
    feature_types = dataset_config['features']
    target_col = dataset_config['target']
    group_id = dataset_config['group_id']

    # Paths for saving/loading preprocessed data
    processor_save_dir = f"./saved_data/{experiment_id}/"
    # Load or preprocess data
    if os.path.exists(processor_save_dir):  # delete this path if you want to process data from scratch...
        print("Loading preprocessed data and DataProcessor...")
        data_processor, processed_data = DataProcessor.load_processor(processor_save_dir)
    else:
        print("Processing data from scratch...")
        data_processor = DataProcessor(feature_types, target_col, group_id)
        processed_data = data_processor.process_from_files(train_file, valid_file, test_file)
        os.makedirs(processor_save_dir)
        data_processor.save(processor_save_dir, processed_data)

    # Split processed data
    train_data = processed_data["train"]
    valid_data = processed_data["valid"]
    test_data = processed_data["test"]

    # Initialize DataLoaders for batching
    batch_size = 32

    train_loader = DataLoader(SingleTaskDataset(data_processor.feature_map, train_data), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(SingleTaskDataset(data_processor.feature_map, valid_data), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SingleTaskDataset(data_processor.feature_map, test_data), batch_size=batch_size, shuffle=False)

    # Initialize Model and Training Components
    print('Model and Training Components Initialization...')
    model = MLP(feature_map=data_processor.feature_map, **model_config)

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    monitor_metric = AvgAucScore()
    metric_functions = [AvgAucScore(), MrrScore(), NdcgScore(k=5)]
    evaluator = ModelEvaluator(metric_functions=metric_functions)
    trainer = Trainer(model=model, optimizer=optimizer, loss_function=criterion, monitor_metric=monitor_metric,
                      monitor_mode="max", task="binary-classification", evaluator=evaluator, expid=experiment_id)

    # Training
    print('Starting Training...')
    trainer.fit(train_loader=train_loader, val_loader=valid_loader, epochs=10, patience=5)

    # Evaluation
    print('Model Evaluation...')
    trainer.evaluate_test(test_loader)
