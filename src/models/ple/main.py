import argparse
import gc
import os
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from src.evaluation.evaluate_model import ModelEvaluator
from src.evaluation.metrics import AvgAucScore, MrrScore, NdcgScore, AccuracyScore, F1Score, ConfusionMatrix
import src.models as models
from src.preprocess.data_preprocessor import DataProcessor
from src.training.train_model import MultitaskTrainer
import json

from src.utils.data import MultiTaskDataset
from src.utils.torch_utils import seed_everything

# Set seed for reproducibility
SEED = 51
seed_everything(SEED)

if __name__ == '__main__':
    ''' Usage: python main.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='PLE_ebnerd_small_x3', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    args = vars(parser.parse_args())

    experiment_id = args['expid']
    config = args['config']
    device = args['gpu']
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
        data_processor = DataProcessor.load_processor(processor_save_dir)
    else:
        print("Processing data from scratch...")
        data_processor = DataProcessor(feature_types, target_col, group_id, processor_save_dir)
        data_processor.process_from_files(train_file, valid_file, test_file)

    # Initialize DataLoaders for batching
    batch_size = 2048

    train_data = data_processor.load_data("train")
    train_loader = DataLoader(MultiTaskDataset(data_processor.feature_map, train_data), batch_size=batch_size, shuffle=True)
    del train_data
    gc.collect()

    valid_data = data_processor.load_data("valid")
    valid_loader = DataLoader(MultiTaskDataset(data_processor.feature_map, valid_data), batch_size=batch_size, shuffle=False)
    del valid_data
    gc.collect()

    # Initialize Model and Training Components
    print('Model and Training Components Initialization...')
    model_class = getattr(models, model_config['model'])
    model = model_class(feature_map=data_processor.feature_map, **model_config["config"])

    # Set up optimizer and loss function
    optimizer_class = optim.AdamW
    optimizer_params = {"lr": 1e-4, "weight_decay": 1e-5}
    tasks = [
        "binary-classification",  # task 1: click prediction
        "binary-classification",  # task 2
    ]

    criterion = [
        nn.BCELoss(),  # task 1: click prediction
        nn.BCELoss(),  # task 2
    ]
    monitor_metric = [  # (weight, metric_fn)
        (1, AvgAucScore()),  # task 1: click prediction
        (0.25, AvgAucScore()),  # task 2
    ]

    metric_functions_1 = [AvgAucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)]  # task 1: click prediction
    metric_functions_2 = [AvgAucScore(), AccuracyScore(), F1Score(), ConfusionMatrix()]  # task 2

    evaluators = [
        ModelEvaluator(metric_functions=metric_functions_1),  # task 1: click prediction
        ModelEvaluator(metric_functions=metric_functions_2)  # task 2
    ]

    trainer = MultitaskTrainer(model=model, optimizer_class=optimizer_class, optimizer_params=optimizer_params,
                               loss_function=criterion,  # adaptive_method="awl",
                               device=device, monitor_metric=monitor_metric, monitor_mode="max", task=tasks,
                               evaluator=evaluators, expid=experiment_id)
    print('******** Training ******** ')
    trainer.fit(train_loader=train_loader, val_loader=valid_loader, epochs=10, patience=5)
    del train_loader, valid_loader
    gc.collect()
    print('******** Model Evaluation ******** ')
    test_data = data_processor.load_data("test")
    test_loader = DataLoader(MultiTaskDataset(data_processor.feature_map, test_data), batch_size=batch_size, shuffle=False)
    del test_data
    gc.collect()
    trainer.evaluate_test(test_loader)
