import argparse
import csv
import gc
import json
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import project modules
from src.evaluation.evaluate_model import ModelEvaluator
import src.evaluation.metrics as metrics
import src.models as models
from src.preprocess.data_preprocessor import DataProcessor
from src.training.train_model import Trainer
from src.utils.data import SingleTaskDataset
from src.utils.torch_utils import seed_everything

# Set working directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))


def run_experiment(seed, model_dir, experiment_id, device):
    """Runs the experiment with a given random seed and returns the evaluation results."""
    seed_everything(seed)  # Set random seed for reproducibility

    # Load experiment configurations
    dataset_config = json.load(open(f"{model_dir}/config/{experiment_id}/dataset_config.json", 'r'))
    model_config = json.load(open(f"{model_dir}/config/{experiment_id}/model_config.json", 'r'))
    trainer_config = json.load(open(f"{model_dir}/config/{experiment_id}/trainer_config.json", 'r'))

    # Extract dataset details
    dataset_id = dataset_config['dataset_id']
    train_file = dataset_config['train_file']
    valid_file = dataset_config['valid_file']
    test_file = dataset_config['test_file']
    feature_types = dataset_config['features']
    target_col = dataset_config['target']
    group_id = dataset_config['group_id']

    # Paths for saving/loading preprocessed data
    processor_save_dir = f"./data/saved_data/{dataset_id}/"

    # Load or preprocess data
    if os.path.exists(processor_save_dir):
        print("Loading preprocessed data and DataProcessor...")
        data_processor = DataProcessor.load_processor(processor_save_dir)
    else:
        print("Processing data from scratch...")
        data_processor = DataProcessor(feature_types, target_col, group_id, processor_save_dir)
        data_processor.process_from_files(train_file, valid_file, test_file)

    # DataLoaders for batching
    batch_size = 2048

    train_data = data_processor.load_data("train")
    train_loader = DataLoader(SingleTaskDataset(data_processor.feature_map, train_data), batch_size=batch_size,
                              shuffle=True)
    del train_data
    gc.collect()

    valid_data = data_processor.load_data("valid")
    valid_loader = DataLoader(SingleTaskDataset(data_processor.feature_map, valid_data), batch_size=batch_size,
                              shuffle=False)
    del valid_data
    gc.collect()

    # Initialize Model and Training Components
    print(f'Initializing model for seed {seed}...')
    model_class = getattr(models, model_config['model'])
    model = model_class(feature_map=data_processor.feature_map, **model_config["config"])

    metric_functions = [metrics.AucScore()]  # Task 1

    evaluators = ModelEvaluator(metric_functions=metric_functions)

    trainer = Trainer(model=model, device=device, evaluator=evaluators, expid=experiment_id, **trainer_config)

    print(f'Training model for seed {seed}...')
    trainer.fit(train_loader=train_loader, val_loader=valid_loader, epochs=10, patience=3)

    del train_loader, valid_loader
    gc.collect()

    # Model Evaluation
    print(f'Evaluating model for seed {seed}...')
    test_data = data_processor.load_data("test")
    test_loader = DataLoader(SingleTaskDataset(data_processor.feature_map, test_data), batch_size=batch_size,
                             shuffle=False)
    del test_data
    gc.collect()

    results = trainer.evaluate_test(test_loader)  # Expecting results as a dictionary
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='./src/models/mlp', help='The model directory')
    parser.add_argument('--expid', type=str, default='MLP_ebnerd_small_x1', help='The experiment ID to run')
    parser.add_argument('--gpu', type=int, default=-1, help='The GPU index, -1 for CPU')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 2024, 281611, 26022024], help='List of random seeds to use')
    parser.add_argument('--output_csv', type=str, default="experiment_results.csv", help="CSV file to store results")

    args = vars(parser.parse_args())

    model_dir = args['model_dir']
    experiment_id = args['expid']
    device = args['gpu']
    seeds = args['seeds']
    output_csv_dir = 'experiments/' + experiment_id
    os.makedirs(output_csv_dir, exist_ok=True)
    output_csv = output_csv_dir + '/' + args['output_csv']

    all_results = []

    for seed in seeds:
        print(f"Running experiment with seed {seed}...")
        results = run_experiment(seed, model_dir, experiment_id, device)
        results['seed'] = seed  # Store seed for reference
        all_results.append(results)

    # Convert results to a DataFrame and save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    # Compute mean and standard deviation
    mean_results = df.drop(columns=['seed']).mean()
    std_results = df.drop(columns=['seed']).std()

    # Save mean and std to CSV
    summary_df = pd.DataFrame({'Metric': mean_results.index, 'Mean': mean_results.values, 'Std': std_results.values})
    summary_csv = output_csv.replace(".csv", "_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print(f"Summary saved to {summary_csv}")
    print(f"\nMean Performance:\n", mean_results)
    print("\nStandard Deviation:\n", std_results)


if __name__ == '__main__':
    main()
