import argparse
import gc
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
from torch.utils.data import DataLoader
import src.models as models
from src.evaluation.evaluate_model import ModelEvaluator
import src.evaluation.metrics as metrics
from src.preprocess.data_preprocessor import DataProcessor
from src.training.train_model import Trainer
import json

from src.utils.data import SingleTaskDataset
from src.utils.torch_utils import seed_everything

if __name__ == '__main__':
    ''' 
    Usage: python run_stl_exp.py --model_dir {model_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='./src/models/mlp', help='the model directory')
    parser.add_argument('--expid', type=str, default='MLP_ebnerd_small_x2', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    args = vars(parser.parse_args())

    experiment_id = args['expid']
    model_dir = args['model_dir']
    device = args['gpu']

    # Set seed for reproducibility
    SEED = 51
    seed_everything(SEED)

    # Configuration
    dataset_config = json.load(open(f"{model_dir}/config/{experiment_id}/dataset_config.json", 'r'))
    model_config = json.load(open(f"{model_dir}/config/{experiment_id}/model_config.json", 'r'))
    trainer_config = json.load(open(f"{model_dir}/config/{experiment_id}/trainer_config.json", 'r'))

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
    print('Model and Training Components Initialization...')
    model_class = getattr(models, model_config['model'])
    model = model_class(feature_map=data_processor.feature_map, **model_config["config"])

    metric_functions = [metrics.AucScore()]
    evaluator = ModelEvaluator(metric_functions=metric_functions)

    trainer = Trainer(model=model, device=device, evaluator=evaluator, expid=experiment_id, **trainer_config)

    # Training
    print('******** Training ******** ')
    trainer.fit(train_loader=train_loader, val_loader=valid_loader, epochs=10, patience=3)
    del train_loader, valid_loader
    gc.collect()
    # Evaluation
    print('******** Model Evaluation ******** ')
    test_data = data_processor.load_data("test")
    test_loader = DataLoader(SingleTaskDataset(data_processor.feature_map, test_data), batch_size=batch_size,
                             shuffle=False)
    del test_data
    gc.collect()
    trainer.evaluate_test(test_loader)
