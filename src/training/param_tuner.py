import gc
import json
import os

import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from torch import nn
from torch.utils.data import DataLoader

from src import AvgAucScore, MrrScore, AccuracyScore, NdcgScore, F1Score, ConfusionMatrix, ModelEvaluator, DataProcessor


class ParamTuner:
    def __init__(self,
                 experiment_id,
                 model_dir,
                 device,
                 model_class,
                 trainer_class,
                 optimizer_class,
                 dataset_class,
                 loss_function,
                 monitor_metric,
                 monitor_mode,
                 evaluator,
                 task,
                 adaptive_method=None,
                 seed=51):
        self.model_class = model_class
        self.trainer_class = trainer_class
        self.optimizer_class = optimizer_class
        self.dataset_class = dataset_class
        self.seed = seed
        self.experiment_id = experiment_id
        self.model_dir = model_dir
        self.device = device
        self.loss_function = loss_function
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.evaluator = evaluator
        self.task = task
        self.adaptive_method = adaptive_method

    def fit(self, param_grid, max_evals): ...


class GridSearch(ParamTuner):
    def __init__(self, model_class, trainer_class, optimizer_class, dataset_class):
        super(GridSearch).__init__(model_class, trainer_class, optimizer_class, dataset_class)

    def fit(self, param_grid, max_evals):
        pass


class TPESearch(ParamTuner):
    def __init__(self,
                 experiment_id,
                 model_dir,
                 device,
                 model_class,
                 trainer_class,
                 optimizer_class,
                 dataset_class,
                 loss_function,
                 monitor_metric,
                 monitor_mode,
                 evaluator,
                 task,
                 adaptive_method=None,
                 seed=51):
        super(TPESearch).__init__(experiment_id,
                                  model_dir,
                                  device,
                                  model_class,
                                  trainer_class,
                                  optimizer_class,
                                  dataset_class,
                                  loss_function,
                                  monitor_metric,
                                  monitor_mode,
                                  evaluator,
                                  task,
                                  adaptive_method,
                                  seed)
        self.tpe_algorithm = tpe.suggest
        self.baeyes_trials = Trials()

    def _objective(self, params):
        print("########################### Starting new trial ###########################")
        print(f"Training with params: {params}")

        # Configuration
        dataset_config = json.load(open(f"{self.model_dir}/config/{self.experiment_id}/dataset_config.json", 'r'))

        train_file = dataset_config['train_file']
        valid_file = dataset_config['valid_file']
        test_file = dataset_config['test_file']
        feature_types = dataset_config['features']
        target_col = dataset_config['target']
        group_id = dataset_config['group_id']

        # Paths for saving/loading preprocessed data
        processor_save_dir = f"./saved_data/{self.experiment_id}/"
        # Load or preprocess data
        if os.path.exists(processor_save_dir):  # delete this path if you want to process data from scratch...
            print("Loading preprocessed data and DataProcessor...")
            data_processor = DataProcessor.load_processor(processor_save_dir)
        else:
            print("Processing data from scratch...")
            data_processor = DataProcessor(feature_types, target_col, group_id, processor_save_dir)
            data_processor.process_from_files(train_file, valid_file, test_file)

        batch_size = params['batch_size']

        train_data = data_processor.load_data("train")
        train_loader = DataLoader(self.dataset_class(data_processor.feature_map, train_data), batch_size=batch_size,
                                  shuffle=True)
        del train_data
        gc.collect()

        valid_data = data_processor.load_data("valid")
        valid_loader = DataLoader(self.dataset_class(data_processor.feature_map, valid_data), batch_size=batch_size,
                                  shuffle=False)
        del valid_data
        gc.collect()

        model = self.model_class(**params)

        # Set up optimizer and loss function
        trainer_config = {
            "optimizer": "AdamW",
            "optimizer_params": {"lr": params["lr"], "weight_decay": params["weight_decay"]},
            "task": ["binary-classification", "binary-classification"],
            "seq_dependence": [None, 0],
            "adaptive_method": {"name": "uw"},
            "loss_function": ["BCELoss", "BCELoss"],
            "log_dir": "./logs/",
            "save_path": "./checkpoints/"
        }

        trainer = self.trainer_class(model=model, device=self.device, monitor_metric=self.monitor_metric,
                                     monitor_mode=self.monitor_mode, evaluator=self.evaluator, expid=self.experiment_id,
                                     **trainer_config)

        print('******** Training ******** ')
        trainer.fit(train_loader=train_loader, val_loader=valid_loader, epochs=10, patience=5)
        del train_loader, valid_loader
        gc.collect()
        if trainer.monitor_mode == "max":
            loss = 1 - trainer.best_val_metric
        else:
            loss = trainer.best_val_metric
        return {'loss': loss, 'params': params, 'status': STATUS_OK}

    def fit(self, param_grid, max_evals):
        space = {key: hp.choice(key, values) for key, values in param_grid.items()}

        best = fmin(fn=self._objective, space=space, algo=self.tpe_algorithm, max_evals=max_evals,
                    trials=self.baeyes_trials, rstate=np.random.default_rng(self.seed), verbose=True,
                    show_progressbar=False)
        return best
