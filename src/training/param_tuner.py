import gc
import json
import os

import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from torch import nn
from torch.utils.data import DataLoader

from src import AvgAucScore, MrrScore, AccuracyScore, NdcgScore, F1Score, ConfusionMatrix, ModelEvaluator, DataProcessor


class ParamTuner:
    def __init__(self, model_class, trainer_class, optimizer_class, dataset_class):
        self.model_class = model_class
        self.trainer_class = trainer_class
        self.optimizer_class = optimizer_class
        self.dataset_class = dataset_class

    def fit(self, param_grid, max_evals): ...


class GridSearch(ParamTuner):
    def __init__(self, model_class, trainer_class, optimizer_class, dataset_class):
        super(GridSearch).__init__(model_class, trainer_class, optimizer_class, dataset_class)

    def fit(self, param_grid, max_evals):
            pass


class TPESearch(ParamTuner):
    def __init__(self, model_class, trainer_class, optimizer_class, dataset_class, seed):
        super(TPESearch).__init__(model_class, trainer_class, optimizer_class, dataset_class)
        self.tpe_algorithm = tpe.suggest
        self.baeyes_trials = Trials()
        self.SEED = seed

    def _objective(self, params):
        print("########################### Starting new trial ###########################")
        print(f"Training with params: {params}")

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
        optimizer = self.optimizer_class(model.parameters(), lr=1e-2, weight_decay=1e-5)
        tasks = [
            "binary-classification",  # task 1: click prediction
            "binary-classification",  # task 2
        ]

        criterion = [  # (weight, loss_fn)
            (1, nn.BCELoss()),  # task 1: click prediction
            (1, nn.BCELoss()),  # task 2
        ]
        monitor_metric = [  # (weight, metric_fn)
            (0.75, AvgAucScore()),  # task 1: click prediction
            (0.25, AvgAucScore()),  # task 2
        ]

        metric_functions_1 = [AvgAucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)]  # task 1: click prediction
        metric_functions_2 = [AvgAucScore(), AccuracyScore(), F1Score(), ConfusionMatrix()]  # task 2

        evaluators = [
            ModelEvaluator(metric_functions=metric_functions_1),  # task 1: click prediction
            ModelEvaluator(metric_functions=metric_functions_2)  # task 2
        ]

        trainer = self.trainer_class(model=model, optimizer=optimizer, loss_function=criterion, device=device,
                                     monitor_metric=monitor_metric, monitor_mode="max", task=tasks,
                                     evaluator=evaluators, expid=experiment_id)

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
                    trials=self.baeyes_trials, rstate=np.random.default_rng(self.SEED), verbose=True,
                    show_progressbar=False)
        return best
