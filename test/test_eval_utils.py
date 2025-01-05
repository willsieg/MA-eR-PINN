import unittest
import torch
import numpy as np
import os
from pathlib import Path
from eval_utils import save_checkpoint, load_checkpoint, plot_training_performance, plot_prediction, calculate_metrics

class TestEvalUtils(unittest.TestCase):

    def setUp(self):
        # Setup common variables for tests
        self.trainer = type('Trainer', (object,), {
            'model': torch.nn.Linear(10, 1),
            'optimizer': torch.optim.Adam(torch.nn.Linear(10, 1).parameters()),
            'scheduler': None,
            'state': None,
            'clip_value': None,
            'device': torch.device('cpu'),
            'use_mixed_precision': False,
            'log_file': None
        })()
        self.train_loader = [torch.randn(10, 10) for _ in range(5)]
        self.val_loader = [torch.randn(10, 10) for _ in range(2)]
        self.test_loader = [torch.randn(10, 10) for _ in range(1)]
        self.checkpoint = {}
        self.config = {'param1': 'value1'}
        self.subset_files = {'train_files': ['file1', 'file2']}
        self.pth_folder = Path('./')

    def test_save_checkpoint(self):
        checkpoint, model_destination_path = save_checkpoint(
            self.trainer, self.train_loader, self.val_loader, self.test_loader, self.checkpoint, self.config, self.subset_files, self.pth_folder
        )
        self.assertTrue(os.path.exists(model_destination_path))

    def test_load_checkpoint(self):
        checkpoint, model_destination_path = save_checkpoint(
            self.trainer, self.train_loader, self.val_loader, self.test_loader, self.checkpoint, self.config, self.subset_files, self.pth_folder
        )
        loaded_checkpoint = load_checkpoint(model_destination_path, torch.device('cpu'))
        self.assertIn('model_name_id', loaded_checkpoint)

    def test_plot_training_performance(self):
        train_losses_per_iter = np.random.rand(10)
        train_losses = np.random.rand(10)
        val_losses = np.random.rand(10)
        lr_history = np.random.rand(10)
        train_batches = 1
        num_epochs = 10
        model_name_id = 'test_model'
        plot_training_performance(train_losses_per_iter, train_losses, val_losses, lr_history, train_batches, num_epochs, model_name_id)

    def test_plot_prediction(self):
        y_true = np.random.rand(100)
        y_pred = np.random.rand(100)
        plot_prediction(y_true, y_pred, plot_active=False)

    def test_calculate_metrics(self):
        y_true = np.random.rand(100)
        y_pred = np.random.rand(100)
        metrics = calculate_metrics(y_true, y_pred)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('std_dev', metrics)
        self.assertIn('mape', metrics)
        self.assertIn('r2', metrics)
        self.assertIn('max_error', metrics)

if __name__ == '__main__':
    unittest.main()