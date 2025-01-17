import os
import torch
import pickle
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from tabulate import tabulate



def save_checkpoint(trainer, train_loader, val_loader, test_loader, checkpoint, config, subset_files, pth_folder, timestamp) -> tuple:

    # Collecting results and meta data for saving dict
    trainer_add_info = {key: getattr(trainer, key) for key in ['model', 'optimizer', 'lr_scheduler', 'l_p_scheduler', 'state', 'clip_value', 'device', 'use_mixed_precision']}
    loader_sizes = {'train_batches': len(train_loader), 'val_batches': len(val_loader), 'test_batches': len(test_loader)}
    checkpoint['CONFIG'] = config
    checkpoint = {**checkpoint, **trainer_add_info, **subset_files, **loader_sizes}

    # Create unique identifier for model name
    model_name_id = f'{trainer.model.__class__.__name__}_{timestamp}'
    checkpoint['model_name_id'] = model_name_id
    model_destination_path = Path(pth_folder, model_name_id + ".pt")

    if trainer.log_file:
        log_file_path = Path(pth_folder, model_name_id + "_log.txt")
        with open(trainer.log_file, 'r') as original_log, open(log_file_path, 'w') as new_log:
            
            new_log.write(original_log.read())
            config_df = pd.DataFrame(list(config.items()), columns=['Parameter', 'Value'])
            config_df['Value'] = config_df['Value'].apply(lambda x: str(x).replace(',', ',\n') if len(str(x)) > 60 else str(x))
        
            new_log.write(f"CONFIG Dictionary:\n{'-'*60}\n")
            new_log.write(tabulate(config_df, headers='keys', colalign=("left", "left"), maxcolwidths=[30, 60]))
            new_log.write(f"\n{'-'*60}\n")

    # Save checkpoint
    torch.save(checkpoint, model_destination_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    # Check saved object size
    print(f"Model saved to:\t {os.path.basename(model_destination_path)}\n{'-'*60}\nModel ID: {model_name_id}\n{'-'*60}\nSize: {os.path.getsize(model_destination_path) / 1024**2:.2f} MB\n{'-'*60}")
    print(f"log_file: {checkpoint['log_file']}")
    if os.path.getsize(model_destination_path) > 100 * 1024**2: 
        print("--> Warning: saved model size exceeds 100MB! Creating a zip file instead ...")
        try:
            import io, zipfile
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            buffer.seek(0)
            model_destination_path_zip = Path(pth_folder, model_name_id + ".zip")
            with zipfile.ZipFile(model_destination_path_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf: 
                zipf.writestr(model_destination_path_zip.name, buffer.read())
                if os.path.getsize(model_destination_path_zip) > 100 * 1024**2:
                    print("--> Warning: zip compressed model size still exceeds 100MB!")
                print(f"Model saved to:\t {model_destination_path_zip}\n{'-'*60}\nSize: {os.path.getsize(model_destination_path_zip) / 1024**2:.2f} MB\n{'-'*60}")
        except Exception as e:
            print(f"An error occurred while trying to save the model as a zip file: {e}")

    return checkpoint, model_destination_path


def load_checkpoint(model_destination_path, DEVICE) -> dict:
    try: 
        checkpoint = torch.load(model_destination_path, weights_only=False, \
            map_location=DEVICE if (torch.cuda.is_available()) else torch.device('cpu'))
    except NotImplementedError:
        import pathlib
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        # temporarily change PosixPath to WindowsPath for loading
        checkpoint = torch.load(model_destination_path, weights_only=False, map_location=DEVICE if torch.cuda.is_available() else torch.device('cpu'))
        pathlib.PosixPath = temp

    print(f"Model loaded from:\t{model_destination_path}\n{'-'*60}\nModel: {checkpoint['model'].__class__.__name__}\tParameters on device: {next(checkpoint['model'].parameters()).device}"
        f"\n{'-'*60}\nTrain/Batch size:\t{checkpoint['train_batches']} / {checkpoint['CONFIG']['BATCH_SIZE']}\n"
        f"Loss:\t\t\t{checkpoint['loss_fn']}\nOptimizer:\t\t{checkpoint['optimizer'].__class__.__name__}\nLR:\t\t\t"
        f"{checkpoint['optimizer'].param_groups[0]['lr']}\nWeight Decay:\t\t{checkpoint['optimizer'].param_groups[0]['weight_decay']}\n{'-'*60}\n")

    return checkpoint


# PLOT TRAINING PERFORMANCE ------------------------------------------------------
def plot_training_performance(results):

    # get required data from results dict
    train_losses_per_iter = results['train_losses_per_iter']
    train_losses, val_losses = results['train_losses'], results['val_losses']
    lr_history = results['lr_history']
    l_p_history = results['l_p_history']
    num_epochs = results['epoch']
    log_file = results['log_file']

    # plot training performance:
    fig, ax1 = plt.subplots(figsize=(12,3))
    ax1.set_xlabel('Epochs')
    ax1.set_xticks(range(1, num_epochs + 1))

    ax1.plot(np.linspace(1, num_epochs, len(train_losses_per_iter)), train_losses_per_iter, label='batch_loss', color='lightblue')
    ax1.plot(range(1, num_epochs + 1), train_losses, label='train_loss', color='blue')
    ax1.plot(range(1, num_epochs + 1), val_losses, label='val_loss', color='red')

    ax1.set_yscale('log')
    fig.tight_layout(pad=0.8)
    ax1.legend()

    ax1.text(0.86, 0.6, f"Train: {train_losses[-1]:.3e}\nVal:    {val_losses[-1]:.3e}", \
        transform=ax1.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # Add the log file name to the plot
    fig.text(0.01, 0.01, f" {os.path.basename(log_file).strip('_log.txt')}", fontsize=8, color='gray', alpha=0.7)

    if pd.Series(l_p_history).nunique() > 1:
        ax2 = ax1.twinx()
        ax2.plot(np.linspace(1, num_epochs, len(l_p_history)), l_p_history, label='l_p', color='green', linestyle='--')
        ax2.set_ylabel('l_p', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        #ax2.set_yscale('log')


    # Save the plot to the log file
    plt.savefig(Path(log_file).with_suffix('.png'))

    plt.show()


# PLOT PREDICTION -----------------------------------------------------------------
def plot_prediction(y_true, y_pred, plot_active=True):
     if plot_active:
          plt.figure(figsize=(18,4))
          plt.xlabel('Time in s')
          plt.ylabel('SOC in %')
          plt.title('Battery State of Charge: Prediction vs. Actual Data')
          plt.plot(y_true, label='Actual Data')  # actual plot
          plt.plot(np.arange(0, len(y_true), 1), y_pred, label='Predicted Data')  # predicted plot
          plt.legend()
          plt.text(0.01, 0.02, f"RMSE: {root_mean_squared_error(y_true, y_pred):.4f}\nStd Dev: {np.std(y_true - y_pred):.4f}",
          transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

          plt.figure(figsize=(18,4))
          plt.xlabel('Time in s')
          plt.ylabel('SOC in %')
          plt.plot(savgol_filter(y_true.flatten(), window_length=60, polyorder=3), label='Actual Data (Smoothed)')  # actual plot
          plt.plot(np.arange(0, len(y_true), 1), savgol_filter(y_pred.flatten(), window_length=60, polyorder=3), label='Predicted Data (Smoothed)')  # predicted plot
          plt.legend()


def calculate_metrics(y_true, y_pred):
    metrics = {
        "rmse": root_mean_squared_error(y_true, y_pred),                  # Root Mean Squared Error
        "mae": np.mean(np.abs(y_true - y_pred)),                          # Mean Absolute Error
        "std_dev": np.std(y_true - y_pred),                               # Standard Deviation
        "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,        # Mean Absolute Percentage Error
        "r2": r2_score(y_true, y_pred),                                   # R-squared
        "max_error": np.max(np.abs(y_true - y_pred))                      # Maximum Error
        }

    print(f"RMSE:\t\t\t{metrics['rmse']:.4f}\
        \nMAE ± STD (MAPE):\t{metrics['mae']:.4f} ± {metrics['std_dev']:.4f} ({metrics['mape']:.2f}%)\nR-squared:\t\t{metrics['r2']:.4f}\n{'-'*60}")
    
    return metrics

def calculate_metrics_per_sequence(scaled_targets, scaled_outputs):
    all_metrics = []

    for y_true, y_pred in zip(scaled_targets, scaled_outputs):
        metrics = {
            "rmse": root_mean_squared_error(y_true, y_pred),                  # Root Mean Squared Error
            "mae": np.mean(np.abs(y_true - y_pred)),                          # Mean Absolute Error
            "std_dev": np.std(y_true - y_pred),                               # Standard Deviation
            "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,        # Mean Absolute Percentage Error
            "r2": r2_score(y_true, y_pred),                                   # R-squared
            "max_error": np.max(np.abs(y_true - y_pred))                      # Maximum Error
            }
        all_metrics.append(metrics)

    mean_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}

    print(f"RMSE:\t\t\t{mean_metrics['rmse']:.4f}\
        \nMAE ± STD (MAPE):\t{mean_metrics['mae']:.4f} ± {mean_metrics['std_dev']:.4f} ({mean_metrics['mape']:.2f}%)\nR-squared:\t\t{mean_metrics['r2']:.4f}\n{'-'*60}")
    
    return mean_metrics

'''
# RESHAPE EVALUATION OUTPUTS (NOT REQUIRED)------------------------------------------------------
def concat_outputs_targets(outputs, targets, original_lengths) -> tuple:
    all_outputs, all_targets, all_original_lengths = [], [], []
    for batch_outputs, batch_targets, batch_lengths in zip(outputs, targets, original_lengths):
        all_outputs.extend(batch_outputs)
        all_targets.extend(batch_targets)
        all_original_lengths.extend(batch_lengths)
        return all_outputs, all_targets, all_original_lengths


def concat_outputs_targets_priors(outputs, targets, priors, original_lengths) -> tuple:
    all_outputs, all_targets, all_priors, all_original_lengths = [], [], [], []
    for batch_outputs, batch_targets, batch_priors, batch_lengths in zip(outputs, targets, priors, original_lengths):
        all_outputs.extend(batch_outputs)
        all_targets.extend(batch_targets)
        all_priors.extend(batch_priors)
        all_original_lengths.extend(batch_lengths)
        return all_outputs, all_targets, all_priors, all_original_lengths
'''       