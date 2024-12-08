
import os
import torch
import pickle
from datetime import datetime

def save_checkpoint(trainer, train_loader, val_loader, test_loader, checkpoint, config, subset_files, pth_folder):

    # Collecting results and meta data for saving dict
    trainer_add_info = {key: getattr(trainer, key) for key in ['model', 'optimizer', 'scheduler', 'state', 'clip_value', 'device', 'use_mixed_precision']}
    loader_sizes = {'train_batches': len(train_loader), 'val_batches': len(val_loader), 'test_batches': len(test_loader)}
    checkpoint['CONFIG'] = config
    checkpoint = {**checkpoint, **trainer_add_info, **subset_files, **loader_sizes}

    # Create unique identifier for model name
    model_name_id = f'{trainer.model.__class__.__name__}_{datetime.now().strftime("%y%m%d_%H%M%S")}'
    model_destination_path = Path(pth_folder, model_name_id + ".pth")

    # Save checkpoint
    torch.save(checkpoint, model_destination_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    # Check saved object size
    print(f"Model saved to:\t {model_destination_path}\n{'-'*60}\nSize: {os.path.getsize(model_destination_path) / 1024**2:.2f} MB\n{'-'*60}")
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



def load_checkpoint(model_destination_path, model, optimizer, DEVICE, GPU_SELECT):
    try: 
        checkpoint = torch.load(model_destination_path, weights_only=False, \
            map_location=DEVICE if (torch.cuda.is_available() and GPU_SELECT is not None) else torch.device('cpu'))
    except NotImplementedError:
        import pathlib
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        # temporarily change PosixPath to WindowsPath for loading
        checkpoint = torch.load(model_destination_path, weights_only=False, map_location=DEVICE if torch.cuda.is_available() else torch.device('cpu'))
        pathlib.PosixPath = temp

    for key in checkpoint.keys(): globals()[key] = checkpoint[key]

    # configure model and optimizer:
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)

    model.eval()  # set model to evaluation mode for inference
    print(f"Model loaded from:\t{model_destination_path}\n{'-'*60}\nModel: {model.__class__.__name__}\tParameters on device: {next(model.parameters()).device}"
            f"\n{'-'*60}\nTrain/Batch size:\t{len(train_loader.dataset)} / {train_loader.batch_size}\n"
            f"Loss:\t\t\t{loss_fn}\nOptimizer:\t\t{optimizer.__class__.__name__}\nLR:\t\t\t"
            f"{optimizer.param_groups[0]['lr']}\nWeight Decay:\t\t{optimizer.param_groups[0]['weight_decay']}\n{'-'*60}\n", model)

    return model, optimizer, checkpoint

    

def plot_training_performance(training_df, train_losses_per_iter, train_losses, val_losses, lr_history, train_batches):
    NUM_EPOCHS = CONFIG['NUM_EPOCHS']
    # plot training performance:
    fig, ax1 = plt.subplots(figsize=(14,4))
    ax1.set_xlabel('Epochs')
    ax1.set_xticks(range(1, NUM_EPOCHS + 1))

    assert len(train_losses_per_iter) == NUM_EPOCHS * train_batches, "Length of train_losses_per_iter might not match the number of iterations."
    ax1.plot(np.linspace(1, NUM_EPOCHS, len(train_losses_per_iter)), train_losses_per_iter, label='batch_loss', color='lightblue')
    ax1.plot(range(1, NUM_EPOCHS + 1), train_losses, label='train_loss', color='blue')
    ax1.plot(range(1, NUM_EPOCHS + 1), val_losses, label='val_loss', color='red')

    ax1.set_yscale('log')
    fig.tight_layout()
    ax1.legend()

    ax1.text(0.86, 0.6, f"Train: {train_losses[-1]:.3e}\nVal:    {val_losses[-1]:.3e}", \
        transform=ax1.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    if pd.Series(lr_history).nunique() > 1:
        ax2 = ax1.twinx()
        ax2.plot(range(1, NUM_EPOCHS + 1), lr_history, label='lr', color='green', linestyle='--')
        ax2.set_ylabel('Learning Rate', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_yscale('log')