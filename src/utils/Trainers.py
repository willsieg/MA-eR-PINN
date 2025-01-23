import os, time, math, torch, random
import pandas as pd
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
#from torchrl.trainers import Trainer as TorchRLTrainer
from tqdm.notebook import tqdm as tqdm_nb
from tqdm import tqdm as tqdm
from IPython.display import HTML, display_html
from tabulate import tabulate
if torch.cuda.is_available(): from torch.amp import GradScaler, autocast
torch.set_default_dtype(torch.float32); torch.set_printoptions(precision=6, sci_mode=True)
SEED = 42; random.seed(SEED); torch.manual_seed(SEED)



#############################################################################################################
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose: print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0
            #self.patience -= 1  # Reduce patience
            if self.verbose: print(f'Validation loss improved. Reducing patience to {self.patience}')


#############################################################################################################
#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
class PTrainer_PINN():

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, 
                 train_loader: DataLoader, num_epochs: int, device: torch.device, 
                 is_notebook: bool = False, val_loader: DataLoader = None, test_loader: DataLoader = None, 
                 lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None, l_p_scheduler = None,
                 state: dict = None, use_mixed_precision: bool = False, clip_value = None, log_file = "latest_run.txt", config = None):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn_pinn = loss_fn
        self.train_loader = train_loader
        self.num_epochs = num_epochs
        self.device = device
        self.is_notebook = is_notebook
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.l_p_scheduler = l_p_scheduler
        self.l_p = None
        self.state = state
        self.use_mixed_precision = use_mixed_precision if torch.cuda.is_available() else False
        if self.use_mixed_precision: self.scaler = GradScaler('cuda')  # Initialize GradScaler
        self.clip_value = clip_value
        self.log_file = log_file if log_file is not None else None
        self.config = config

        # Early Stopping:
        self.early_stopping = EarlyStopping(patience=8, verbose=True)

        # Redirect print statements to a log file
        if self.log_file: self.log = open(self.log_file, "w")

    def print_and_log(self, *args, **kwargs):
        print(*args, **kwargs)
        if self.log_file: print(*args, **kwargs, file=self.log)

    def only_log(self, *args, **kwargs):
        if self.log_file: print(*args, **kwargs, file=self.log)

    # EVALUATION ROUTINE DEFINITION -----------------------------------------------------------------
    def evaluate_model(self) -> tuple:
        if self.test_loader is None: self.print_and_log("No test data available."); return None
        else:
            self.model.eval()  # Set model to evaluation mode
            test_loss = 0.0
            all_outputs, all_targets, all_priors, all_original_lengths = [], [], [], []

            with torch.no_grad():
                for inputs, targets, priors, original_lengths in self.test_loader:
                    inputs, targets, priors = inputs.to(self.device), targets.to(self.device), priors.to(self.device)
                    # -------------------------------------
                    if self.use_mixed_precision:
                        with autocast(device_type='cuda'):
                            outputs = self.model(inputs)  # inputs are packed, outputs are not ! --> see forward method in model
                            outputs = outputs.squeeze()
                            mask = torch.arange(outputs.size(1))[None, :] < original_lengths[:, None]
                            outputs_masked = outputs[mask]
                            targets_masked = targets[mask]
                            priors_masked = priors[mask]
                            loss = self.loss_fn_pinn(outputs_masked.squeeze(), targets_masked, priors_masked, self.l_p)
                            test_loss += loss.item()
                    else:
                        outputs = self.model(inputs)  # inputs are packed, outputs are not ! --> see forward method in model
                        outputs = outputs.squeeze()
                        mask = torch.arange(outputs.size(1))[None, :] < original_lengths[:, None]
                        outputs_masked = outputs[mask]
                        targets_masked = targets[mask]
                        priors_masked = priors[mask]
                        loss = self.loss_fn_pinn(outputs_masked.squeeze(), targets_masked, priors_masked, self.l_p)
                        test_loss += loss.item()
                    # -------------------------------------
                    # Detach tensors from the computation graph and move them to CPU
                    outputs = outputs.detach().cpu().numpy()
                    targets = targets.detach().cpu().numpy()
                    priors = priors.detach().cpu().numpy()
                    original_lengths = original_lengths.detach().cpu().numpy()
                    # -------------------------------------
                    # Remove the padded endings of each sequence and restore their original lengths
                    # Collect all outputs and targets
                    for seq_output, seq_target, seq_prior, seq_length in zip(outputs, targets, priors, original_lengths):
                        all_outputs.append(seq_output[:seq_length])
                        all_targets.append(seq_target[:seq_length])
                        all_priors.append(seq_prior[:seq_length])
                        all_original_lengths.append(seq_length)
            # -------------------------------------
            test_loss /= len(self.test_loader)  # Calculate average test loss
            return test_loss, all_outputs, all_targets, all_priors, all_original_lengths

    # VALIDATION ROUTINE DEFINITION -----------------------------------------------------------------
    def validate_model(self, epoch: int) -> float:
        if self.val_loader is None: self.print_and_log("No validation data available."); return None
        else:
            self.model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():  # Disable gradient calculation
                for inputs, targets, priors, original_lengths in self.val_loader:
                    inputs, targets, priors = inputs.to(self.device), targets.to(self.device), priors.to(self.device)

                    # -------------------------------------
                    if self.use_mixed_precision:
                        with autocast(device_type='cuda'):
                            outputs = self.model(inputs)  # inputs are packed, outputs are not ! --> see forward method in model
                            outputs = outputs.squeeze()
                            mask = torch.arange(outputs.size(1))[None, :] < original_lengths[:, None]
                            outputs = outputs[mask]
                            targets = targets[mask]
                            priors = priors[mask]
                            loss = self.loss_fn_pinn(outputs.squeeze(), targets, priors, self.l_p)
                            val_loss += loss.item()
                    else:
                        outputs = self.model(inputs)  # inputs are packed, outputs are not ! --> see forward method in model
                        outputs = outputs.squeeze()
                        mask = torch.arange(outputs.size(1))[None, :] < original_lengths[:, None]
                        outputs = outputs[mask]
                        targets = targets[mask]
                        priors = priors[mask]
                        loss = self.loss_fn_pinn(outputs.squeeze(), targets, priors, self.l_p)
                        val_loss += loss.item()
            # -------------------------------------          
            val_loss /= len(self.val_loader)  # Calculate average validation loss
            if self.lr_scheduler:
                lr1 = self.lr_scheduler.get_last_lr()[0]
                self.lr_scheduler.step()  # Adjust learning rate based on validation loss
                lr2 = self.lr_scheduler.get_last_lr()[0]
                self.lr_history.append(lr2)
                if lr1 != lr2: self.print_and_log(f"Learning rate updated after epoch {epoch}: {lr1} -> {lr2}")
            return val_loss

    # TRAINING ROUTINE DEFINITION -----------------------------------------------------------------
    def train_model(self) -> dict:

        # output info on training process
        self.print_and_log(f"{'-'*60}\nTraining Started.\tProcess ID: {os.getpid()} \n{'-'*60}\n"
                      f"Model: {self.model.__class__.__name__}\tParameters on device: {str(next(self.model.parameters()).device).upper()}\n{'-'*60}\n"
                      f"Train/Batch size:\t{len(self.train_loader.dataset)} / {len(self.train_loader)}\n"
                      f"Loss:\t\t\t{self.loss_fn_pinn}\nOptimizer:\t\t{self.optimizer.__class__.__name__}\nLR:\t\t\t"
                      f"{self.optimizer.param_groups[0]['lr']}\nWeight Decay:\t\t{self.optimizer.param_groups[0]['weight_decay']}\n{'-'*60}")
        
        # Load state dict if provided
        start_epoch = 1
        if self.state:
            self.model.load_state_dict(self.state['model_state_dict'])
            self.optimizer.load_state_dict(self.state['optimizer_state_dict'])
            self.train_losses = self.state['train_losses']
            self.train_losses_per_iter = self.state['train_losses_per_iter']
            self.lr_history = self.state['lr_history']
            self.l_p_history = self.state['l_p_history']
            self.val_losses = self.state['val_losses']
            self.training_table = self.state['training_table']
            start_epoch = self.state['epoch'] + 1
        else:
            self.train_losses = [] 
            self.train_losses_per_iter = []
            self.val_losses = []  
            self.training_table = []  
            self.lr_history = []
            self.l_p_history = []
            if self.is_notebook: display_html(HTML(initialize_table()))

        # TRAINING LOOP:
        # ---------------------------------------------------------------------
        start_time = time.perf_counter()
        for epoch in range(start_epoch, self.num_epochs + 1):
            self.model.train()  # set model to training mode
            running_loss = 0.0
            num_iterations = math.ceil(len(self.train_loader.dataset) / self.train_loader.batch_size)
            header_printed = False
            if not header_printed: self.only_log(f"\n{'-'*60}\n{'Epoch':<14}{'Iteration':<14}{'Batch Loss':<16}{'Train Loss':<14}\n{'-'*60}")

            tqdm_version = tqdm_nb if self.is_notebook else tqdm
            with tqdm_version(enumerate(self.train_loader, 1), unit="batch", total=num_iterations, leave=False) as tepoch:
                for iter, (inputs, targets, priors, original_lengths) in tepoch:  # ----> note: (packed_inputs, padded_targets, padded_priors, lengths)
                    tepoch.set_description(f"Epoch {epoch}/{self.num_epochs}")
                    # -------------------------------------------------------------
                    # Move data to the GPU
                    inputs, targets, priors = inputs.to(self.device), targets.to(self.device), priors.to(self.device)
                
                    # zero gradients -> forward pass -> obtain loss function -> apply backpropagation -> update weights:
                    self.optimizer.zero_grad()

                    # A) use mixed precision calculation ------------------------------------------------------------------------------
                    if self.use_mixed_precision:
                        with autocast(device_type='cuda'):  # Enable autocast for mixed precision training
                            outputs = self.model(inputs)   # inputs are packed, outputs are not ! --> see forward method in model
                            outputs = outputs.squeeze()
                            mask = torch.arange(outputs.size(1))[None, :] < original_lengths[:, None]
                            outputs = outputs[mask]
                            targets = targets[mask]
                            priors = priors[mask]
                            self.l_p = self.l_p_scheduler.get_value((epoch-1)*num_iterations + iter)
                            self.l_p_history.append(self.l_p)
                            loss = self.loss_fn_pinn(outputs.squeeze(), targets, priors, self.l_p)
                            

                        self.scaler.scale(loss).backward()  # Scale the loss and perform backward pass
                        if self.clip_value is not None:
                            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_value)  # optional: Gradient Value Clipping
                        self.scaler.step(self.optimizer)  # Update model parameters
                        self.scaler.update()  # Update the scale for next iteration

                    # B) Normal precision calculation ------------------------------------------------------------------------------
                    else:
                        outputs = self.model(inputs)  # inputs are packed, outputs are not ! --> see forward method in model
                        # -------------------------------------
                        outputs = outputs.squeeze()
                        mask = torch.arange(outputs.size(1))[None, :] < original_lengths[:, None]

                        outputs = outputs[mask]
                        targets = targets[mask]
                        priors = priors[mask]
                        # -------------------------------------
                        self.l_p = self.l_p_scheduler.get_value((epoch-1)*num_iterations + iter)
                        self.l_p_history.append(self.l_p)
                        loss = self.loss_fn_pinn(outputs.squeeze(), targets, priors, self.l_p)
                        loss.backward()
                        if self.clip_value is not None:
                            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_value)  # optional: Gradient Value Clipping
                        self.optimizer.step()

                    # -------------------------------------------------------------
                    # Update the performance table
                    if iter % (num_iterations // 4) == 0 and iter != num_iterations // 4 * 4 and False:
                        add_row(self.training_table, f" ", f"{iter}", f"{loss.item():.6f}", " ")
                        self.only_log(f"{epoch:<14}{iter:<14}{loss.item():.6f}")
                        if self.is_notebook:
                            display_html(HTML(f"""<script>addRow("", "{iter}", "{loss.item():.6f}", "");</script>"""))
                        else:
                            print_row(self.training_table)
                    elif iter == 1:
                        add_row(self.training_table, f"{epoch}/{self.num_epochs}", f"{iter}/{num_iterations}", f"{loss.item():.6f}", " ")
                        self.only_log(f"{epoch:<14}{iter:<14}{loss.item():.6f}")
                        if self.is_notebook:
                            display_html(HTML(f"""<script>addRow("<b>{epoch}/{self.num_epochs}", "{iter}/{num_iterations}", "{loss.item():.6f}", "");</script>"""))
                        else:
                            print_row(self.training_table)

                    # -------------------------------------------------------------
                    # Update running loss and progress bar
                    self.train_losses_per_iter.append(loss.item())
                    running_loss += loss.item()  # accumulate loss for epoch
                    tepoch.set_postfix(loss=loss.item()); tepoch.update(1)

            # Calculate average training loss for the epoch
            avg_train_loss = running_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)

            # Update the performance table
            add_row(self.training_table, f" ", f"{iter}", f"{loss.item():.6f}", f"{avg_train_loss:.6f}")
            self.only_log(f"{epoch:<14}{iter:<14}{loss.item():.6f}\t\t{avg_train_loss:.6f}")
            if self.is_notebook:
                display_html(HTML(f"""<script>addRow("", "{iter}", "{loss.item():.6f}", "<b>{avg_train_loss:.6f}");</script>"""))
            else:
                print_row(self.training_table)

            # VALIDATION
            if self.val_loader:
                val_loss = self.validate_model(epoch)
                self.val_losses.append(val_loss)
                # Update the performance table
                add_row(self.training_table, f"Val", f"Validation Loss:", f"{val_loss:.6f}", "")
                self.only_log(f"\n{'Val':<14}{'Validation Loss:':<14}\t\t\t\t{val_loss:.6f}")
                if self.is_notebook:
                    display_html(HTML(f"""<script>addRow("<b>Val", "Validation Loss:", "<b>{val_loss:.6f}", "");</script>"""))
                else:
                    print_row(self.training_table)

            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f"Early stopping.\n{'-'*60}\n")
                break

        elapsed_time = round(time.perf_counter() - start_time)
        self.print_and_log(f"{'-'*60}\nTraining Completed.\tExecution Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}\n{'-'*60}\n")

        if self.config is not None:
                config_df = pd.DataFrame(list(self.config.items()), columns=['Parameter', 'Value'])
                config_df['Value'] = config_df['Value'].apply(lambda x: str(x).replace(',', ',\n') if len(str(x)) > 120 else str(x))
                self.only_log(f"CONFIG Dictionary:\n{'-'*129}\n", tabulate(config_df, headers='keys', colalign=("left", "left"), \
                    maxcolwidths=[30, 120]), f"\n{'-'*129}\n")
        if self.log_file: self.log.close()


        return {
            # model and optimizer states
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),

            # training performance
            "training_table": self.training_table,
            "train_losses": self.train_losses,
            "train_losses_per_iter": list(self.train_losses_per_iter),
            "val_losses": self.val_losses,
            'lr_history': self.lr_history,
            'l_p_history': self.l_p_history,
            
            # settings and meta data
            "loss_fn": self.loss_fn_pinn,
            "epoch": epoch,
            "elapsed_train_time": elapsed_time,
            "log_file": self.log_file
        }


#############################################################################################################
#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
class PTrainer_Standard():

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, 
                 train_loader: DataLoader, num_epochs: int, device: torch.device, 
                 is_notebook: bool = False, val_loader: DataLoader = None, test_loader: DataLoader = None, 
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None, state: dict = None, 
                 use_mixed_precision: bool = False, clip_value = None, log_file = "current_run.txt"):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.num_epochs = num_epochs
        self.device = device
        self.is_notebook = is_notebook
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.state = state
        self.use_mixed_precision = use_mixed_precision if torch.cuda.is_available() else False
        if self.use_mixed_precision: self.scaler = GradScaler('cuda')  # Initialize GradScaler
        self.clip_value = clip_value
        self.log_file = log_file if log_file is not None else None

        # Redirect print statements to a log file
        if self.log_file: self.log = open(self.log_file, "w")

    def print_and_log(self, *args, **kwargs):
        print(*args, **kwargs)
        if self.log_file: print(*args, **kwargs, file=self.log)

    def only_log(self, *args, **kwargs):
        if self.log_file: print(*args, **kwargs, file=self.log)

    # EVALUATION ROUTINE DEFINITION -----------------------------------------------------------------
    def evaluate_model(self) -> tuple:
        if self.test_loader is None: self.print_and_log("No test data available."); return None
        else:
            self.model.eval()  # Set model to evaluation mode
            test_loss = 0.0
            all_outputs, all_targets, all_original_lengths = [], [], []

            with torch.no_grad():
                for inputs, targets, original_lengths in self.test_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    # -------------------------------------
                    if self.use_mixed_precision:
                        with autocast(device_type='cuda'):
                            outputs = self.model(inputs)  # inputs are packed, outputs are not ! --> see forward method in model
                            outputs = outputs.squeeze()
                            mask = torch.arange(outputs.size(1))[None, :] < original_lengths[:, None]
                            outputs_masked = outputs[mask]
                            targets_masked = targets[mask]
                            loss = self.loss_fn(outputs_masked.squeeze(), targets_masked)
                            test_loss += loss.item()
                    else:
                        outputs = self.model(inputs)  # inputs are packed, outputs are not ! --> see forward method in model
                        outputs = outputs.squeeze()
                        mask = torch.arange(outputs.size(1))[None, :] < original_lengths[:, None]
                        outputs_masked = outputs[mask]
                        targets_masked = targets[mask]
                        loss = self.loss_fn(outputs_masked.squeeze(), targets_masked)
                        test_loss += loss.item()
                    # -------------------------------------
                    # Detach tensors from the computation graph and move them to CPU
                    outputs = outputs.detach().cpu().numpy()
                    targets = targets.detach().cpu().numpy()
                    original_lengths = original_lengths.detach().cpu().numpy()
                    # -------------------------------------
                    # Remove the padded endings of each sequence and restore their original lengths
                    # Collect all outputs and targets
                    for seq_output, seq_target, seq_prior, seq_length in zip(outputs, targets, priors, original_lengths):
                        all_outputs.append(seq_output[:seq_length])
                        all_targets.append(seq_target[:seq_length])
                        all_original_lengths.append(seq_length)
            # -------------------------------------
            test_loss /= len(self.test_loader)  # Calculate average test loss
            return test_loss, all_outputs, all_targets, all_original_lengths

    # VALIDATION ROUTINE DEFINITION -----------------------------------------------------------------
    def validate_model(self, epoch: int) -> float:
        if self.val_loader is None: self.print_and_log("No validation data available."); return None
        else:
            self.model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():  # Disable gradient calculation
                for inputs, targets, original_lengths in self.val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    # -------------------------------------
                    if self.use_mixed_precision:
                        with autocast(device_type='cuda'):
                            outputs = self.model(inputs)  # inputs are packed, outputs are not ! --> see forward method in model
                            outputs = outputs.squeeze()
                            mask = torch.arange(outputs.size(1))[None, :] < original_lengths[:, None]
                            outputs = outputs[mask]
                            targets = targets[mask]
                            loss = self.loss_fn(outputs.squeeze(), targets)
                            val_loss += loss.item()
                    else:
                        outputs = self.model(inputs)  # inputs are packed, outputs are not ! --> see forward method in model
                        outputs = outputs.squeeze()
                        mask = torch.arange(outputs.size(1))[None, :] < original_lengths[:, None]
                        outputs = outputs[mask]
                        targets = targets[mask]
                        loss = self.loss_fn(outputs.squeeze(), targets)
                        val_loss += loss.item()
            # -------------------------------------          
            val_loss /= len(self.val_loader)  # Calculate average validation loss
            if self.scheduler:
                lr1 = self.scheduler.get_last_lr()[0]
                self.scheduler.step(val_loss)  # Adjust learning rate based on validation loss
                lr2 = self.scheduler.get_last_lr()[0]
                self.lr_history.append(lr2)
                if lr1 != lr2: self.print_and_log(f"Learning rate updated after epoch {epoch}: {lr1} -> {lr2}")
            return val_loss

    # TRAINING ROUTINE DEFINITION -----------------------------------------------------------------
    def train_model(self) -> dict:

        # output info on training process
        self.print_and_log(f"{'-'*60}\nTraining Started.\tProcess ID: {os.getpid()} \n{'-'*60}\n"
                      f"Model: {self.model.__class__.__name__}\tParameters on device: {str(next(self.model.parameters()).device).upper()}\n{'-'*60}\n"
                      f"Train/Batch size:\t{len(self.train_loader.dataset)} / {self.train_loader.batch_size}\n"
                      f"Loss:\t\t\t{self.loss_fn}\nOptimizer:\t\t{self.optimizer.__class__.__name__}\nLR:\t\t\t"
                      f"{self.optimizer.param_groups[0]['lr']}\nWeight Decay:\t\t{self.optimizer.param_groups[0]['weight_decay']}\n{'-'*60}")
        
        # Load state dict if provided
        start_epoch = 1
        if self.state:
            self.model.load_state_dict(self.state['model_state_dict'])
            self.optimizer.load_state_dict(self.state['optimizer_state_dict'])
            self.train_losses = self.state['train_losses']
            self.train_losses_per_iter = self.state['train_losses_per_iter']
            self.lr_history = self.state['lr_history']
            self.val_losses = self.state['val_losses']
            self.training_table = self.state['training_table']
            start_epoch = self.state['epoch'] + 1
        else:
            self.train_losses = [] 
            self.train_losses_per_iter = []
            self.val_losses = []  
            self.training_table = []  
            self.lr_history = []
            if self.is_notebook: display_html(HTML(initialize_table()))

        # TRAINING LOOP:
        # ---------------------------------------------------------------------
        start_time = time.perf_counter()
        for epoch in range(start_epoch, self.num_epochs + 1):
            self.model.train()  # set model to training mode
            running_loss = 0.0
            num_iterations = math.ceil(len(self.train_loader.dataset) / self.train_loader.batch_size)
            header_printed = False
            if not header_printed: self.only_log(f"\n{'-'*60}\n{'Epoch':<14}{'Iteration':<14}{'Batch Loss':<16}{'Train Loss':<14}\n{'-'*60}")

            tqdm_version = tqdm_nb if self.is_notebook else tqdm
            with tqdm_version(enumerate(self.train_loader, 1), unit="batch", total=num_iterations, leave=False) as tepoch:
                for iter, (inputs, targets, original_lengths) in tepoch:  # ----> note: (packed_inputs, padded_targets, lengths)
                    tepoch.set_description(f"Epoch {epoch}/{self.num_epochs}")
                    #print("Dataloader: ", type(inputs), type(targets), type(original_lengths))
                    #print(f"Shape of inputs: {inputs.data.shape}, {type(inputs)}")
                    #print(f"Shape of targets: {targets.shape}, {type(targets)}")
                    #print(f"Shape of original_lengths: {original_lengths.shape}, {type(original_lengths)}")
                    # -------------------------------------------------------------
                    # Move data to the GPU
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                

                    # zero gradients -> forward pass -> obtain loss function -> apply backpropagation -> update weights:
                    self.optimizer.zero_grad()

                    # A) use mixed precision calculation ------------------------------------------------------------------------------
                    if self.use_mixed_precision:
                        with autocast(device_type='cuda'):  # Enable autocast for mixed precision training
                            outputs = self.model(inputs)   # inputs are packed, outputs are not ! --> see forward method in model
                            outputs = outputs.squeeze()
                            mask = torch.arange(outputs.size(1))[None, :] < original_lengths[:, None]
                            outputs = outputs[mask]
                            targets = targets[mask]
                            loss = self.loss_fn(outputs.squeeze(), targets)
                        self.scaler.scale(loss).backward()  # Scale the loss and perform backward pass
                        if self.clip_value is not None:
                            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_value)  # optional: Gradient Value Clipping
                        self.scaler.step(self.optimizer)  # Update model parameters
                        self.scaler.update()  # Update the scale for next iteration

                    # B) Normal precision calculation ------------------------------------------------------------------------------
                    else:
                        #print("Forwarding.")
                        outputs = self.model(inputs)  # inputs are packed, outputs are not ! --> see forward method in model

                        # -------------------------------------
                        #print(f"Shape of outputs: {outputs.shape}, {type(outputs)}")
                        outputs = outputs.squeeze()
                        mask = torch.arange(outputs.size(1))[None, :] < original_lengths[:, None]

                        #print(f"Shape of mask: {mask.shape}, {type(mask)}")
                        #print(f"Masking.")
                        outputs = outputs[mask]
                        targets = targets[mask]
                        # -------------------------------------
                        #print(f"Shape of outputs after mask: {outputs.shape}, {type(outputs)}")
                        #print(f"Shape of targets after mask: {targets.shape}, {type(targets)}")

                        loss = self.loss_fn(outputs.squeeze(), targets)
                        loss.backward()
                        if self.clip_value is not None:
                            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_value)  # optional: Gradient Value Clipping
                        self.optimizer.step()

                    # -------------------------------------------------------------
                    # Update the performance table
                    if iter % (num_iterations // 4) == 0 and iter != num_iterations // 4 * 4:
                        add_row(self.training_table, f" ", f"{iter}", f"{loss.item():.6f}", " ")
                        self.only_log(f"{epoch:<14}{iter:<14}{loss.item():.6f}")
                        if self.is_notebook:
                            display_html(HTML(f"""<script>addRow("", "{iter}", "{loss.item():.6f}", "");</script>"""))
                        else:
                            print_row(self.training_table)
                    elif iter == 1:
                        add_row(self.training_table, f"{epoch}/{self.num_epochs}", f"{iter}/{num_iterations}", f"{loss.item():.6f}", " ")
                        self.only_log(f"{epoch:<14}{iter:<14}{loss.item():.6f}")
                        if self.is_notebook:
                            display_html(HTML(f"""<script>addRow("<b>{epoch}/{self.num_epochs}", "{iter}/{num_iterations}", "{loss.item():.6f}", "");</script>"""))
                        else:
                            print_row(self.training_table)

                    # -------------------------------------------------------------
                    # Update running loss and progress bar
                    self.train_losses_per_iter.append(loss.item())
                    running_loss += loss.item()  # accumulate loss for epoch
                    tepoch.set_postfix(loss=loss.item()); tepoch.update(1)

            # Calculate average training loss for the epoch
            avg_train_loss = running_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)

            # Update the performance table
            add_row(self.training_table, f" ", f"{iter}", f"{loss.item():.6f}", f"{avg_train_loss:.6f}")
            self.only_log(f"{epoch:<14}{iter:<14}{loss.item():.6f}\t\t{avg_train_loss:.6f}")
            if self.is_notebook:
                display_html(HTML(f"""<script>addRow("", "{iter}", "{loss.item():.6f}", "<b>{avg_train_loss:.6f}");</script>"""))
            else:
                print_row(self.training_table)

            # VALIDATION
            if self.val_loader:
                val_loss = self.validate_model(epoch)
                self.val_losses.append(val_loss)
                # Update the performance table
                add_row(self.training_table, f"Val", f"Validation Loss:", f"{val_loss:.6f}", "")
                self.only_log(f"\n{'Val':<14}{'Validation Loss:':<14}\t\t\t\t{val_loss:.6f}")
                if self.is_notebook:
                    display_html(HTML(f"""<script>addRow("<b>Val", "Validation Loss:", "<b>{val_loss:.6f}", "");</script>"""))
                else:
                    print_row(self.training_table)

        elapsed_time = round(time.perf_counter() - start_time)
        self.print_and_log(f"{'-'*60}\nTraining Completed.\tExecution Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}\n{'-'*60}\n")
        if self.log_file: self.log.close()
        return {
            # model and optimizer states
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),

            # training performance
            "training_table": self.training_table,
            "train_losses": self.train_losses,
            "train_losses_per_iter": list(self.train_losses_per_iter),
            "val_losses": self.val_losses,
            'lr_history': self.lr_history,
            
            # settings and meta data
            "loss_fn": self.loss_fn,
            "epoch": epoch,
            "elapsed_train_time": elapsed_time
        }


#############################################################################################################
#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# Note: use this class with LSTM class, that takes and returns hidden and cell tensors explicitly!
class PTrainer_keep_hidden_cell():

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, 
                 train_loader: DataLoader, num_epochs: int, device: torch.device, 
                 is_notebook: bool = False, val_loader: DataLoader = None, test_loader: DataLoader = None, 
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None, state: dict = None, 
                 use_mixed_precision: bool = False, clip_value = None, log_file = None):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.num_epochs = num_epochs
        self.device = device
        self.is_notebook = is_notebook
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.state = state
        self.use_mixed_precision = use_mixed_precision if torch.cuda.is_available() else False
        if self.use_mixed_precision: self.scaler = GradScaler('cuda')  # Initialize GradScaler
        self.clip_value = clip_value
        self.log_file = log_file if log_file is not None else None

        # Redirect print statements to a log file
        if self.log_file: self.log = open(self.log_file, "w")

    def print_and_log(self, *args, **kwargs):
        print(*args, **kwargs)
        if self.log_file: print(*args, **kwargs, file=self.log)

    def only_log(self, *args, **kwargs):
        if self.log_file: print(*args, **kwargs, file=self.log)

    # EVALUATION ROUTINE DEFINITION -----------------------------------------------------------------
    def evaluate_model(self) -> tuple:
        if self.test_loader is None: self.print_and_log("No test data available."); return None
        else:
            self.model.eval()  # Set model to evaluation mode
            test_loss = 0.0
            all_outputs, all_targets, all_original_lengths = [], [], []

            with torch.no_grad():
                for inputs, targets, original_lengths in self.test_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    # -------------------------------------
                    if self.use_mixed_precision:
                        with autocast(device_type='cuda'):
                            outputs, (_, _) = self.model(inputs)  # inputs are packed, outputs are not ! --> see forward method in model
                            outputs = outputs.squeeze()
                            mask = torch.arange(outputs.size(1))[None, :] < original_lengths[:, None]
                            outputs_masked = outputs[mask]
                            targets_masked = targets[mask]
                            loss = self.loss_fn(outputs_masked.squeeze(), targets_masked).mean()
                            test_loss += loss.item()
                    else:
                        outputs, (_, _) = self.model(inputs)  # inputs are packed, outputs are not ! --> see forward method in model
                        outputs = outputs.squeeze()
                        mask = torch.arange(outputs.size(1))[None, :] < original_lengths[:, None]
                        outputs_masked = outputs[mask]
                        targets_masked = targets[mask]
                        loss = self.loss_fn(outputs_masked.squeeze(), targets_masked).mean()
                        test_loss += loss.item()
                    # -------------------------------------
                    # Detach tensors from the computation graph and move them to CPU
                    outputs = outputs.detach().cpu().numpy()
                    targets = targets.detach().cpu().numpy()
                    # Remove the padded endings of each sequence and restore their original lengths
                    unpadded_outputs = [output[:length] for output, length in zip(outputs, original_lengths)]
                    unpadded_targets = [target[:length] for target, length in zip(targets, original_lengths)]
                    # -------------------------------------
                    # Collect all outputs and targets
                    all_outputs.append(unpadded_outputs)
                    all_targets.append(unpadded_targets)
                    all_original_lengths.append(original_lengths.detach().cpu().numpy())
            # -------------------------------------
            test_loss /= len(self.test_loader)  # Calculate average test loss
            return test_loss, all_outputs, all_targets, all_original_lengths

    # VALIDATION ROUTINE DEFINITION -----------------------------------------------------------------
    def validate_model(self, epoch: int) -> float:
        if self.val_loader is None: self.print_and_log("No validation data available."); return None
        else:
            self.model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():  # Disable gradient calculation
                for inputs, targets, original_lengths in self.val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    # -------------------------------------
                    if self.use_mixed_precision:
                        with autocast(device_type='cuda'):
                            outputs, (_, _) = self.model(inputs)  # inputs are packed, outputs are not ! --> see forward method in model
                            outputs = outputs.squeeze()
                            mask = torch.arange(outputs.size(1))[None, :] < original_lengths[:, None]
                            outputs = outputs[mask]
                            targets = targets[mask]
                            loss = self.loss_fn(outputs.squeeze(), targets).mean()
                            val_loss += loss.item()
                    else:
                        outputs, (_, _) = self.model(inputs)  # inputs are packed, outputs are not ! --> see forward method in model
                        outputs = outputs.squeeze()
                        mask = torch.arange(outputs.size(1))[None, :] < original_lengths[:, None]
                        outputs = outputs[mask]
                        targets = targets[mask]
                        loss = self.loss_fn(outputs.squeeze(), targets).mean()
                        val_loss += loss.item()
            # -------------------------------------          
            val_loss /= len(self.val_loader)  # Calculate average validation loss
            if self.scheduler:
                lr1 = self.scheduler.get_last_lr()[0]
                self.scheduler.step(val_loss)  # Adjust learning rate based on validation loss
                lr2 = self.scheduler.get_last_lr()[0]
                self.lr_history.append(lr2)
                if lr1 != lr2: self.print_and_log(f"Learning rate updated after epoch {epoch}: {lr1} -> {lr2}")
            return val_loss

    # TRAINING ROUTINE DEFINITION -----------------------------------------------------------------
    def train_model(self) -> dict:

        # output info on training process
        self.print_and_log(f"{'-'*60}\nTraining Started.\tProcess ID: {os.getpid()} \n{'-'*60}\n"
                      f"Model: {self.model.__class__.__name__}\tParameters on device: {str(next(self.model.parameters()).device).upper()}\n{'-'*60}\n"
                      f"Train/Batch size:\t{len(self.train_loader.dataset)} / {self.train_loader.batch_size}\n"
                      f"Loss:\t\t\t{self.loss_fn}\nOptimizer:\t\t{self.optimizer.__class__.__name__}\nLR:\t\t\t"
                      f"{self.optimizer.param_groups[0]['lr']}\nWeight Decay:\t\t{self.optimizer.param_groups[0]['weight_decay']}\n{'-'*60}")
        
        # Load state dict if provided
        start_epoch = 1
        if self.state:
            self.model.load_state_dict(self.state['model_state_dict'])
            self.optimizer.load_state_dict(self.state['optimizer_state_dict'])
            self.train_losses = self.state['train_losses']
            self.train_losses_per_iter = self.state['train_losses_per_iter']
            self.lr_history = self.state['lr_history']
            self.val_losses = self.state['val_losses']
            self.training_table = self.state['training_table']
            start_epoch = self.state['epoch'] + 1
        else:
            self.train_losses = [] 
            self.train_losses_per_iter = []
            self.val_losses = []  
            self.training_table = []  
            self.lr_history = []
            if self.is_notebook: display_html(HTML(initialize_table()))

        # TRAINING LOOP:
        # ---------------------------------------------------------------------
        start_time = time.perf_counter()
        for epoch in range(start_epoch, self.num_epochs + 1):
            self.model.train()  # set model to training mode
            running_loss = 0.0
            num_iterations = math.ceil(len(self.train_loader.dataset) / self.train_loader.batch_size)
            header_printed = False
            if not header_printed: self.only_log(f"\n{'-'*60}\n{'Epoch':<14}{'Iteration':<14}{'Batch Loss':<16}{'Train Loss':<14}\n{'-'*60}")

            tqdm_version = tqdm_nb if self.is_notebook else tqdm
            with tqdm_version(enumerate(self.train_loader, 1), unit="batch", total=num_iterations, leave=False) as tepoch:
                for iter, (inputs, targets, original_lengths) in tepoch:  # ----> note: (packed_inputs, padded_targets, lengths)
                    tepoch.set_description(f"Epoch {epoch}/{self.num_epochs}")
                    # -------------------------------------------------------------
                    # Move data to the GPU
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    # Initialize hidden and cell states at the beginning of each epoch
                    if iter == 1:
                        h_0 = torch.zeros(self.model.num_layers, inputs.batch_sizes[0].item(), self.model.hidden_size).to(self.device)
                        c_0 = torch.zeros(self.model.num_layers, inputs.batch_sizes[0].item(), self.model.hidden_size).to(self.device)

                    
                    # In case of a shorter last batch, Adjust the size of h_0 and c_0 if the batch size changes 
                    # (Warning: shortest batch must be last if it exist!)
                    elif h_0.size(1) != len(original_lengths): 
                        h_0 = torch.zeros(self.model.num_layers, len(original_lengths), self.model.hidden_size).to(self.device)
                        c_0 = torch.zeros(self.model.num_layers, len(original_lengths), self.model.hidden_size).to(self.device)

                    # zero gradients -> forward pass -> obtain loss function -> apply backpropagation -> update weights:
                    self.optimizer.zero_grad()

                    # A) use mixed precision calculation ------------------------------------------------------------------------------
                    if self.use_mixed_precision:
                        with autocast(device_type='cuda'):  # Enable autocast for mixed precision training
                            outputs, (h_0, c_0) = self.model(inputs, h_0, c_0)   # inputs are packed, outputs are not ! --> see forward method in model
                            h_0 = h_0.detach()
                            c_0 = c_0.detach()
                            outputs = outputs.squeeze()
                            mask = torch.arange(outputs.size(1))[None, :] < original_lengths[:, None]
                            outputs = outputs[mask]
                            targets = targets[mask]
                            loss = self.loss_fn(outputs.squeeze(), targets).mean()
                        self.scaler.scale(loss).backward()  # Scale the loss and perform backward pass
                        if self.clip_value is not None:
                            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_value)  # optional: Gradient Value Clipping
                        self.scaler.step(self.optimizer)  # Update model parameters
                        self.scaler.update()  # Update the scale for next iteration

                    # B) Normal precision calculation ------------------------------------------------------------------------------
                    else:
                        #print("Forwarding.")
                        outputs, (h_0, c_0) = self.model(inputs, h_0, c_0)  # inputs are packed, outputs are not ! --> see forward method in model
                        h_0 = h_0.detach()
                        c_0 = c_0.detach()
                        # -------------------------------------
                        #print(f"Shape of outputs: {outputs.shape}, {type(outputs)}")
                        outputs = outputs.squeeze()
                        mask = torch.arange(outputs.size(1))[None, :] < original_lengths[:, None]

                        #print(f"Shape of mask: {mask.shape}, {type(mask)}")
                        #print(f"Masking.")
                        outputs = outputs[mask]
                        targets = targets[mask]
                        # -------------------------------------
                        #print(f"Shape of outputs after mask: {outputs.shape}, {type(outputs)}")
                        #print(f"Shape of targets after mask: {targets.shape}, {type(targets)}")

                        loss = self.loss_fn(outputs.squeeze(), targets).mean()
                        loss.backward()
                        if self.clip_value is not None:
                            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_value)  # optional: Gradient Value Clipping
                        self.optimizer.step()

                    # -------------------------------------------------------------
                    # Update the performance table
                    if iter % (num_iterations // 4) == 0 and iter != num_iterations // 4 * 4:
                        add_row(self.training_table, f" ", f"{iter}", f"{loss.item():.6f}", " ")
                        self.only_log(f"{epoch:<14}{iter:<14}{loss.item():.6f}")
                        if self.is_notebook:
                            display_html(HTML(f"""<script>addRow("", "{iter}", "{loss.item():.6f}", "");</script>"""))
                        else:
                            print_row(self.training_table)
                    elif iter == 1:
                        add_row(self.training_table, f"{epoch}/{self.num_epochs}", f"{iter}/{num_iterations}", f"{loss.item():.6f}", " ")
                        self.only_log(f"{epoch:<14}{iter:<14}{loss.item():.6f}")
                        if self.is_notebook:
                            display_html(HTML(f"""<script>addRow("<b>{epoch}/{self.num_epochs}", "{iter}/{num_iterations}", "{loss.item():.6f}", "");</script>"""))
                        else:
                            print_row(self.training_table)

                    # -------------------------------------------------------------
                    # Update running loss and progress bar
                    self.train_losses_per_iter.append(loss.item())
                    running_loss += loss.item()  # accumulate loss for epoch
                    tepoch.set_postfix(loss=loss.item()); tepoch.update(1)

            # Calculate average training loss for the epoch
            avg_train_loss = running_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)

            # Update the performance table
            add_row(self.training_table, f" ", f"{iter}", f"{loss.item():.6f}", f"{avg_train_loss:.6f}")
            self.only_log(f"{epoch:<14}{iter:<14}{loss.item():.6f}\t\t{avg_train_loss:.6f}")
            if self.is_notebook:
                display_html(HTML(f"""<script>addRow("", "{iter}", "{loss.item():.6f}", "<b>{avg_train_loss:.6f}");</script>"""))
            else:
                print_row(self.training_table)

            # VALIDATION
            if self.val_loader:
                val_loss = self.validate_model(epoch)
                self.val_losses.append(val_loss)
                # Update the performance table
                add_row(self.training_table, f"Val", f"Validation Loss:", f"{val_loss:.6f}", "")
                self.only_log(f"\n{'Val':<14}{'Validation Loss:':<14}\t\t\t\t{val_loss:.6f}")
                if self.is_notebook:
                    display_html(HTML(f"""<script>addRow("<b>Val", "Validation Loss:", "<b>{val_loss:.6f}", "");</script>"""))
                else:
                    print_row(self.training_table)

        elapsed_time = round(time.perf_counter() - start_time)
        self.print_and_log(f"{'-'*60}\nTraining Completed.\tExecution Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}\n{'-'*60}\n")
        if self.log_file: self.log.close()
        return {
            # model and optimizer states
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),

            # training performance
            "training_table": self.training_table,
            "train_losses": self.train_losses,
            "train_losses_per_iter": list(self.train_losses_per_iter),
            "val_losses": self.val_losses,
            'lr_history': self.lr_history,
            
            # settings and meta data
            "loss_fn": self.loss_fn,
            "epoch": epoch,
            "elapsed_train_time": elapsed_time
        }



#############################################################################################################
#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
class Trainer_no_padding():

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, 
                 train_loader: DataLoader, num_epochs: int, device: torch.device, 
                 is_notebook: bool = False, val_loader: DataLoader = None, 
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None, state: dict = None, 
                 use_mixed_precision: bool = False, clip_value = None):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.num_epochs = num_epochs
        self.device = device
        self.is_notebook = is_notebook
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.state = state
        self.use_mixed_precision = use_mixed_precision if torch.cuda.is_available() else False
        if self.use_mixed_precision: self.scaler = GradScaler('cuda')  # Initialize GradScaler
        self.clip_value = clip_value

    def validate_model(self, epoch: int) -> float:
        self.model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient calculation
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs.squeeze(), targets)
                val_loss += loss.item()
        val_loss /= len(self.val_loader)  # Calculate average validation loss
        if self.scheduler:
            lr1 = self.scheduler.get_last_lr()[0]
            self.scheduler.step(val_loss)  # Adjust learning rate based on validation loss
            lr2 = self.scheduler.get_last_lr()[0]
            self.lr_history.append(lr2)
            if lr1 != lr2:
                self.print_and_log(f"Learning rate updated after epoch {epoch}: {lr1} -> {lr2}")
        return val_loss

    # TRAINING ROUTINE DEFINITION -----------------------------------------------------------------
    def train_model(self) -> dict:
       # output info on training process
        self.print_and_log(f"{'-'*60}\nTraining Started.\tProcess ID: {os.getpid()} \n{'-'*60}\n"
              f"Model: {self.model.__class__.__name__}\t\tParameters on device: {str(next(self.model.parameters()).device).upper()}\n{'-'*60}\n"
              f"Train/Batch size:\t{len(self.train_loader.dataset)} / {self.train_loader.batch_size}\n"
              f"Loss:\t\t\t{self.loss_fn}\nOptimizer:\t\t{self.optimizer.__class__.__name__}\nLR:\t\t\t"
              f"{self.optimizer.param_groups[0]['lr']}\nWeight Decay:\t\t{self.optimizer.param_groups[0]['weight_decay']}\n{'-'*60}")
        
        # Load state dict if provided
        start_epoch = 1
        if self.state:
            self.model.load_state_dict(self.state['model_state_dict'])
            self.optimizer.load_state_dict(self.state['optimizer_state_dict'])
            self.train_losses = self.state['train_losses']
            self.lr_history = self.state['lr_history']
            self.val_losses = self.state['val_losses']
            self.training_table = self.state['training_table']
            start_epoch = self.state['epoch'] + 1
        else:
            self.train_losses = [] 
            self.val_losses = []  
            self.training_table = []  
            self.lr_history = []
            if self.is_notebook: display_html(HTML(initialize_table()))

        # TRAINING LOOP:
        start_time = time.perf_counter()
        for epoch in range(start_epoch, self.num_epochs + 1):
            self.model.train()  # set model to training mode
            running_loss = 0.0
            num_iterations = math.ceil(len(self.train_loader.dataset) / self.train_loader.batch_size)
            header_printed = False

            tqdm_version = tqdm_nb if self.is_notebook else tqdm
            with tqdm_version(enumerate(self.train_loader, 1), unit="batch", total=num_iterations, leave=False) as tepoch:
                for iter, (inputs, targets) in tepoch:
                    tepoch.set_description(f"Epoch {epoch}/{self.num_epochs}")

                    # -------------------------------------------------------------
                    # Move data to the GPU
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    # zero gradients -> forward pass -> obtain loss function -> apply backpropagation -> update weights:
                    self.optimizer.zero_grad()

                    # A) use mixed precision calculation
                    if self.use_mixed_precision:
                        with autocast(device_type='cuda'):  # Enable autocast for mixed precision training
                            outputs = self.model(inputs)
                            loss = self.loss_fn(outputs.squeeze(), targets)
                        self.scaler.scale(loss).backward()  # Scale the loss and perform backward pass
                        if self.clip_value is not None:
                            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_value)  # optional: Gradient Value Clipping
                        self.scaler.step(self.optimizer)  # Update model parameters
                        self.scaler.update()  # Update the scale for next iteration

                    # B) Normal precision calculation
                    else:
                        outputs = self.model(inputs)
                        loss = self.loss_fn(outputs.squeeze(), targets)
                        loss.backward()
                        if self.clip_value is not None:
                            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_value) # optional: Gradient Value Clipping
                        self.optimizer.step()

                    # -------------------------------------------------------------
                    # Update the performance table
                    if iter % (num_iterations // 4) == 0 and iter != num_iterations // 4 * 4:
                        add_row(self.training_table, f" ", f"{iter}", f"{loss.item():.6f}", " ")
                        if self.is_notebook:
                            display_html(HTML(f"""<script>addRow("", "{iter}", "{loss.item():.6f}", "");</script>"""))
                        else:
                            print_row(self.training_table)
                    elif iter == 1:
                        add_row(self.training_table, f"{epoch}/{self.num_epochs}", f"{iter}/{num_iterations}", f"{loss.item():.6f}", " ")
                        if self.is_notebook:
                            display_html(HTML(f"""<script>addRow("<b>{epoch}/{self.num_epochs}", "{iter}/{num_iterations}", "{loss.item():.6f}", "");</script>"""))
                        else:
                            print_row(self.training_table)

                    # -------------------------------------------------------------
                    # Update running loss and progress bar
                    running_loss += loss.item()  # accumulate loss for epoch
                    tepoch.set_postfix(loss=loss.item())
                    tepoch.update(1)

            # Calculate average training loss for the epoch
            avg_train_loss = running_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)

            # Update the performance table
            add_row(self.training_table, f" ", f"{iter}", f"{loss.item():.6f}", f"{avg_train_loss:.6f}")
            if self.is_notebook:
                display_html(HTML(f"""<script>addRow("", "{iter}", "{loss.item():.6f}", "<b>{avg_train_loss:.6f}");</script>"""))
            else:
                print_row(self.training_table)

            # VALIDATION
            if self.val_loader:
                val_loss = self.validate_model(epoch)
                self.val_losses.append(val_loss)
                # Update the performance table
                add_row(self.training_table, f"Val", f"Validation Loss:", f"{val_loss:.6f}", "")
                if self.is_notebook:
                    display_html(HTML(f"""<script>addRow("<b>Val", "Validation Loss:", "<b>{val_loss:.4f}", "");</script>"""))
                else:
                    print_row(self.training_table)

        elapsed_time = round(time.perf_counter() - start_time)
        self.print_and_log(f"{'-'*60}\nTraining Completed.\tExecution Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}\n{'-'*60}\n")
        return {
            # model and optimizer states
            "model": self.model,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            # training performance
            "training_table": self.training_table,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            'lr_history': self.lr_history,
            # settings and meta data
            "loss_fn": self.loss_fn,
            "epoch": epoch,
            "elapsed_train_time": elapsed_time
        }



#############################################################################################################
#############################################################################################################
# Initialize a HTML table for performance tracking (if running in a notebook)
def initialize_table() -> str:
    table_html = """
    <table id="training_table" style="width:60%; border-collapse: collapse;">
        <thead style="position: sticky; top: 0; z-index: 1;">
            <tr>
                <th style="font-weight:bold; width:15%; text-align:left; padding: 10px; background-color: #404040;">Epoch</th>
                <th style="font-weight:bold; width:25%; text-align:left; padding: 10px; background-color: #404040;">Iteration</th>
                <th style="font-weight:bold; width:30%; text-align:left; padding: 10px; background-color: #404040;">Batch Loss</th>
                <th style="font-weight:bold; width:30%; text-align:left; padding: 10px; background-color: #404040;">Train Loss</th>
            </tr>
        </thead>
        <tbody>
        </tbody>
    </table>
    <script>
        function addRow(epoch, step, loss, running_loss) {
            var table = document.getElementById("training_table").getElementsByTagName('tbody')[0];
            var row = table.insertRow(-1);
            var cell1 = row.insertCell(0);
            var cell2 = row.insertCell(1);
            var cell3 = row.insertCell(2);
            var cell4 = row.insertCell(3);
            cell1.style.textAlign = "left";
            cell2.style.textAlign = "left";
            cell3.style.textAlign = "left";
            cell4.style.textAlign = "left";
            cell1.innerHTML = epoch;
            cell2.innerHTML = step;
            cell3.innerHTML = loss;
            cell4.innerHTML = running_loss;
            var scrollableDiv = document.getElementById("scrollable_table");
            scrollableDiv.scrollTop = scrollableDiv.scrollHeight;
        }
    </script>
    """
    return """<div id="scrollable_table" style="height: 300px; overflow-y: scroll;">""" + table_html + """</div>"""

##########################################################
# ---------------------------------------------------
def add_row(table: list, epoch: str, iteration: str, batch_loss: str, train_loss: str):
    table.append([epoch, iteration, batch_loss, train_loss])

##########################################################
# Function to print the performance table
header_printed = False
def print_row(training_table: list):
    global header_printed
    headers = ["Epoch", "Iteration", "Batch Loss", "Train Loss"]
    col_widths = [14, 14, 14, 14]  # Define fixed column widths

    def format_row(row: list) -> list:
        return [str(item).ljust(width) for item, width in zip(row, col_widths)]

    if not header_printed:
        formatted_headers = format_row(headers)
        tqdm.write(tabulate([training_table[-1]], headers=formatted_headers, tablefmt="plain", colalign=("left", "left", "left", "left")))
        header_printed = True
    else:
        formatted_row = format_row(training_table[-1])
        tqdm.write(tabulate([training_table[-1]], headers=format_row(["", "", "", ""]), tablefmt="plain", colalign=("left", "left", "left", "left")))