import os, time, math, torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from torchrl.trainers import Trainer as TorchRLTrainer
from tqdm.notebook import tqdm as tqdm_nb
from tqdm import tqdm as tqdm
from IPython.display import HTML, display_html
from tabulate import tabulate
if torch.cuda.is_available(): from torch.amp import GradScaler, autocast


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
class Trainer():

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
                print(f"Learning rate updated after epoch {epoch}: {lr1} -> {lr2}")
        return val_loss

    # TRAINING ROUTINE DEFINITION -----------------------------------------------------------------
    def train_model(self) -> dict:
       # output info on training process
        print(f"{'-'*60}\nTraining Started.\tProcess ID: {os.getpid()} \n{'-'*60}\n"
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
        print(f"{'-'*60}\nTraining Completed.\tExecution Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}\n{'-'*60}\n")
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


'''#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
class Trainer_packed():

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
                print(f"Learning rate updated after epoch {epoch}: {lr1} -> {lr2}")
        return val_loss

    # TRAINING ROUTINE DEFINITION -----------------------------------------------------------------
    def train_model(self) -> dict:
       # output info on training process
        print(f"{'-'*60}\nTraining Started.\tProcess ID: {os.getpid()} \n{'-'*60}\n"
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
                for iter, (inputs, targets, length) in tepoch:  # ----> note: (packed_inputs, padded_targets, lengths)
                    tepoch.set_description(f"Epoch {epoch}/{self.num_epochs}")

                    # -------------------------------------------------------------
                    # Move data to the GPU
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    # zero gradients -> forward pass -> obtain loss function -> apply backpropagation -> update weights:
                    self.optimizer.zero_grad()

                    # A) use mixed precision calculation
                    if self.use_mixed_precision:
                        with autocast(device_type='cuda'):  # Enable autocast for mixed precision training
                            outputs = self.model(inputs)   # inputs are packed, outputs are not ! --> see forward method in model
                            loss = self.loss_fn(outputs.squeeze(), targets)
                        self.scaler.scale(loss).backward()  # Scale the loss and perform backward pass
                        if self.clip_value is not None:
                            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_value)  # optional: Gradient Value Clipping
                        self.scaler.step(self.optimizer)  # Update model parameters
                        self.scaler.update()  # Update the scale for next iteration

                    # B) Normal precision calculation
                    else:
                        print(f"Shape of inputs: {inputs.shape}")
                        outputs = self.model(inputs)  # inputs are packed, outputs are not ! --> see forward method in model
                        print(f"Shape of outputs: {inputs.shape}")
                        loss = self.loss_fn(outputs, targets)  # UPDATE: outpts.squeeze() --> outputs
                        loss.backward()
                        if self.clip_value is not None:
                            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_value)  # optional: Gradient Value Clipping
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
        print(f"{'-'*60}\nTraining Completed.\tExecution Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}\n{'-'*60}\n")
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
'''

################################################################################################################################################    
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

# ---------------------------------------------------
def add_row(table: list, epoch: str, iteration: str, batch_loss: str, train_loss: str):
    table.append([epoch, iteration, batch_loss, train_loss])

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