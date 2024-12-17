import os
import time
import math
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchrl.trainers import Trainer as TorchRLTrainer
from tqdm import tqdm
from tabulate import tabulate
from IPython.display import display, HTML, Javascript, display_javascript, display_html

if torch.cuda.is_available(): from torch.cuda.amp import GradScaler, autocast

class Trainer(TorchRLTrainer):
    def __init__(self, model, optimizer, loss_fn, train_loader, num_epochs, device, is_notebook=False, val_loader=None, scheduler=None, state=None, use_mixed_precision=True):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device
        self.train_losses = []
        self.val_losses = []
        training_table = []
        self.state = state
        #self.header_printed = False
        self.use_mixed_precision = use_mixed_precision if torch.cuda.is_available() else False
        if self.use_mixed_precision: self.scaler = GradScaler()  # Initialize GradScaler
        self.is_notebook = is_notebook

    def validate_model(self, epoch):
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
                lr1 = self.scheduler.get_last_lr()
                self.scheduler.step(val_loss)  # Adjust learning rate based on validation loss
                if lr1 != self.scheduler.get_last_lr():
                    print(f"Learning rate updated after epoch {epoch}: {lr1} -> {self.scheduler.get_last_lr()}")
            return val_loss

    def train(self):

        if self.is_notebook:
            #from IPython.display import display, HTML, Javascript
            from tqdm.notebook import tqdm
            from tabulate import tabulate
            display(HTML("<h3>Training Progress</h3>"))
        else:
            from tqdm import tqdm
            from tabulate import tabulate
            print('imported tqdm')

        # output info on training process
        print(f"Training Started.\tProcess ID: {os.getpid()} \n{'-'*60}\n"
              f"Model: {self.model.__class__.__name__}\t\tParameters on device: {str(next(self.model.parameters()).device).upper()}\n{'-'*60}\n"
              f"Train/Batch size:\t{len(self.train_loader.dataset)} / {self.train_loader.batch_size}\n"
              f"Loss:\t\t\t{self.loss_fn}\nOptimizer:\t\t{self.optimizer.__class__.__name__}\nLR:\t\t\t"
              f"{self.optimizer.param_groups[0]['lr']}\nWeight Decay:\t\t{self.optimizer.param_groups[0]['weight_decay']}\n{'-'*60}")

        '''start_epoch = 1
        if self.state:
            self.model.load_state_dict(self.state['model_state_dict'])
            self.optimizer.load_state_dict(self.state['optimizer_state_dict'])
            start_epoch = self.state['epoch'] + 1
            self.train_losses = self.state['train_losses']
            self.val_losses = self.state['val_losses']
            self.training_table = self.state['training_table']
        else:
            if self.is_notebook:
                display(HTML(initialize_table()))'''

                
        # Load state dict if provided
        start_epoch = 1
        if self.state:
            self.model.load_state_dict(self.state['model_state_dict'])
            self.optimizer.load_state_dict(self.state['optimizer_state_dict'])
            start_epoch = self.state['epoch'] + 1
            self.train_losses = self.state['train_losses']
            self.val_losses = self.state['val_losses']
            self.training_table = self.state['training_table']
        else:
            train_losses, val_losses, training_table = [], [], []  # collect loss
            global tab
            tab = initialize_table()
            if self.is_notebook: 
                display_html(HTML(tab))

        # TRAINING LOOP:
        start_time = time.perf_counter()
        for epoch in range(start_epoch, self.num_epochs + 1):
            self.model.train()  # set model to training mode
            running_loss = 0.0
            num_iterations = math.ceil(len(self.train_loader.dataset) / self.train_loader.batch_size)
            header_printed = False

            with tqdm(enumerate(self.train_loader, 1), unit="batch", total=num_iterations, leave=False) as tepoch:
                for iter, (inputs, targets) in tepoch:
                    tepoch.set_description(f"Epoch {epoch}/{self.num_epochs}")

                    # -------------------------------------------------------------
                    # Move data to the GPU
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    # zero gradients -> forward pass -> obtain loss function -> apply backpropagation -> update weights:
                    # use mixed precision calculation
                    if self.use_mixed_precision:
                        with autocast():  # Enable autocast for mixed precision training
                            outputs = self.model(inputs)
                            loss = self.loss_fn(outputs.squeeze(), targets)
                        self.scaler.scale(loss).backward()  # Scale the loss and perform backward pass
                        self.scaler.step(self.optimizer)  # Update model parameters
                        self.scaler.update()  # Update the scale for next iteration

                    # Normal calculation
                    else:
                        outputs = self.model(inputs)
                        loss = self.loss_fn(outputs.squeeze(), targets)
                        loss.backward()
                        # nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0) # optional: Gradient Value Clipping
                        self.optimizer.step()

                    # -------------------------------------------------------------
                    # Update the performance table
                    if iter % (num_iterations // 4) == 0 and iter != num_iterations // 4 * 4:
                        add_row(training_table, f" ", f"{iter}", f"{loss.item():.6f}", " ")
                        if self.is_notebook:
                            display(Javascript(f"""addRow("", "{iter}", "{loss.item():.6f}", "");"""))
                        else:
                            print_row(training_table)
                    elif iter == 1:
                        add_row(training_table, f"{epoch}/{self.num_epochs}", f"{iter}/{num_iterations}", f"{loss.item():.6f}", " ")
                        if self.is_notebook:
                            print('hello')

                            display_javascript(Javascript(f"""addRow("<b>{epoch}</b>", "{iter}/{num_iterations}", "{loss.item():.6f}", "");"""))
                            #display(Javascript(f"""addRow("<b>{epoch}/{self.num_epochs}", "{iter}/{num_iterations}", "{loss.item():.6f}", "");"""))
                        else:
                            print_row(training_table)

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
                display(Javascript(f"""addRow("", "{iter}", "{loss.item():.6f}", "<b>{avg_train_loss:.6f}");"""))
            else:
                self.print_row(self.training_table)

            # VALIDATION
            if self.val_loader:
                val_loss = self.validate_model(epoch)
                self.val_losses.append(val_loss)
                # Update the performance table
                add_row(self.training_table, f" ", f"Validation Loss:", f"{val_loss:.6f}", f"")
                if self.is_notebook:
                    display(Javascript(f"""addRow("<b>Val", "Validation Loss:", "<b>{val_loss:.4f}", "");"""))
                else:
                    self.print_row(self.training_table)

        elapsed_time = round(time.perf_counter() - start_time)
        print(f"{'-'*60}\nTraining Completed.\tExecution Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}\n")
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "epoch": epoch,
            "training_table": self.training_table,
            "model": self.model,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_fn": self.loss_fn,
            "elapsed_train_time": elapsed_time
        }


# Function to print the performance table
header_printed = False
def print_row(training_table):
    global header_printed
    headers = ["Epoch", "Iteration", "Batch Loss", "Train Loss"]
    col_widths = [14, 14, 14, 14]  # Define fixed column widths

    def format_row(row):
        return [str(item).ljust(width) for item, width in zip(row, col_widths)]

    if not header_printed:
        formatted_headers = format_row(headers)
        tqdm.write(tabulate([training_table[-1]], headers=formatted_headers, tablefmt="plain", colalign=("left", "left", "left", "left")))
        header_printed = True
    else:
        formatted_row = format_row(training_table[-1])
        tqdm.write(tabulate([training_table[-1]], headers=format_row(["", "", "", ""]), tablefmt="plain", colalign=("left", "left", "left", "left")))


################################################################################################################################################    
# Initialize a HTML table for performance tracking (if running in a notebook)

# ---------------------------------------------------
def add_row(table, epoch, iteration, batch_loss, train_loss):
    table.append([epoch, iteration, batch_loss, train_loss])

def initialize_table():
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