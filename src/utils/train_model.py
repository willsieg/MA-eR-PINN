import os
import time
import math
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from tabulate import tabulate


# TRAINING ROUTINE DEFINITION -----------------------------------------------------------------
def train_model(model, optimizer, scheduler, loss_fn, train_loader, num_epochs, device, is_notebook, val_loader = None, state=None):

    if is_notebook:
        from IPython.display import display, HTML, Javascript
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
        from tabulate import tabulate
        print('imported tqdm')

    def validate_model(model, val_loader, loss_fn):
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient calculation
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs.squeeze(), targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)  # Calculate average validation loss
        scheduler.step(val_loss)    # Adjust learning rate based on validation loss
        return val_loss

    # output info on training process
    print(f"Training Started.\tProcess ID: {os.getpid()} \n{'-'*60}\n"
        f"Model: {model.__class__.__name__}\t\tParameters on device: {str(next(model.parameters()).device).upper()}\n{'-'*60}\n"
        f"Train/Batch size:\t{len(train_loader.dataset)} / {train_loader.batch_size}\n"
        f"Loss:\t\t\t{loss_fn}\nOptimizer:\t\t{optimizer.__class__.__name__}\nLR:\t\t\t"
        f"{optimizer.param_groups[0]['lr']}\nWeight Decay:\t\t{optimizer.param_groups[0]['weight_decay']}\n{'-'*60}")

    # Load state dict if provided
    start_epoch = 1
    if state:
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        start_epoch = state['epoch'] + 1
        train_losses = state['train_losses']
        val_losses = state['val_losses']
        training_table = state['training_table']
    else:
        train_losses, val_losses, training_table = [], [], []  # collect loss
        if is_notebook: display(HTML(initialize_table()))

    # TRAINING LOOP:
    start_time = time.perf_counter()
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()   # set model to training mode
        running_loss = 0.0
        num_iterations = math.ceil(len(train_loader.dataset) / train_loader.batch_size)
        header_printed = False
        
        with tqdm(enumerate(train_loader, 1), unit="batch", total=num_iterations, leave=False) as tepoch:
            for iter, (inputs, targets) in tepoch:
                tepoch.set_description(f"Epoch {epoch}/{num_epochs}")

                # -------------------------------------------------------------
                # Move data to the GPU
                inputs, targets = inputs.to(device), targets.to(device)  
                # zero gradients -> forward pass -> obtain loss function -> apply backpropagation -> update weights:
                optimizer.zero_grad()
                outputs = model(inputs) 
                loss = loss_fn(outputs.squeeze(), targets) 
                loss.backward() 
                # nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0) # optional: Gradient Value Clipping
                optimizer.step()

                # -------------------------------------------------------------
                # Update the performance table
                if iter % (num_iterations//4) == 0 and iter != num_iterations//4*4:
                    add_row(training_table, f" ", f"{iter}",f"{loss.item():.6f}", " ")
                    if is_notebook:
                        display(Javascript(f"""addRow("", "{iter}", "{loss.item():.6f}", "");"""))
                    else:
                        print_row(training_table)
                elif iter == 1:
                    add_row(training_table, f"{epoch}/{num_epochs}", f"{iter}/{num_iterations}",f"{loss.item():.6f}", " ")
                    if is_notebook:
                        display(Javascript(f"""addRow("<b>{epoch}/{num_epochs}", "{iter}/{num_iterations}", "{loss.item():.6f}", "");"""))
                    else:
                        print_row(training_table)
                        
                # -------------------------------------------------------------
                # Update running loss and progress bar
                running_loss += loss.item() # acculumate loss for epoch
                tepoch.set_postfix(loss=loss.item()); tepoch.update(1)

        # Calculate average training loss for the epoch
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Update the performance table
        add_row(training_table, f" ", f"{iter}",f"{loss.item():.6f}", f"{avg_train_loss:6f}")
        if is_notebook:
            display(Javascript(f"""addRow("", "{iter}", "{loss.item():.6f}", "<b>{avg_train_loss:.6f}");"""))
        else:
            print_row(training_table)

        # VALIDATION
        if val_loader:
            val_loss = validate_model(model, val_loader, loss_fn)
            val_losses.append(val_loss)
            # Update the performance table
            add_row(training_table, f" ", f"Validation Loss:",f"{val_loss:.6f}", f"")
            if is_notebook:
                display(Javascript(f"""addRow("<b>Val", "Validation Loss:", "<b>{val_loss:.4f}", "");"""))
            else:
                print_row(training_table)

    print(f"{'-'*60}\nTraining Completed.\tExecution Time: {time.strftime('%H:%M:%S', time.gmtime(time.perf_counter() - start_time))}\n")
    return {"train_losses": train_losses, "val_losses": val_losses, "epoch": epoch, "training_table": training_table, "model": model, 
    "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),"loss_fn": loss_fn}


################################################################################################################################################    
# Initialize a HTML table for performance tracking (if running in a notebook)
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

# ---------------------------------------------------
def add_row(table, epoch, iteration, batch_loss, train_loss):
    table.append([epoch, iteration, batch_loss, train_loss])

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