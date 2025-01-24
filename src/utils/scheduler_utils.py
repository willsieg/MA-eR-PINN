import math

class ParameterScheduler:
    def __init__(self, initial_value, schedule_type='constant', **kwargs):
        self.initial_value = initial_value
        self.schedule_type = schedule_type
        self.kwargs = kwargs

        # Ensure all necessary parameters are provided
        if self.schedule_type == 'time_based':
            if 'decay_rate' not in self.kwargs:
                raise ValueError("decay_rate must be provided for time_based schedule")
        elif self.schedule_type == 'step_based':
            if 'drop_rate' not in self.kwargs or 'epochs_drop' not in self.kwargs:
                raise ValueError("drop_rate and epochs_drop must be provided for step_based schedule")
        elif self.schedule_type == 'exponential':
            if 'decay_rate' not in self.kwargs:
                raise ValueError("decay_rate must be provided for exponential schedule")
        elif self.schedule_type == 'linear':
            if 'absolute_reduction' not in self.kwargs:
                raise ValueError("absolute_reduction must be provided for linear schedule")
        elif self.schedule_type == 'cosine_annealing':
            if 'total_epochs' not in self.kwargs:
                raise ValueError("total_epochs must be provided for cosine_annealing schedule")
        elif self.schedule_type == 'cyclic':
            if 'base_lr' not in self.kwargs or 'max_lr' not in self.kwargs or 'step_size' not in self.kwargs:
                raise ValueError("base_lr, max_lr, and step_size must be provided for cyclic schedule")
                
        elif self.schedule_type == 'reverse_sigmoid':
            if 'total_epochs' not in self.kwargs:
                raise ValueError("total_epochs must be provided for reverse_sigmoid schedule")

    def get_value(self, epoch):
        if self.schedule_type == 'constant':
            value = self.initial_value
        elif self.schedule_type == 'time_based':
            decay_rate = self.kwargs['decay_rate']
            value = self.initial_value / (1 + decay_rate * epoch)
        elif self.schedule_type == 'step_based':
            drop_rate = self.kwargs['drop_rate']
            epochs_drop = self.kwargs['epochs_drop']
            value = self.initial_value * math.pow(drop_rate, math.floor((1 + epoch) / epochs_drop))
        elif self.schedule_type == 'exponential':
            decay_rate = self.kwargs['decay_rate']
            value = self.initial_value * math.pow(decay_rate, epoch)
        elif self.schedule_type == 'linear':
            absolute_reduction = self.kwargs['absolute_reduction']
            value = self.initial_value - absolute_reduction * epoch
        elif self.schedule_type == 'cosine_annealing':
            total_epochs = self.kwargs['total_epochs']
            value = self.initial_value * (1 + math.cos(math.pi * epoch / total_epochs)) / 2
        elif self.schedule_type == 'cyclic':
            base_lr = self.kwargs['base_lr']
            max_lr = self.kwargs['max_lr']
            step_size = self.kwargs['step_size']
            cycle = math.floor(1 + epoch / (2 * step_size))
            x = abs(epoch / step_size - 2 * cycle + 1)
            value = base_lr + (max_lr - base_lr) * max(0, (1 - x))
        elif self.schedule_type == 'reverse_sigmoid':
            total_epochs = self.kwargs['total_epochs']
            value = self.initial_value / (1 + math.exp(-10 * (epoch / total_epochs - 0.5)))
        else:
            raise ValueError(f"Unknown schedule_type: {self.schedule_type}")

        return value

# Example usage
if __name__ == "__main__":
    # Constant value
    scheduler = ParameterScheduler(initial_value=0.1, schedule_type='constant')
    for epoch in range(5):
        print(f"Epoch {epoch}: {scheduler.get_value(epoch)}")

    # Time-based decay
    scheduler = ParameterScheduler(initial_value=0.1, schedule_type='time_based', decay_rate=0.1)
    for epoch in range(5):
        print(f"Epoch {epoch}: {scheduler.get_value(epoch)}")

    # Step-based decay
    scheduler = ParameterScheduler(initial_value=0.1, schedule_type='step_based', drop_rate=0.5, epochs_drop=2)
    for epoch in range(5):
        print(f"Epoch {epoch}: {scheduler.get_value(epoch)}")

    # Exponential decay
    scheduler = ParameterScheduler(initial_value=0.1, schedule_type='exponential', decay_rate=0.96)
    for epoch in range(5):
        print(f"Epoch {epoch}: {scheduler.get_value(epoch)}")

    # Linear decay with absolute reduction
    scheduler = ParameterScheduler(initial_value=0.1, schedule_type='linear', absolute_reduction=0.01)
    for epoch in range(5):
        print(f"Epoch {epoch}: {scheduler.get_value(epoch)}")

    # Cosine Annealing
    scheduler = ParameterScheduler(initial_value=0.1, schedule_type='cosine_annealing', total_epochs=10)
    for epoch in range(10):
        print(f"Epoch {epoch}: {scheduler.get_value(epoch)}")

    # Cyclic Learning Rate
    # initial value not used
    scheduler = ParameterScheduler(initial_value=0.1, schedule_type='cyclic', base_lr=0.01, max_lr=0.1, step_size=5)
    for epoch in range(20):
        print(f"Epoch {epoch}: {scheduler.get_value(epoch)}")

    # Reverse Sigmoid decay
    scheduler = ParameterScheduler(initial_value=1.0, schedule_type='reverse_sigmoid', total_epochs=10)
    for epoch in range(10):
        print(f"Epoch {epoch}: {scheduler.get_value(epoch)}")