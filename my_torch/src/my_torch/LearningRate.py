class LRScheduler:
    def __init__(self):
        pass

    def get_learning_rate(self, current_lr, epoch):
        raise NotImplementedError("This method should be overridden by subclasses.")

class ExponentialLR(LRScheduler):
    def __init__(self, decay_rate: float):
        super().__init__()
        self.decay_rate = decay_rate

    def get_learning_rate(self, current_lr, epoch):
        return current_lr * self.decay_rate

class ConstantLR(LRScheduler):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    def get_learning_rate(self, current_lr, epoch):
        return self.learning_rate