class TrainerState:
    def __init__(self):
        self.current_step = 0
        self.optimizer_step = 0
        self.epoch = 0
        self.best_metric_value = float("inf")
        self.best_checkpoint_step = -1
