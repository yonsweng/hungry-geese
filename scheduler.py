class ExponentialWarmUpSceduler():
    def __init__(self, init):
        self.target_lr = target_lr
        self.factor = (target_lr / init_lr) ** (1 / steps)

    def __call__(self, lr):
        if
        if lr < self.target_lr:
            self.lr *= self.factor
        return self.lr
