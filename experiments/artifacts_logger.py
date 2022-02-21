from pytorch_lightning.callbacks import Callback


class ArtifactLogger(Callback):
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        return super().on_train_epoch_end(trainer, pl_module)

    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")