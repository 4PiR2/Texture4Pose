from pytorch_lightning.plugins import TorchCheckpointIO


class CkptIO(TorchCheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        super().save_checkpoint(checkpoint, path, storage_options)

    def load_checkpoint(self, path, storage_options=None):
        checkpoint = super().load_checkpoint(path, storage_options)
        return checkpoint

    def remove_checkpoint(self, path):
        super().remove_checkpoint(path)
        print('remove:', path)
