import torch.utils.data as Data

class Dataset(Data.TensorDataset):
    def __init__(self, n_train, n_val, n_test, batch_size, num_workers) -> None:
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.batch_size = batch_size
        self.num_workers = num_workers
