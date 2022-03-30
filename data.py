import numpy as np


class Dataset:
    """
        An abstract class representing a :class:`Dataset`
    """
    
    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class Dataloader:
    """
        An abstract class representing a :class:'Dataloader'
    """
    def __iter__(self):
        return self

    def __next__(self):
        return NotImplementedError

class Cifar10Dataset(Dataset):
    def __init__(self, cifar_features, cifar_labels, normalize=True, one_hot=True):
        super().__init__()
        if normalize:
            self.features = cifar_features / 255.0
        else:
            self.features = cifar_features
        if one_hot:
            self.labels = np.zeros((cifar_labels.size, cifar_labels.max() + 1))
            self.labels[np.arange(cifar_labels.size), cifar_labels] = 1
        else:
            self.labels = cifar_labels
    
    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

class CifarDataloader(Dataloader):
    def __init__(self, dataset, batch_size) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.curr_index = 0
    
    def __next__(self):
        if self.curr_index < len(self.dataset):
            self.curr_index += self.batch_size
            return self.dataset[self.curr_index - self.batch_size: min(self.curr_index, len(self.dataset))]
        else:
            self.reset()
            raise StopIteration()
    
    def reset(self):
        self.curr_index = 0

    def __len__(self):
        return len(self.dataset) // self.batch_size