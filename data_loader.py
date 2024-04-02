import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class MultiViewData(Dataset):
    """
    A class to load multi-view data.
    """

    def __init__(self, root, train=True):
        """
        Initialize the dataset.

        :param root: Path and name of the data file.
        :param train: Whether to load the training set (True) or test set (False).
        """
        super(MultiViewData, self).__init__()
        self.root = root
        self.train = train

        data_path = f'{self.root}.mat'
        dataset = sio.loadmat(data_path)
        view_count = int((len(dataset) - 5) / 2)

        self.X = {}
        data_key_suffix = 'train' if train else 'test'
        y_key = f'gt_{data_key_suffix}'

        for v_num in range(view_count):
            x_key = f'x{v_num + 1}_{data_key_suffix}'
            self.X[v_num] = self.normalize(dataset[x_key])

        y = dataset[y_key] - 1 if np.min(dataset[y_key]) == 1 else dataset[y_key]
        self.y = np.ravel(y)

    def __getitem__(self, index):
        """
        Get a data point and its label by index.

        :param index: Index of the data point.
        :return: A tuple of data (from all views) and label.
        """
        data = {v_num: self.X[v_num][index].astype(np.float32) for v_num in self.X}
        target = self.y[index]
        return data, target

    def __len__(self):
        """
        Get the total number of data points.

        :return: The size of the dataset.
        """
        return len(next(iter(self.X.values())))

    @staticmethod
    def normalize(x, min_value=0):
        """
        Normalize the data.

        :param x: The data to be normalized.
        :param min_value: Indicates the feature range
        :return: Normalized data.
        """
        if min_value == 0:
            feature_range = (0, 1)
        else:
            feature_range = (-1, 1)
        scaler = MinMaxScaler(feature_range=feature_range)
        return scaler.fit_transform(x)