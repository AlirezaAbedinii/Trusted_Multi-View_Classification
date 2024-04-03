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

        param root: Path and name of the data file.
        param train: Whether to load the training set (True) or test set (False).
        """
        super(MultiViewData, self).__init__()
        self.root = root
        self.train = train

        data_path = f'{self.root}.mat'

        # loads the dataset from a MATLAB file
        dataset = sio.loadmat(data_path)

        # length of the dataset returned from the MATLAB file, there are 5 additional entries not related to data views
        view_count = int((len(dataset) - 5) / 2)

        # hold data from each view
        self.X = {}
        data_key_suffix = 'train' if train else 'test'
        y_key = f'gt_{data_key_suffix}'

        # loops through each view, normalizes the data using the normalize method, and adds it to self.X.
        for v_num in range(view_count):
            x_key = f'x{v_num + 1}_{data_key_suffix}'
            self.X[v_num] = self.normalize(dataset[x_key])

        # Labels are loaded from the dataset. If the minimum value is 1 (MATLAB indexing),
        # it is subtracted by 1 to match Python's 0-indexing. Then, y is flattened using np.ravel.
        y = dataset[y_key] - 1 if np.min(dataset[y_key]) == 1 else dataset[y_key]
        self.y = np.ravel(y)

    def __getitem__(self, index):
        """
        Retrieves a single data point and its label, given an index.
        creates a dictionary from all views, converted to np.float32 type,
        and retrieves the corresponding label from self.y
        """
        data = {v_num: self.X[v_num][index].astype(np.float32) for v_num in self.X}
        target = self.y[index]
        return data, target

    def __len__(self):
        """
        Get the total number of data points.
        """
        return len(next(iter(self.X.values())))

    @staticmethod
    def normalize(x, min_value=0):
        """
        Normalize the data.
        """
        if min_value == 0:
            feature_range = (0, 1)
        else:
            feature_range = (-1, 1)
        scaler = MinMaxScaler(feature_range=feature_range)
        return scaler.fit_transform(x)