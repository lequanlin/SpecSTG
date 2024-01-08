import os
import numpy as np
import csv


class DataLoader_STG():

    """Load and pre-process data for STG forecasting
       Download data at https://github.com/guoshnBJTU/ASTGNN/tree/main/data"""

    def __init__(self, normalization = True):
        self.normalization = normalization

        self.X = None
        self.X_norm = None
        self.A = None
        self.edge_weight = None
        self.edge_index = None
        self.input_size = None
        self.mean = None
        self.std = None

    def __call__(self, dataset_name):

        if dataset_name == 'PEMS04S':
            self.Load_PM04(datatype='speed')

        if dataset_name == 'PEMS04F':
            self.Load_PM04(datatype='flow')

        if dataset_name == 'PEMS08S':
            self.Load_PM08(datatype='speed')

        if dataset_name == 'PEMS08F':
            self.Load_PM08(datatype='flow')

    def Load_PM04(self, datatype):

        """

        The datasets are collected by the Caltrans Performance Measurement System (PeMS) (Chen et al. 2001)
         in real time every 30 seconds. The traffic data are aggregated into every 5-minute interval from
         the raw data. The system has more than 39,000 detectors deployed on the highway in the major
         metropolitan areas in California. Geographic information about the sensor stations are recorded in
         the datasets. There are three kinds of traffic measurements considered in our experiments,
         including total flow, average speed, and average occupancy.

        The dataset refers to the traffic data in San Francisco Bay Area, containing 307 detectors in District 04.
        The time span of this dataset is from January 1st to February 28th in 2018. Total time length is 16,992.
        For more information, please refer to <https://ojs.aaai.org/index.php/AAAI/article/view/3881>

        """

        raw_data_dir = os.path.join(os.getcwd(), "./Data/PEMS04/")

        # Get graph signals
        X = np.load(os.path.join(raw_data_dir, "pems04.npz"))['data']
        X = X.astype(np.float32)

        # 0 for flow, 1 for occupancy, 2 for speed
        if datatype == 'speed':
            X = X[:, :, 2].T
        else:
            X = X[:, :, 0].T
        self.X = X

        if self.normalization:
            mean = np.mean(X)
            std = np.std(X)
            X = (X - mean) / std
            self.X_norm = X
            self.mean = mean
            self.std = std

        # Get adjacency matrix
        A = get_adjacency_matrix("./Data/PEMS04/distance.csv", X.shape[0])
        A = A + A.T
        A[A != 0] = 1
        self.A = A

        # Get edges
        edge_index = np.nonzero(A)
        edge_weight = np.float32(A[edge_index])
        edge_index = np.stack(edge_index, axis=0)
        self.edge_weight = edge_weight
        self.edge_index = edge_index

        self.input_size = 11

    def Load_PM08(self, datatype):

        """

        The datasets are collected by the Caltrans Performance Measurement System (PeMS) (Chen et al. 2001)
         in real time every 30 seconds. The traffic data are aggregated into every 5-minute interval from
         the raw data. The system has more than 39,000 detectors deployed on the highway in the major
         metropolitan areas in California. Geographic information about the sensor stations are recorded in
         the datasets. There are three kinds of traffic measurements considered in our experiments,
         including total flow, average speed, and average occupancy.

        The dataset refers to the traffic data in San Francisco Bay Area, containing 170 detectors in District 08.
        The time span of this dataset is from January 1st to February 28th in 2018. Time length is 17,856.
         For more information, please refer to <https://ojs.aaai.org/index.php/AAAI/article/view/3881>

        """

        raw_data_dir = os.path.join(os.getcwd(), "./Data/PEMS08/")

        # Get graph signals
        X = np.load(os.path.join(raw_data_dir, "pems08.npz"))['data']
        X = X.astype(np.float32)

        # 0 for flow, 1 for occupancy, 2 for speed
        if datatype == 'speed':
            X = X[:, :, 2].T
        else:
            X = X[:, :, 0].T

        self.X = X

        if self.normalization:
            mean = np.mean(X)
            std = np.std(X)
            X = (X - mean) / std
            self.X_norm = X
            self.mean = mean
            self.std = std

        # Get adjacency matrix
        A = get_adjacency_matrix("./Data/PEMS08/distance.csv", X.shape[0])
        A = A + A.T
        A[A != 0] = 1
        self.A = A

        # Get edges
        edge_index = np.nonzero(A)
        edge_weight = np.float32(A[edge_index])
        edge_index = np.stack(edge_index, axis=0)
        self.edge_weight = edge_weight
        self.edge_index = edge_index

        self.input_size = 11


def get_adjacency_matrix(distance_df_filename, num_of_vertices):

    """
    To get adjacency matrix for dataset PEMS04 and PEMS08

    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    """

    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        edges = [(int(i[0]), int(i[1])) for i in reader]

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    for i, j in edges:
        A[i, j] = 1

    return A




