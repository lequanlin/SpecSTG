import numpy as np
import torch
from torch_geometric.utils import get_laplacian


class SpecTransform():

    def __init__(self,
                 dataset: str = 'PEMS08F',
                 ):

        self.dataset = dataset

        self.X_f = None
        self.transform_matrix = None
        self.reconstruct_matrix = None
        self.Lambda = None

    def __call__(self, X, edge_weight, edge_index, num_nodes):

        L = get_laplacian(torch.from_numpy(edge_index), torch.from_numpy(edge_weight), num_nodes=num_nodes,
                          normalization='sym')
        L_edge_index = L[0].numpy()
        L_edge_weight = L[1].numpy()
        L = np.zeros((num_nodes, num_nodes))
        for idx, weight in enumerate(L_edge_weight):
            i = L_edge_index[0, idx]
            j = L_edge_index[1, idx]
            L[i, j] = weight
        if "PEMS04" in self.dataset:
            lambdas, U = np.linalg.eigh(L)
        else:
            lambdas, U = np.linalg.eig(L)

        U = U.astype(np.float32)
        lambdas = lambdas.astype(np.float32)
        self.Lambda = lambdas
        self.U = U
        self.Fourier_transform(X)

    # Fourier transform
    def Fourier_transform(self, X):

        X_f = self.U.T @ X
        self.X_f = X_f
        self.transform_matrix =  self.U.T
        self.reconstruct_matrix=  self.U

    def Fourier_reconstruct(self, X_f):

        return self.reconstruct_matrix @ X_f












