import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpecConv(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            out_size: int,
            K: int,
            bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.K = K
        self.theta = torch.nn.Parameter(torch.Tensor(K))
        self.linear = torch.nn.Linear(input_size, out_size, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        self.linear.reset_parameters()
        ones(self.theta)

    def forward(self, x, Lambda):
        """
        Input assumed to be x: [batch*num_nodes, input_size]
        Lambda: [batch*num_nodes] of eigenvalues
        """

        out = self.theta[0] * x
        Lambda = Lambda.view(-1,1)
        for it in range(1,self.K):
            lambda_mul = Lambda.pow(it)
            out += self.theta[it] * lambda_mul * x

        out = self.linear(out)

        return out

class SGGRU(torch.nn.Module):
    r"""An implementation of the Chebyshev Graph Convolutional Gated Recurrent Unit
    Cell. For details see this paper: `"Structured Sequence Modeling with Graph
    Convolutional Recurrent Networks." <https://arxiv.org/abs/1612.07659>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        K: int,
        dropout_rate: float,
        normalization: str = "sym",
        bias: bool = True,
    ):
        super(SGGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()
        self.edge_index = None
        self.edge_weight = None
        self.dropout_rate = dropout_rate

    def _create_update_gate_parameters_and_layers(self):

        self.conv_x_z = SpecConv(
            input_size=self.input_size,
            out_size=self.hidden_size,
            K=self.K,
            bias=self.bias,
        )

        self.conv_h_z = SpecConv(
            input_size=self.hidden_size,
            out_size=self.hidden_size,
            K=self.K,
            bias=self.bias,
        )

    def _create_reset_gate_parameters_and_layers(self):

        self.conv_x_r = SpecConv(
            input_size = self.input_size,
            out_size = self.hidden_size,
            K=self.K,
            bias=self.bias,
        )


        self.conv_h_r = SpecConv(
            input_size=self.hidden_size,
            out_size=self.hidden_size,
            K=self.K,
            bias=self.bias,
        )

    def _create_candidate_state_parameters_and_layers(self):

        self.conv_x_h = SpecConv(
            input_size=self.input_size,
            out_size=self.hidden_size,
            K=self.K,
            bias=self.bias,
        )

        self.conv_h_h = SpecConv(
            input_size=self.hidden_size,
            out_size=self.hidden_size,
            K=self.K,
            bias=self.bias,
        )

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], X.shape[2], self.hidden_size).to(X.device)
        return H

    def _calculate_update_gate(self, X, H, Lambda):
        batches = X.shape[0]
        num_nodes = X.shape[1]
        Z = self.conv_x_z(X.reshape(batches * num_nodes, -1),Lambda)
        Z = Z + self.conv_h_z(H.reshape(batches * num_nodes, -1),Lambda)
        Z = torch.sigmoid(Z).reshape(batches, num_nodes, -1)
        return Z

    def _calculate_reset_gate(self, X, H, Lambda):
        batches = X.shape[0]
        num_nodes = X.shape[1]
        R = self.conv_x_r(X.reshape(batches * num_nodes, -1),Lambda)
        R = R + self.conv_h_r(H.reshape(batches * num_nodes, -1),Lambda)
        R = torch.sigmoid(R).reshape(batches, num_nodes, -1)
        return R

    def _calculate_candidate_state(self, X, H, R, Lambda):
        batches = X.shape[0]
        num_nodes = X.shape[1]
        H_tilde = self.conv_x_h(X.reshape(batches * num_nodes, -1),Lambda)
        H_tilde = H_tilde + self.conv_h_h((H * R).reshape(batches * num_nodes, -1),Lambda)
        H_tilde = torch.tanh(H_tilde).reshape(batches, num_nodes, -1)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        Lambda: torch.FloatTensor,
        H: torch.FloatTensor = None,

    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features:  [Batch x SequenceLen x nodes x in_channels].
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden initial state matrix for all nodes. [Batch x out_channels]
            * **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **States** *(PyTorch Float Tensor)* - All hidden state matrix for all nodes along the sequence.
        """
        L = X.shape[1]
        H = self._set_hidden_state(X, H)
        States = []
        for i in range(L):
            Z = self._calculate_update_gate(X[:,i,:,:], H, Lambda)
            R = self._calculate_reset_gate(X[:,i,:,:], H, Lambda)
            H_tilde = self._calculate_candidate_state(X[:,i,:,:], H, R, Lambda)
            H = self._calculate_hidden_state(Z, H, H_tilde)
            States.append(H)
        return torch.stack(States, dim=1), H