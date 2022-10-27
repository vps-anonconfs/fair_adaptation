import torch


def get_network(network_name, **model_kwargs):
    network_name = network_name.lower()
    if network_name == 'ffn':
        return FFN(**model_kwargs)
    raise AssertionError(f"Network name: {network_name} not supported!!")


class BaseModel(torch.nn.Module):
    @property
    def num_classes(self):
        raise NotImplementedError()

    # the following two properties are needed for FairCertModule.IntervalBoundForward for test split metrics
    @property
    def layers(self):
        """
        Should return the names of layers used
        """
        raise NotImplementedError()

    @property
    def activations(self):
        """
        Return the activations at each layer
        """
        raise NotImplementedError()


class FFN(BaseModel):
    def __init__(self, input_dimension: int, num_cls: int, hidden_dim: int = 128, hidden_lay: int = 1):
        super().__init__()
        _out = hidden_dim if hidden_lay > 0 else num_cls
        layers = [torch.nn.Linear(in_features=input_dimension, out_features=_out)]
        if hidden_lay > 0:
            layers.append(torch.nn.ReLU())
            for hi in range(hidden_lay - 1):
                layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
                layers.append(torch.nn.ReLU())
                _out = hidden_dim
            layers.append(torch.nn.Linear(_out, num_cls))

        self.model = torch.nn.Sequential(*layers)
        self._num_classes = num_cls
        self._layers = ["linear"] * (hidden_lay + 1)
        self._activations = [torch.nn.ReLU()] * (hidden_lay + 1)
        self._input_dim = input_dimension

    def forward(self, x):
        return self.model(x)

    @property
    def input_dim(self):
        return self._input_dim
    
    @property
    def num_classes(self):
        return self._num_classes

    @property
    def layers(self):
        return self._layers

    @property
    def activations(self):
        return self._activations


if __name__ == '__main__':
    m = FFN(32, 2)
    logits = m(torch.zeros([32, 32]))
    print(f"shape: {logits.shape}")
