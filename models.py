from torch import nn


class SyntheticMLP(nn.Module):
    def __init__(self, input_dim=100, hidden_dims=(64, 32), n_out=2, bias=True):
        super(SyntheticMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=bias))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_out, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


SYNTHETIC_MLP_HIDDEN_DIMS = {
    "SyntheticMLP": (64, 32),
    "SyntheticMLPDeep": (256, 128, 64, 32),
}


def get_model(model, n_c, bias=False, dataset="SyntheticBinary", synthetic_dim=100):
    if model not in SYNTHETIC_MLP_HIDDEN_DIMS:
        raise ValueError("Only SyntheticMLP and SyntheticMLPDeep are supported in this pipeline")
    return SyntheticMLP(
        input_dim=synthetic_dim,
        hidden_dims=SYNTHETIC_MLP_HIDDEN_DIMS[model],
        n_out=n_c,
        bias=True,
    )
