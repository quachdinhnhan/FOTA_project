import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    LSTM model for speech command classification (Exact Colab/Notebook Architecture).
    Input shape: [batch, 1, 80, 87]
    
    Includes BatchNorm2d with frozen parameters to prevent NaN gradients on Apple Silicon MPS.
    """

    def __init__(
        self,
        input_size: int = 80,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 10,
        in_channels: int = 1,
    ):
        super().__init__()
        self.seq = nn.Sequential(
            nn.BatchNorm2d(in_channels),
        )
        # Freeze BatchNorm parameters to prevent NaN gradients on MPS
        for param in self.seq.parameters():
            param.requires_grad = False

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: [batch, 1, 80, 87]
        """
        x = self.seq(x)
        _n, _c, _h, _w = x.shape
        _x = x.permute(0, 3, 2, 1)        # [batch, 1, 80, 87] -> [batch, 87, 80, 1]
        _x = _x.reshape(_n, _w, _h * _c)   # [batch, 87, 80]

        h0 = torch.zeros(self.num_layers, _n, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, _n, self.hidden_size, device=x.device)

        hsn, (hn, cn) = self.lstm(_x, (h0, c0))
        out = self.fc(hsn[:, -1, :])        # Use last hidden state for prediction
        return out
