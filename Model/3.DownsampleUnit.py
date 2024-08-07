class DownsampleUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float):
        super(DownsampleUnit, self).__init__()
        self.norm_act = nn.Sequential(OrderedDict([
            ("0_normalization", nn.BatchNorm1d(in_channels)),
            ("1_activation", nn.GELU()),
        ]))
        self.block = nn.Sequential(OrderedDict([
            ("0_convolution", nn.Conv1d(in_channels, out_channels, 5, stride=stride, padding=2, bias=False)),
            ("1_normalization", nn.BatchNorm1d(out_channels)),
            ("2_activation", nn.GELU()),
            ("3_dropout", nn.Dropout(dropout, inplace=True)),
            ("4_convolution", nn.Conv1d(out_channels, out_channels, 5, stride=1, padding=2, bias=False)),
        ]))
        self.downsample = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        x = self.norm_act(x)
        return self.block(x) + self.downsample(x)