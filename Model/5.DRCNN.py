class DRCNN(nn.Module):
    def __init__(self, width_factor: int, drop: float, in_channels: int, labels: int):
        super(DRCNN, self).__init__()

        self.filters = [16, 1 * 16 * width_factor, 2 * 16 * width_factor, 4 * 16 * width_factor,6 * 16 * width_factor]


        self.conv = nn.Conv1d(in_channels, self.filters[0], 8, stride=3, padding=1, bias=False)
        self.block1 = Block(self.filters[0], self.filters[1], 3, drop)
        self.block2 = Block(self.filters[1], self.filters[2], 3, drop)
        self.block3 = Block(self.filters[2], self.filters[3], 3, drop)
        self.block4 = Block(self.filters[3], self.filters[4], 3, drop)
        self.normalization4 = nn.BatchNorm1d(self.filters[4])
        self.activation5 = nn.GELU()
        self.avgpool6 = nn.AvgPool1d(kernel_size = 5)
        self.flattening7 = nn.Flatten()

        self.res_conv1 = nn.Conv1d(self.filters[0],self.filters[2],kernel_size= 7 ,stride= 9,bias = False,padding = 1)
        self.bn_res_conv1 = nn.BatchNorm1d(num_features=self.filters[2], momentum=0.99, eps=1e-3)
        self.gelu = nn.GELU()

        self.res_conv2 = nn.Conv1d(self.filters[2],self.filters[4],kernel_size= 7 ,stride= 9,bias = False,padding = 2)
        self.bn_res_conv2 = nn.BatchNorm1d(num_features=self.filters[4], momentum=0.99, eps=1e-3)


        self.fcc = nn.Linear(1920, 200)
        self.fcc_drop = nn.Dropout(0.5)
        self.fcc_norm = nn.LayerNorm(200)
        self.fcc_act = nn.GELU()


        self.classification8 = nn.Linear(200, out_features=labels)


    def forward(self, x):

        output = self.conv(x)

        res = self.res_conv1(output)
        res = self.bn_res_conv1(res)
        res = self.gelu(res)


        output = self.block1(output)
        output = self.block2(output)

        output = output + res

        res = self.res_conv2(output)
        res = self.bn_res_conv2(res)
        res = self.gelu(res)

        output = self.block3(output)
        output = self.block4(output)

        output = output + res


        output = self.normalization4(output)
        output = self.activation5(output)

        output = self.avgpool6(output)
        output = self.flattening7(output)


        output = self.fcc(output)
        output = self.fcc_drop(output)
        output = self.fcc_norm(output)
        output = self.fcc_act(output)


        output = self.classification8(output)
        output = nn.functional.log_softmax(output,dim = 1)
        #output = torch.sigmoid(output)
        return output