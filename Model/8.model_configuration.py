device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = WideResNet(1,0.5,1,2)
model.apply(weight_init)

model.to(device)

criterion = nn.KLDivLoss(reduction="batchmean")
learning_rate = 0.000015
optimizer = torch.optim.Adam(params = model.parameters(),lr = learning_rate)

scheduler = StepLR(optimizer, learning_rate, 100)