data_file = pd.read_csv("intesect_colon.txt",sep = "\t")
list_columns = list(data_file.columns)

list_index = []
for i in list_columns:
    if (i.startswith("TCGA")) | (i.startswith("GSM")):
        list_index.append(i)

train_data = np.array(data_file[list_index].T)
label_positive = np.zeros((337,2))
label_positive[:,0] = 1
label_negative = np.zeros((182,2))
label_negative[:,1] = 1
train_label = np.concatenate([label_positive,label_negative])
#train_label = np.array([1] * 337 + [0] * 182)

np.random.seed(114)
np.random.shuffle(train_data)
np.random.seed(114)
np.random.shuffle(train_label)

thereshold = int(0.8 * len(train_data))
methyl_train = train_data[:thereshold]
label_train = train_label[:thereshold]
methyl_test = train_data[thereshold:]
label_test = train_label[thereshold:]

traindataset = OutfitDataset(methyl_train,label_train, transform=True)
traindata = torch.utils.data.DataLoader(traindataset, batch_size=64,
                        shuffle=True)
testdataset = OutfitDataset(methyl_test,label_test, transform=True)
testdata = torch.utils.data.DataLoader(testdataset, batch_size=64,
                        shuffle=True)
