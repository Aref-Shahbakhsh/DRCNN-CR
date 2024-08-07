test_data = pd.read_csv("cfdna_colon_test_minmax.txt", sep = "\t")
test_data = np.array(test_data)
label_positive = np.zeros((148,2))
label_positive[:,0] = 1
label_negative = np.zeros((67,2))
label_negative[:,1] = 1
test_label = np.concatenate([label_positive,label_negative])

np.random.seed(55)
np.random.shuffle(test_data)
np.random.seed(55)
np.random.shuffle(test_label)

thereshold = int(0.5 * len(test_data))
finetune_methyl = test_data[:thereshold]
finetune_label = test_label[:thereshold]
test_methyl = test_data[thereshold:]
test_label = test_label[thereshold:]

fintune_dataset = OutfitDataset(finetune_methyl,finetune_label, transform=True)
fintune_data = torch.utils.data.DataLoader(fintune_dataset, batch_size=64, shuffle=True)
test_dataset = OutfitDataset(test_methyl,test_label, transform=True)
test_dataload = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
