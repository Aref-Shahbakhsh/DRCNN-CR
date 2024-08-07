class OutfitDataset(Dataset):

    def __init__(self, methyl,label, device=None,transform=None):
        self.methyl = methyl
        self.transform = transform
        self.label = label
        self.device = device
    def __len__(self):
        return len(self.label)
    def __getitem__(self,idx):
        methyl_data = self.methyl[idx]
        label_data = self.label[idx]

        if self.transform:
            methyl_data = torch.from_numpy(np.array(methyl_data))
            label_data = torch.from_numpy(np.array(label_data))

        return (methyl_data), \
               (label_data)