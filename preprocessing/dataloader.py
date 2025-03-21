import numpy as np
import torch

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.num_samples = len(samples)
        self.data = samples

    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        sample, label = self.data[idx]
        return sample, label

def MyDataLoader(train_file, test_file, batch_size, num_workers=1):
    print("----Loading dataset----")
    
    training = torch.load(train_file)
    validation = torch.load(test_file)
    
    train_dataset = MyDataset(training)
    eval_dataset = MyDataset(validation)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    y_train = [y for x, y in training]
    _, train_distr = np.unique(y_train, return_counts=True)
    weights = sum(train_distr)/train_distr
    sample_weights = torch.tensor(weights/sum(weights), dtype=torch.float32)

    print('Dataset: MAHNOB-HCI')
    print("#Traning samples: ", len(train_dataset))
    print("#Validation samples: ", len(eval_dataset))
    print("#Training distribution: ", train_distr)
    print("-------------------------")

    return train_loader, eval_loader, sample_weights