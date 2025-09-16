from model import SimpleRNN
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import time


def generate_data(sample_num, seq_length):
    inputs = np.random.rand(sample_num, seq_length, 1)
    lables = np.sum(inputs, axis=1)
    return inputs, lables

input_size = 1
hidden_size = 100
output_size = 1

sample_num = 100000
seq_length = 2
 
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleRNN(input_size, hidden_size, output_size, device)
    model = model.to(device)

    inputs, lables = generate_data(sample_num, seq_length)
    inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
    lables = torch.tensor(lables, dtype=torch.float32).to(device)
    dataset = Data.TensorDataset(inputs, lables)
    dataloader = Data.DataLoader(dataset=dataset, 
                                 batch_size=32, 
                                 shuffle=True,
                                 num_workers=0)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epoch_num = 20
    time_total = 0
    model.train()
    for epoch in range(epoch_num):
        loss_total = 0
        begin = time.time()
        for i, (inputs, lables) in enumerate(dataloader):
            inputs = inputs.to(device)
            lables = lables.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, lables)
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.item() / inputs.size(0)
        end = time.time()
        time_total += (int)(end - begin)
        print("-"*20)
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, epoch_num, loss_total))
        print("Cost Time: {}m{}s".format(time_total//60, time_total%60))
        print("-"*20)
    
    torch.save(model, 'best.pth')
