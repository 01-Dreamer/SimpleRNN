import torch
import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.f = nn.Linear(hidden_size, output_size)
 
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.rnn(x, h0)
        out = self.f(out[:, -1, :])
        return out

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleRNN(input_size=1, hidden_size=5, output_size=1, device=device)
    model = model.to(device)
    inputs = torch.tensor([[[0.1], [0.1]],[[0.1], [0.1]]], dtype=torch.float32)
    inputs = inputs.to(device)
    outputs = model(inputs)
    print(outputs)
