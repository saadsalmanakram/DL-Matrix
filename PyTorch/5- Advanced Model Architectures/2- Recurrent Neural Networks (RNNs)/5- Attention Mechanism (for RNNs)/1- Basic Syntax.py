import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)
    
    def forward(self, query, keys, values):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights, values)
        return context

class Seq2SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2SeqModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        context = self.attention(rnn_out[:, -1, :], rnn_out, rnn_out)
        out = self.fc(context)
        return out

# Example usage
input_size = 10
hidden_size = 20
output_size = 1
model = Seq2SeqModel(input_size, hidden_size, output_size)
