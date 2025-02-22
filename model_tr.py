import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from language import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(-1, 1, self.hidden_size)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.hidden = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        hidden = self.hidden(input)
        output = self.softmax(self.out(hidden))
        return output

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size + MAX_LENGTH, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.wh = nn.Linear(self.hidden_size, 1, bias=False)
        self.ws = nn.Linear(self.hidden_size, 1, bias=False)
        self.wx = nn.Linear(self.hidden_size, 1)

    def forward(self, input, hidden, encoder_outputs, pg_mat):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        output, hidden = self.gru(embedded, hidden)

        attn_weights = F.softmax(
            torch.mm(
                self.attn(hidden[0]), torch.t(encoder_outputs)
                )
            , dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        p_gen = torch.sigmoid(self.wh(attn_applied[0]) + self.ws(hidden[0]) + self.wx(embedded[0]))[0,0]

        atten_p = torch.mm(attn_weights, pg_mat*(1-p_gen))

        output = torch.cat((hidden[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        

        output = F.softmax(self.out(output[0]), dim=1)
        output = output * p_gen
        output = torch.cat((output, atten_p),1)
        print ("-----------------------------")
        print (output)
        output = torch.log(output)
        print (output)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)