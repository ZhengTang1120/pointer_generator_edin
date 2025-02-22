import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from language import *

device = torch.device("cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, pretrained):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(pretrained, freeze=False)
        self.rnn = nn.LSTM(hidden_size, hidden_size, bidirectional=True)

    def forward(self, input):
        embedded = self.embedding(input).view(-1, 1, self.hidden_size)
        output, hidden = self.rnn(embedded)
        return output, hidden

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.hidden = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, entity):
        input  = torch.cat((input, entity.repeat(1, input.size(0)).view(input.size(0), -1)), 1)
        hidden = torch.tanh(self.hidden(input))
        output = self.softmax(self.out(hidden))
        return output

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size*2, bias=False)
        self.attn_combine = nn.Linear(self.hidden_size * 4, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.LSTM(3 * self.hidden_size, self.hidden_size, bidirectional=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.wh = nn.Linear(self.hidden_size * 2, 1, bias=False)
        self.ws = nn.Linear(self.hidden_size * 2, 1, bias=False)
        self.wx = nn.Linear(self.hidden_size, 1)

    def forward(self, input, hidden, encoder_outputs, trigger, pg_mat):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        output, hidden = self.rnn(torch.cat((embedded, trigger.view(1, 1, -1)), 2), hidden)

        attn_weights = F.softmax(
            torch.mm(
                self.attn(hidden[0].view( 1,-1)), torch.t(encoder_outputs)
                )
            , dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        # p_gen = torch.sigmoid(self.wh(attn_applied[0]) + self.ws(hidden[0].view( 1,-1)) + self.wx(embedded[0]))[0,0]

        # atten_p = torch.mm(attn_weights, pg_mat*(1-p_gen))

        output = torch.cat((hidden[0].view( 1,-1), attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        

        output = F.softmax(self.out(output[0]), dim=1)
        # output = output * p_gen
        # output = torch.cat((output, atten_p),1)
        output = torch.log(output)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)