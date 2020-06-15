import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from language import *
import random
# from torch_geometric.nn import GCNConv

SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cpu')#"cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, pretrained):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(pretrained, freeze=False)
        self.lemma_embedding = nn.Embedding(2, 5)
        self.rnn = nn.LSTM(305, hidden_size, bidirectional=True)
        
        # self.gcn = GCNConv(2 * self.hidden_size, 2 * self.hidden_size)

        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, input, cause_pos, effect_pos, edge_index):
        lemma = [1 if i in cause_pos or i in effect_pos else 0 for i in range(input.size(0))]
        lemma = torch.tensor(lemma, dtype=torch.long, device=device).view(-1, 1)
        lemma_embeded = self.lemma_embedding(lemma).view(-1, 1, 5)
        embedded = self.embedding(input).view(-1, 1, 300)
        embedded = torch.cat((embedded, lemma_embeded), dim=2)
        output, hidden = self.rnn(embedded)
        # output = self.gcn(output.view(-1, 2 * self.hidden_size), edge_index)
        outputs  = output.view(input.size(0), -1)
        cause_vec        = outputs[cause_pos[0]:cause_pos[-1]+1]
        effect_vec       = outputs[effect_pos[0]:effect_pos[-1]+1]
        cause, cw = self.event_summary(cause_vec)
        effect, ew = self.event_summary(effect_vec)
        return outputs, hidden, cause, effect, cw, ew

    def event_summary(self, event):
        attn_weights = F.softmax(torch.t(self.attn(event)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 event.unsqueeze(0))
        return attn_applied[0, 0], attn_weights

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Classifier, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, cause, effect):
        # cause, cw = self.event_summary(cause)
        # effect, ew = self.event_summary(effect)
        input  = torch.cat((cause, effect))
        hidden = self.hidden(input)
        output = self.sigmoid(self.out(hidden))
        return output

    # def event_summary(self, event):
    #     attn_weights = F.softmax(torch.t(self.attn(event)), dim=1)
    #     attn_applied = torch.bmm(attn_weights.unsqueeze(0),
    #                              event.unsqueeze(0))
    #     return attn_applied[0, 0], attn_weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size + 512, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size*2, bias=False)
        self.attn_combine = nn.Linear(self.hidden_size * 4, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.LSTM(5 * self.hidden_size, self.hidden_size, bidirectional=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.softmax = nn.LogSoftmax(dim=1)

        self.wh = nn.Linear(self.hidden_size * 2, 1, bias=False)
        self.ws = nn.Linear(self.hidden_size * 2, 1, bias=False)
        self.wx = nn.Linear(self.hidden_size, 1)

    def forward(self, input, hidden, encoder_outputs, cause, effect, pg_mat):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        output, hidden = self.rnn(torch.cat((embedded, cause.view(1, 1, -1), effect.view(1, 1, -1)), 2), hidden)

        attn_weights = F.softmax(
            torch.mm(
                self.attn(hidden[0].view( 1,-1)), torch.t(encoder_outputs)
                )
            , dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        p_gen = torch.sigmoid(self.wh(attn_applied[0]) + self.ws(hidden[0].view( 1,-1)) + self.wx(embedded[0]))[0,0]

        atten_p = torch.mm(attn_weights, pg_mat*(1-p_gen))

        output = torch.cat((hidden[0].view( 1,-1), attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = torch.tanh(output)
        

        output = F.softmax(self.out(output[0]), dim=1)
        # output = self.softmax(self.out(output[0]))
        output = output * p_gen
        output = torch.cat((output, atten_p),1)
        output = torch.log(output)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)