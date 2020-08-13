import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from language import *
import random
from torch_geometric.nn import GCNConv

device = torch.device('cpu')#"cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, syn_size, hidden_size, pretrained):
        super(EncoderRNN, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(pretrained, freeze=False)
        self.lemma_embedding = nn.Embedding(2, 5)
        self.syn_embedding = nn.Embedding(syn_size, hidden_size)
        
        self.rnn = nn.LSTM(embedding_size + 5, hidden_size, bidirectional=True)

        self.linear  = nn.Linear(hidden_size * 2,   hidden_size)
        self.linear2 = nn.Linear(hidden_size + 610, hidden_size)

        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, input, syn_labels, cause_pos, effect_pos, edge_index):
        lemma = [1 if i in cause_pos or i in effect_pos else 0 for i in range(input.size(0))]
        lemma = torch.tensor(lemma, dtype=torch.long, device=device).view(-1, 1)
        lemma_embeded = self.lemma_embedding(lemma).view(-1, 1, 5)
        
        embedded = self.embedding(input).view(-1, 1, 300)
        embedded = torch.cat((embedded, lemma_embeded), dim=2)
        
        syn_embedded         = self.syn_embedding(syn_labels).view(syn_labels.size(0), -1)
        first_word_embedded  = embedded[edge_index[0],:,:].view(syn_labels.size(0), -1)
        second_word_embedded = embedded[edge_index[1],:,:].view(syn_labels.size(0), -1)
        syn_embedded         = torch.cat((syn_embedded, first_word_embedded, second_word_embedded), 1)
        syn_embedded         = self.linear2(syn_embedded)

        output, hidden = self.rnn(embedded)
        output  = self.linear(output)
        outputs = output.view(input.size(0), -1)
        
        cause_vec = outputs[cause_pos[0]:cause_pos[-1]+1]
        effect_vec = outputs[effect_pos[0]:effect_pos[-1]+1]
        cause, cw = self.event_summary(cause_vec)
        effect, ew = self.event_summary(effect_vec)
        return outputs, cause, effect, cw, ew, syn_embedded

    def event_summary(self, event):
        attn_weights = F.softmax(torch.t(self.attn(event)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 event.unsqueeze(0))
        return attn_applied[0, 0], attn_weights

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Classifier, self).__init__()

        self.hidden_size = hidden_size

        self.attn = nn.Linear(input_size, hidden_size, bias=False)
        self.gcn = GCNConv(hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size * 3, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, i, encoder_outputs, syn_embeddeds, cause, effect, edge_index):
        edge_weights = F.softmax(
            torch.mm(
                self.attn(encoder_outputs[i].view( 1,-1)), torch.t(syn_embeddeds)
                )
            , dim=1)
        edge_weights = edge_weights.squeeze(0)
        outputs = self.gcn(encoder_outputs, edge_index, edge_weights)
        output = torch.cat((outputs[i], cause, effect))
        output = self.sigmoid(self.out(output))
        return output


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size + 512, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.atten_gcn = GCNConv(hidden_size, hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.softmax = nn.LogSoftmax(dim=1)

        self.wh = nn.Linear(self.hidden_size, 1, bias=False)
        self.ws = nn.Linear(self.hidden_size, 1, bias=False)
        self.wx = nn.Linear(self.hidden_size, 1)

    def forward(self, input, hidden, encoder_outputs, syn_embeddeds, edge_index, pg_mat):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)

        attn_weights = F.softmax(
            torch.mm(
                self.attn(hidden[0].view( 1,-1)), torch.t(syn_embeddeds)
                )
            , dim=1)
        
        outputs = self.atten_gcn(encoder_outputs, edge_index, attn_weights.squeeze(0))

        attn_applied = outputs[0]

        p_gen = torch.sigmoid(self.wh(attn_applied) + self.ws(hidden[0].view( 1,-1)) + self.wx(embedded[0]))[0,0]+1e-7

        atten_p = torch.mm(attn_weights, pg_mat*(1-p_gen+1e-7))

        output = torch.cat((hidden[0].view(1 ,-1), attn_applied.view(1, -1)), 1)
        
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