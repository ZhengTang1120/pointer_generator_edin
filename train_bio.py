from model import *
from bio_utils import *

import random
import numpy as np
import argparse
import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def tensorFromIndexes(indexes):
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def makeIndexes(lang, seq):
    for word in seq:
        if word not in lang.word2index:
            print (word)
    indexes = [lang.word2index[word] if word in lang.word2index else 1 for word in seq]
    indexes.append(EOS_token)
    return indexes

def makeOutputIndexes(lang, output, input):
    sourceset = {}
    id2source = {}
    pg_mat = np.ones((len(input) + 1, len(input) + 1)) * 1e-10
    # for i, word in enumerate(input):
    #     if word not in sourceset:
    #         sourceset[word] = lang.n_words + len(sourceset)
    #         id2source[sourceset[word]] = word
    #     pg_mat[sourceset[word]-lang.n_words][i] = 1
    # indexes = [sourceset[word] if word in sourceset else lang.word2index[word] for word in output]
    indexes = [lang.word2index[word] if word in lang.word2index else 0 for word in output]

    indexes.append(EOS_token)
    return indexes, pg_mat, id2source

def train(input_tensor, entity_pos, triggers_tensor, rule_infos, encoder, classifier, decoder, 
    encoder_optimizer, classifier_optimizer, decoder_optimizer, criterion):
    teacher_forcing_ratio = 0.5

    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)

    loss = 0

    encoder_output, encoder_hidden = encoder(
            input_tensor, encoder_hidden)

    encoder_outputs  = encoder_output.view(input_length, -1)
    entity_vec       = encoder_outputs[entity_pos]
    classify_outputs = classifier(encoder_outputs, entity_vec)

    loss = criterion(classify_outputs, triggers_tensor)

    for rule_tensor, pg_mat, id2source, triggers_pos in rule_infos:
        rule_length    = rule_tensor.size(0)
        decoder_input  = torch.tensor([[SOS_token]], device=device)
        trigger_vec    = encoder_outputs[triggers_pos]
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(rule_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, trigger_vec, pg_mat)
                loss += criterion(decoder_output, rule_tensor[di])
                decoder_input = rule_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(rule_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, trigger_vec, pg_mat)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                loss += criterion(decoder_output, rule_tensor[di])
                if decoder_input.item() == EOS_token:
                    break

    loss.backward()

    encoder_optimizer.step()
    classifier_optimizer.step()
    # decoder_optimizer.step()

    return loss.item()

def evaluate(encoder, decoder, datapoint, input_lang, pl1, char, rule_lang):
    print (datapoint[0])
    input        = makeIndexes(input_lang, datapoint[0])
    entity       = datapoint[1]
    entity_pos   = datapoint[2]
    triggers_pos = datapoint[3]
    triggers_lbl = datapoint[4]
    trigger_ids  = [input_lang.label2id[triggers_lbl[triggers_pos.index(i)]] if i in triggers_pos else 0 for i, _ in enumerate(input)]
    pos_list     = datapoint[5]
    chars        = [[char.word2index[c] if c in char.word2index else 0 for c in w] for w in datapoint[0]+["EOS"]]
    rules        = datapoint[6]

    with torch.no_grad():
        input_tensor   = tensorFromIndexes(input)
        trigger_ids  = [input_lang.label2id[triggers_lbl[triggers_pos.index(i)]] if i in triggers_pos else 0 for i, _ in enumerate(input)]
        rule_infos    = list()
        if (len(triggers_pos)!=len(rules)):
            print(triggers_pos, rules)
        for i, rule in enumerate(rules):
            rule_ids, pg_mat, id2source = makeOutputIndexes(rule_lang, rule, datapoint[0])
            rule_tensor                 = tensorFromIndexes(rule_ids)
            rule_infos.append((rule_tensor, torch.tensor(pg_mat, dtype=torch.float, device=device), id2source, triggers_pos[i]))

        encoder_hidden = encoder.initHidden()
        input_length = input_tensor.size(0)

        encoder_output, encoder_hidden = encoder(
                input_tensor, encoder_hidden)

        encoder_outputs  = encoder_output.view(input_length, -1)
        entity_vec       = encoder_outputs[entity_pos]
        classify_outputs = classifier(encoder_outputs, entity_vec)

        _, predicted = torch.max(classify_outputs, 1)


        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        for di in range(50):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, trigger_vec, pg_mat)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                if topi.item() in output_lang.index2word:
                    decoded_words.append(output_lang.index2word[topi.item()])
                # elif topi.item() in id2source:
                #     decoded_words.append(id2source[topi.item()])
                else:
                    decoded_words.append('UNK')

            decoder_input = topi.squeeze().detach()

        print (predicted, decoded_words)
        print (trigger_ids, rules)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('datadir')
    parser.add_argument('dev_datadir')
    parser.add_argument('outdir')
    args = parser.parse_args()

    input_lang = Lang("input")
    pl1        = Lang("position")
    char       = Lang("char")
    rule_lang  = Lang("rule")
    raw_train  = list()

    input_lang, pl1, char, rule_lang, raw_train   = prepare_data(args.datadir, input_lang, pl1, char, rule_lang, raw_train)
    input_lang, pl1, char, rule_lang, raw_train = prepare_data("pubmed_loc", input_lang, pl1, char, rule_lang, raw_train, "valids_loc.json")
    input2_lang, pl2, char2, rule_lang2, raw_test = prepare_data(args.dev_datadir, valids="valids.json")
    trainning_set = list()

    for datapoint in raw_train:
        input        = makeIndexes(input_lang, datapoint[0])
        entity       = datapoint[1]
        entity_pos   = datapoint[2]
        triggers_pos = datapoint[3]
        triggers_lbl = datapoint[4]
        trigger_ids  = [input_lang.label2id[triggers_lbl[triggers_pos.index(i)]] if i in triggers_pos else 0 for i, _ in enumerate(input)]
        pos_list     = datapoint[5]
        chars        = [[char.word2index[c] for c in w] for w in datapoint[0]+["EOS"]]
        rules        = datapoint[6]


        intput_tensor   = tensorFromIndexes(input)
        triggers_tensor = tensorFromIndexes(trigger_ids).view(-1)
        rule_infos    = list()
        if (len(triggers_pos)!=len(rules)):
            print(triggers_pos, rules)
        for i, rule in enumerate(rules):
            rule_ids, pg_mat, id2source = makeOutputIndexes(rule_lang, rule, datapoint[0])
            rule_tensor                 = tensorFromIndexes(rule_ids)
            rule_infos.append((rule_tensor, torch.tensor(pg_mat, dtype=torch.float, device=device), id2source, triggers_pos[i]))
        trainning_set.append((intput_tensor, entity_pos, triggers_tensor, rule_infos))
    
    learning_rate = 0.001
    hidden_size = 256

    encoder    = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder    = AttnDecoderRNN(hidden_size, rule_lang.n_words, dropout_p=0.1).to(device)
    classifier = Classifier(2 * hidden_size, hidden_size, len(input_lang.labels)).to(device)

    encoder_optimizer    = optim.SGD(encoder.parameters(), lr=learning_rate)
    classifier_optimizer = optim.SGD(classifier.parameters(), lr=learning_rate)
    decoder_optimizer    = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    for _ in range(20):
        print_loss_total = 0

        start = time.time()
        for i, data in enumerate(trainning_set):
            loss = train(data[0], data[1], data[2], data[3],
                     encoder, classifier, decoder, 
                     encoder_optimizer, classifier_optimizer, 
                     decoder_optimizer, criterion)

            print_loss_total += loss

            if (i+1) % 5000 == 0:
                print_loss_avg = print_loss_total / 5000
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, (i+1) / len(trainning_set)),
                        (i+1), (i+1) / len(trainning_set) * 100, print_loss_avg))
        print_loss_avg = print_loss_total / 853
        print_loss_total = 0
        print('%s (%d %d%%) %.4f' % (timeSince(start, (i+1) / len(trainning_set)),
                (i+1), (i+1) / len(trainning_set) * 100, print_loss_avg))

        for datapoint in raw_test:
            evaluate(encoder, decoder, datapoint, input_lang, pl1, char, rule_lang)