from language import *

from model_tr import *

import random

import numpy as np

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

def splitsent(sentence, sentence2):
    for word in sentence.split():
        if word == 'i' and 'je ' in sentence2:
            yield 'je'
        else:
            yield word

def indexesFromSentence(lang, sentence, lang2, sentence2):
    sourceset = {}
    id2source = {}
    pg_mat = np.ones((len(sentence.split()) + 1, len(sentence.split()) + 1)) * 1e-10
    for i, word in enumerate(sentence.split()):
        if word not in sourceset:
            sourceset[word] = lang2.n_words + len(sourceset)
            id2source[sourceset[word]] = word
        pg_mat[sourceset[word]-lang2.n_words][i] = 1
    indexes = [lang.word2index[word] for word in sentence.split()]
    indexes2 = [sourceset[word] if word in sourceset else lang2.word2index[word] for word in list(splitsent(sentence2, sentence))]

    indexes.append(EOS_token)
    indexes2.append(EOS_token)
    return indexes, indexes2, pg_mat, id2source

def tensorFromIndexes(indexes):
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def train(input_tensor, target_tensor, pg_mat, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    teacher_forcing_ratio = 0.5

    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    print (encoder_outputs)


    encoder_hidden = encoder.initHidden()
    encoder_output, encoder_hidden = encoder(
            input_tensor, encoder_hidden)

    encoder_outputs = encoder_output.view(input_length, -1)

    print (encoder_outputs)
    exit()

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, pg_mat)
            print (decoder_output)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, pg_mat)
            print (decoder_output)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def evaluate(encoder, decoder, sentence, pg_mat, id2source, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromIndexes(sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_output, encoder_hidden = encoder(
            input_tensor, encoder_hidden)

        encoder_outputs = encoder_output.view(input_length, -1)

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, pg_mat)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                if topi.item() in output_lang.index2word:
                    decoded_words.append(output_lang.index2word[topi.item()])
                elif topi.item() in id2source:
                    decoded_words.append(id2source[topi.item()])
                else:
                    decoded_words.append('UNK')

            decoder_input = topi.squeeze().detach()

        return decoded_words

def evaluateRandomly(encoder, decoder, n=1):
    for i in range(n):
        pair = random.choice(pairs)
        if 'i ' in pair[1] and 'je ' in pair[0]:
            pair[1].replace('i ', 'je ')
        print('>', pair[0])
        print('=', pair[1])
        sentence, target, pg_mat, id2source = indexesFromSentence(input_lang, pair[0], output_lang, pair[1])
        output_words = evaluate(encoder, decoder, sentence, torch.tensor(pg_mat, dtype=torch.float, device=device), id2source)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

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

if __name__ == '__main__':

    trainning_set = list()
    for pair in pairs:
        source, target, pg_mat, id2source = indexesFromSentence(input_lang, pair[0], output_lang, pair[1])
        source_tensor = tensorFromIndexes(source)
        target_tensor = tensorFromIndexes(target)
        trainning_set.append((source_tensor, target_tensor, torch.tensor(pg_mat, dtype=torch.float, device=device)))

    learning_rate = 0.001
    hidden_size = 256

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    for _ in range(20):
        random.shuffle(trainning_set)

        print_loss_total = 0

        start = time.time()
        for i, data in enumerate(trainning_set):
            loss = train(data[0], data[1], data[2], encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)

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

        evaluateRandomly(encoder, decoder)