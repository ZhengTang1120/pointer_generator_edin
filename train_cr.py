from model import *
from utils import *
import json
import os
import argparse
from nltk.translate.bleu_score import corpus_bleu
from collections import defaultdict

EOS_token = 0

def makeIndexes(lang, seq):
    indexes = [lang.word2index[word] if word in lang.word2index else 1 for word in seq]
    indexes.append(EOS_token)
    return indexes

def tensorFromIndexes(indexes):
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def makeOutputIndexes(lang, output, input):
    sourceset = {}
    id2source = {}
    pg_mat = np.ones((len(input) + 1, len(input) + 1)) * 1e-10
    for i, word in enumerate(input):
        if word not in sourceset:
            sourceset[word] = lang.n_words + len(sourceset)
            id2source[sourceset[word]] = word
        pg_mat[sourceset[word]-lang.n_words][i] = 1
    indexes = [sourceset[word] if word in sourceset else lang.word2index[word] for word in output]
    # indexes.reverse()
    # indexes = [lang.word2index[word] if word in lang.word2index else 0 for word in output]

    indexes.append(EOS_token)
    return indexes, pg_mat, id2source

def train(input_tensor, label_tensor, cause_pos, effect_pos, rule_info, gold, encoder, classifier, 
    decoder, encoder_optimizer, classifier_optimizer, decoder_optimizer):

    criterion1 = nn.BCELoss()

    encoder.train()
    classifier.train()

    encoder_optimizer.zero_grad()
    classifier_optimizer.zero_grad()

    loss = 0

    encoder_outputs, encoder_hidden, cause_vec, effect_vec, cw, ew = encoder(input_tensor, cause_pos, effect_pos)
    
    if gold:
        classify_output = classifier(cause_vec, effect_vec)
        loss += criterion1(classify_output, label_tensor)

    loss.backward()

    clipping_value = 1#arbitrary number of your choosing
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clipping_value)
    torch.nn.utils.clip_grad_norm_(classifier.parameters(), clipping_value)

    encoder_optimizer.step()
    classifier_optimizer.step()

    return loss.item()

def evaluate(encoder, classifier, decoder, test, input_lang, rule_lang):
    encoder.eval()
    classifier.eval()

    t = 0.0
    p = 0.0
    tp = 0.0

    references = []
    candidates = []

    for datapoint in test:
        input_tensor = datapoint[0]
        label        = datapoint[1].item()
        cause_pos    = datapoint[2]
        effect_pos   = datapoint[3]

        encoder_outputs, encoder_hidden, cause_vec, effect_vec, cw, ew = encoder(input_tensor, cause_pos, effect_pos)
    
        classify_output = classifier(cause_vec, effect_vec)
        classify_output = classify_output.detach()
        
        if label == 1:
            t += 1
        if np.round(classify_output).item() == 1:
            p += 1
            if (np.round(classify_output).item()==label):
                tp += 1

    print ('result', tp, p, t)

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument('lr')
    args = parser.parse_args()

    input_lang = Lang("input")
    rule_lang  = Lang("rule")
    trainning_set = list()
    with open('LDC_training.json') as f:
        raw_train = json.load(f)[:7000]
    for datapoint in raw_train:
        input_lang.addSentence(datapoint[2])
    random.shuffle(raw_train)

    for datapoint in raw_train:
        if len(datapoint[2]) < 512 and datapoint[1] != 'hastopic':
            input = makeIndexes(input_lang, datapoint[2])
            input_tensor   = tensorFromIndexes(input)
            if len(datapoint) > 5 and datapoint[5]:
                rule_ids, pg_mat, id2source = makeOutputIndexes(rule_lang, datapoint[5], datapoint[2])
                pg_mat = torch.tensor(pg_mat, dtype=torch.float, device=device)
                rule_tensor = tensorFromIndexes(rule_ids)
                rule = [rule_tensor, pg_mat, id2source]
                if len(datapoint)>6:
                    gold = datapoint[6]
                else:
                    gold = False
            else:
                rule = []
                gold = True

            if datapoint[1] == 'not_causal':
                label = 0
            else:
                label = 1

            temp = datapoint[5] if len(datapoint)>5 else []

            label_tensor = torch.tensor([label], dtype=torch.float, device=device)
            trainning_set.append((input_tensor, label_tensor, datapoint[3], datapoint[4], rule, gold))
    chunks = [trainning_set[i:i + 700] for i in range(0, len(trainning_set), 700)]
    embeds = torch.FloatTensor(load_embeddings("glove.840B.300d.txt", input_lang))
    learning_rate = float(args.lr)
    hidden_size = 100
    
    encoder    = EncoderRNN(input_lang.n_words, hidden_size, embeds).to(device)
    classifier = Classifier(4 * hidden_size, hidden_size, 1).to(device)

    encoder_optimizer    = optim.Adam(encoder.parameters(), lr=learning_rate)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    for epoch in range(100):

        for j, c in enumerate(chunks):
            trainning_set = list()
            for chunk in chunks[:j] + chunks[j+1:]:
                trainning_set += chunk
            dev_set = c
            random.shuffle(trainning_set)
            total_loss = 0
            for i, data in enumerate(trainning_set):
                loss = train(data[0], data[1], data[2], data[3],
                         data[4], data[5],
                         encoder, classifier, None, encoder_optimizer, 
                         classifier_optimizer, None)
                total_loss += loss

            evaluate(encoder, classifier, None, dev_set, input_lang, rule_lang)
