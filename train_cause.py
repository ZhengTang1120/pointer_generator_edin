from model import *
from utils import *
import json
import os
import argparse
from nltk.translate.bleu_score import corpus_bleu
from collections import defaultdict

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

    # indexes = [lang.word2index[word] if word in lang.word2index else 0 for word in output]

    indexes.append(EOS_token)
    return indexes, pg_mat, id2source

def train(input_tensor, label_tensor, cause_pos, effect_pos, rule_info, gold, encoder, classifier, 
    decoder, encoder_optimizer, classifier_optimizer, decoder_optimizer):

    criterion2 = nn.NLLLoss()
    criterion1 = nn.BCELoss()

    encoder.train()
    classifier.train()
    decoder.train()

    encoder_optimizer.zero_grad()
    classifier_optimizer.zero_grad()

    loss = 0

    encoder_outputs, encoder_hidden, cause_vec, effect_vec, cw, ew = encoder(input_tensor, cause_pos, effect_pos)
    
    if gold:
        classify_output = classifier(cause_vec, effect_vec)
        loss += criterion1(classify_output, label_tensor)

    if len(rule_info)!=0:
        rule_tensor, pg_mat, id2source = rule_info
        rule_length    = rule_tensor.size(0)
        decoder_input  = torch.tensor([[0]], device=device)
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < 0.5 else False
        pred = list()
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(rule_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, cause_vec, effect_vec, pg_mat)
                loss += criterion2(decoder_output, rule_tensor[di])
                topv, topi = decoder_output.topk(1)
                pred.append(topi.item())
                decoder_input = rule_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(rule_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, cause_vec, effect_vec, pg_mat)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                loss += criterion2(decoder_output, rule_tensor[di])
                if decoder_input.item() == EOS_token:
                    break
    loss.backward()

    encoder_optimizer.step()
    classifier_optimizer.step()
    decoder_optimizer.step()

    return loss.item()

def evaluate(encoder, classifier, decoder, test, input_lang, rule_lang):
    encoder.eval()
    classifier.eval()
    decoder.eval()

    t = 0.0
    p = 0.0
    tp = 0.0

    references = []
    candidates = []

    for datapoint in test:
        if len(datapoint[2]) < 512 and datapoint[1] != 'hastopic':
            input = makeIndexes(input_lang, datapoint[2])
            input_tensor   = tensorFromIndexes(input)
            if datapoint[1]=='not_causal':
                label = 0
            else:
                label = 1
                t+=1
            rule_ids, pg_mat, id2source = makeOutputIndexes(rule_lang, [], datapoint[2])
            pg_mat = torch.tensor(pg_mat, dtype=torch.float, device=device)

            cause_pos, effect_pos = datapoint[3], datapoint[4]

            with torch.no_grad():
                input_length = input_tensor.size(0)


                encoder_outputs, encoder_hidden, cause_vec, effect_vec, cw, ew = encoder(input_tensor, cause_pos, effect_pos)
                # classify_output = classifier(cause_vec, effect_vec)

                decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
                decoder_hidden = encoder_hidden
                decoded_rule = []
                for di in range(220):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                            decoder_input, decoder_hidden, encoder_outputs, cause_vec, effect_vec, pg_mat)
                    topv, topi = decoder_output.data.topk(1)
                    if topi.item() == EOS_token:
                            # decoded_rule.append('<EOS>')
                            break
                    else:
                        if topi.item() in rule_lang.index2word:
                            decoded_rule.append(rule_lang.index2word[topi.item()])
                        elif topi.item() in id2source:
                            decoded_rule.append(id2source[topi.item()])
                        else:
                            decoded_rule.append('UNK')

                    decoder_input = topi.squeeze().detach()
                gold = True
                if len(datapoint)>5:
                    if len(datapoint)>6:
                        gold = datapoint[6]
                    else:
                        gold = False
                    rule = datapoint[5]
                    print (decoded_rule)
                    candidates.append(decoded_rule)
                    print (rule)
                    references.append([rule])
                    print ()
                
                # print (cw, datapoint[2][cause_pos[0]:cause_pos[-1]+1])
                # print (ew, datapoint[2][effect_pos[0]:effect_pos[-1]+1])
                # print ()

                if gold and np.round(classify_output).item() == 1:
                    p += 1
                    if (np.round(classify_output).item()==label):
                        tp += 1

    print (eval_rules(references, candidates))#(tp, p, t, eval_rules(references, candidates))

def eval_rules(references, candidates):
    c = 0.0
    for i, r in enumerate(candidates):
        if r == references[i][0]:
            c += 1
        else:
            print ("cand", r)
            print ("ref ", references[i][0])
            print ()
    # print (len(candidates))
    return c/len(candidates), corpus_bleu(references, candidates)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('lr')
    args = parser.parse_args()

    input_lang = Lang("input")
    rule_lang  = Lang("rule")
    trainning_set = list()
    with open('LDC_training.json') as f:
        raw_train1 = json.load(f)
    with open('eidos_training.json') as f:
        raw_train2 = json.load(f)
    with open('eidos_extra.json') as f:
        raw_train2 = json.load(f)
    rd = defaultdict(int)
    with open("rules_cause.json") as f:
        rules = json.load(f)
        for datapoint in raw_train2:
            try:
                rd[datapoint[5]] = len(rules[datapoint[5]])
                datapoint[5] = rules[datapoint[5]]
            except KeyError:
                pass
    random.shuffle(raw_train1)
    random.shuffle(raw_train2)
    raw_test  = raw_train1[:300:] + raw_train2[:300]
    # with open('test_%s.json'%args.train, 'w') as f:
    #     f.write(json.dumps(raw_test))
    raw_train = raw_train1[300:] + raw_train2[300:5300]
    for datapoint in raw_train:
        input_lang.addSentence(datapoint[2])
        if len(datapoint) > 5 and datapoint[5]:
            rule_lang.addSentence(datapoint[5])

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
    embeds = torch.FloatTensor(load_embeddings("glove.840B.300d.txt", input_lang))
    learning_rate = float(args.lr)
    hidden_size = 100
    print (rule_lang.n_words)
    print (rule_lang.word2index)
    encoder    = EncoderRNN(input_lang.n_words, hidden_size, embeds).to(device)
    classifier = Classifier(4 * hidden_size, hidden_size, 1).to(device)
    decoder    = AttnDecoderRNN(hidden_size, rule_lang.n_words, dropout_p=0.1).to(device)

    encoder_optimizer    = optim.Adam(encoder.parameters(), lr=learning_rate)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    decoder_optimizer    = optim.Adam(decoder.parameters(), lr=learning_rate)

    for epoch in range(100):

        random.shuffle(trainning_set)
        total_loss = 0
        for i, data in enumerate(trainning_set):

            loss = train(data[0], data[1], data[2], data[3],
                     data[4], data[5],
                     encoder, classifier, decoder, encoder_optimizer, 
                     classifier_optimizer, decoder_optimizer)
            total_loss += loss

        print (total_loss)

        evaluate(encoder, classifier, decoder, raw_test, input_lang, rule_lang)
        # os.mkdir("model_cause_e%s/%d"%(args.train, epoch))
        # PATH = "model_cause_e%s/%d"%(args.train, epoch)
        # torch.save(encoder, PATH+"/encoder")
        # torch.save(classifier, PATH+"/classifier")
        # torch.save(decoder, PATH+"/decoder")
