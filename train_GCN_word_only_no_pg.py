from model_GCN_word_only_no_pg import *
from utils import *
import json
import os
import argparse
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from collections import defaultdict

def makeIndexes(lang, seq):
    indexes = [lang.word2index[word] if word in lang.word2index else 1 for word in seq]
    return indexes

def tensorFromIndexes(indexes):
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def makeOutputIndexes(lang, output, labels):
    sourceset = {}
    id2source = {}
    pg_mat = np.ones((len(labels), len(labels))) * 1e-10
    for i, label in enumerate(labels):
        if label not in sourceset:
            sourceset[label] = lang.n_words + len(sourceset)
            id2source[sourceset[label]] = label
        pg_mat[sourceset[label]-lang.n_words][i] = 1
    # indexes = [sourceset[token] if token in sourceset else lang.word2index[token] for token in output]
    # indexes.reverse()
    indexes = [lang.word2index[word] if word in lang.word2index else 0 for word in output]
    indexes.append(EOS_token)
    return indexes, pg_mat, id2source

def train(datapoint, encoder, decoder, classifier, encoder_optimizer, decoder_optimizer, classifier_optimizer):
    
    input_tensor, cause_pos, effect_pos, trigger_pos, dep_tensor, edge_index, label_tensor, rule_info, gold = datapoint

    criterion2 = nn.NLLLoss()
    criterion1 = nn.BCELoss()

    encoder.train()
    classifier.train()
    decoder.train()

    encoder_optimizer.zero_grad()
    classifier_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0

    encoder_outputs, cause_vec, effect_vec, cw, ew = encoder(input_tensor, dep_tensor, cause_pos, effect_pos, edge_index)

    if gold:
        predicts = torch.empty(size=label_tensor.size(), device=device)
        for i in range(encoder_outputs.size(0)):
            context     = classifier(i, encoder_outputs, cause_vec, effect_vec, edge_index)
            predicts[i] = context
        loss = criterion1(predicts, label_tensor)

    if len(rule_info)!=0:
        rule_tensor, pg_mat, id2source = rule_info
        rule_length    = rule_tensor.size(0)
        if len(trigger_pos) != 0:
            trigger_vec     = encoder_outputs[trigger_pos[0]:trigger_pos[-1]+1]
            trigger_vec, tw = encoder.event_summary(trigger_vec)
        else:
            trigger_vec = encoder_outputs[0]
        trigger_vec = trigger_vec.view(1,1,-1)
        decoder_hidden = (trigger_vec, trigger_vec)
        decoder_input  = torch.tensor([[0]], device=device)
        for di in range(rule_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, 
                decoder_hidden, encoder_outputs, 
                edge_index, pg_mat)
            loss += criterion2(decoder_output, rule_tensor[di])
            decoder_input = rule_tensor[di]

        loss.backward()

        clipping_value = 1#arbitrary number of your choosing
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clipping_value)
        if gold:
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), clipping_value)
        if len(rule_info)!=0:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), clipping_value)

        encoder_optimizer.step()
        classifier_optimizer.step()
        decoder_optimizer.step()

        return loss.item()

def eval(encoder, classifier, decoder, raw, input_lang, depen_lang, rule_lang):
    encoder.eval()
    classifier.eval()
    decoder.eval()

    t = 0.0
    p = 0.0
    tp = 0.0
    tt = 0.0
    tc = 0.0

    references = []
    candidates = []

    for datapoint in raw:
        label      = 0 if datapoint[1] == 'not_causal' else 1
        sent       = datapoint[2]
        cause      = datapoint[3]
        effect     = datapoint[4]
        trigger    = datapoint[5]
        rule       = datapoint[6]
        edge_index = datapoint[7]
        edge_label = datapoint[8]
        gold       = datapoint[9]

        if label == 1:
            t += 1

        input        = makeIndexes(input_lang, sent)
        input_tensor = tensorFromIndexes(input)

        dep_labels = makeIndexes(depen_lang, edge_label)
        dep_tensor = tensorFromIndexes(dep_labels)

        edge_index   = torch.tensor(edge_index, dtype=torch.long, device=device)

        _, pg_mat, id2source = makeOutputIndexes(rule_lang, [], sent)
        pg_mat = torch.tensor(pg_mat, dtype=torch.float, device=device)

        with torch.no_grad():
            pred_trigger = []
            pred_label   = None
            decoded_rule = []

            encoder_outputs, cause_vec, effect_vec, cw, ew = encoder(input_tensor, dep_tensor, cause, effect, edge_index)

            for i in range(encoder_outputs.size(0)):
                context = classifier(i, encoder_outputs, cause_vec, effect_vec, edge_index)
                if i == 0:
                    if np.round(context).item() == 0:
                        pred_label = 0
                        break
                    else:
                        pred_label = 1
                else:
                    if np.round(context).item() == 1:
                        pred_trigger.append(i)

            a_set = set(trigger) 
            b_set = set(pred_trigger) 
            if len(trigger) != 0:
                tc += 1
            if len(a_set.intersection(b_set)) > 0: 
                tt += 1 
            if pred_label == 1:
                p += 1
                if label == 1:
                    tp += 1

                if len(pred_trigger) != 0:
                    trigger_vec     = encoder_outputs[pred_trigger[0]:pred_trigger[-1]+1]
                    trigger_vec, tw = encoder.event_summary(trigger_vec)
                else:
                    trigger_vec = encoder_outputs[0]
                trigger_vec = trigger_vec.view(1,1,-1)
                decoder_hidden = (trigger_vec, trigger_vec)
                decoder_input  = torch.tensor([[0]], device=device)
                for di in range(220):
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, 
                        decoder_hidden, encoder_outputs, 
                        edge_index, pg_mat)
                    topv, topi = decoder_output.topk(1)
                    if topi.item() == EOS_token:
                        break
                    else:
                        if topi.item() in rule_lang.index2word:
                            decoded_rule.append(rule_lang.index2word[topi.item()])
                        elif topi.item() in id2source:
                            decoded_rule.append(id2source[topi.item()])
                        else:
                            decoded_rule.append('UNK')

                    decoder_input = topi.squeeze().detach()

            if len(rule) != 0:
                # print (decoded_rule)
                # print (rule)
                # print (sentence_bleu([rule], decoded_rule))
                # print ()
                candidates.append(decoded_rule)
                references.append([rule])
    if p != 0:
        return tp, t, p, tp/t, tp/p, tt/tc, corpus_bleu(references, candidates)
    else:
        return tp, t, p, tp/t, 0, tt/tc, corpus_bleu(references, candidates)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('lr')
    parser.add_argument('seed')
    args = parser.parse_args()

    input_lang = Lang("input")
    depen_lang = Lang("depen")
    rule_lang  = Lang("rule")
    trainning_set = list()

    SEED = int(args.seed)

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cpu')#"cuda" if torch.cuda.is_available() else "cpu")

    with open('train_GCN.json') as f:
        raw_train = json.load(f)
    with open('dev_GCN.json') as f:
        raw_dev = json.load(f)

    for datapoint in raw_train:

        label      = datapoint[1]
        sent       = datapoint[2]
        cause      = datapoint[3]
        effect     = datapoint[4]
        trigger    = datapoint[5]
        rule       = datapoint[6]
        edge_index = datapoint[7]
        edge_label = datapoint[8]
        gold       = datapoint[9]

        input_lang.addSentence(sent)
        if len(rule)!=0:
            rule_lang.addSentence(rule)
        depen_lang.addSentence(edge_label)

    embeds, embedding_size = load_embeddings("glove.840B.300d.txt", input_lang)
    embeds = torch.FloatTensor(embeds)
    hidden_size = 100
    learning_rate = float(args.lr)

    encoder    = EncoderRNN(input_lang.n_words, embedding_size, depen_lang.n_words, hidden_size, embeds).to(device)
    classifier = Classifier(hidden_size, hidden_size, 1).to(device)
    decoder    = AttnDecoderRNN(hidden_size, rule_lang.n_words, dropout_p=0.1).to(device)

    encoder_optimizer    = optim.Adam(encoder.parameters(), lr=learning_rate)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    decoder_optimizer    = optim.Adam(decoder.parameters(), lr=learning_rate)

    for datapoint in raw_train:

        label      = datapoint[1]
        sent       = datapoint[2]
        cause      = datapoint[3]
        effect     = datapoint[4]
        trigger    = datapoint[5]
        rule       = datapoint[6]
        edge_index = datapoint[7]
        edge_label = datapoint[8]
        gold       = datapoint[9]

        input        = makeIndexes(input_lang, sent)
        input_tensor = tensorFromIndexes(input)

        dep_labels = makeIndexes(depen_lang, edge_label)
        dep_tensor = tensorFromIndexes(dep_labels)

        if label == 'not_causal':
            label_tensor = torch.tensor([0 for i in sent], dtype=torch.float, device=device)
        else:
            label_tensor = torch.tensor([1 if i in trigger+[0] else 0 for i,w in enumerate(sent)], dtype=torch.float, device=device)

        edge_index   = torch.tensor(edge_index, dtype=torch.long, device=device)

        if len(rule)!=0:
            rule_ids, pg_mat, id2source = makeOutputIndexes(rule_lang, rule, sent)
            pg_mat = torch.tensor(pg_mat, dtype=torch.float, device=device)
            rule_tensor = tensorFromIndexes(rule_ids)
            rule_info = [rule_tensor, pg_mat, id2source]
        else:
            rule_info = []

        trainning_set.append((input_tensor, cause, effect, trigger, dep_tensor, edge_index, label_tensor, rule_info, gold))

    for epoch in range(100):
        random.shuffle(trainning_set)
        for datapoint in trainning_set:
            train(datapoint, encoder, decoder, classifier, encoder_optimizer, decoder_optimizer, classifier_optimizer)

        os.mkdir("model__GCN_word_only_no_pg_%d/%d"%(SEED, epoch))
        PATH = "model__GCN_word_only_no_pg_%d/%d"%(SEED, epoch)
        torch.save(encoder, PATH+"/encoder")
        torch.save(classifier, PATH+"/classifier")
        torch.save(decoder, PATH+"/decoder")

        print (eval(encoder, classifier, decoder, raw_dev, input_lang, depen_lang, rule_lang))