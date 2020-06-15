from model import *
from utils import *
import json
import os
import argparse
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from collections import defaultdict
import pickle

EOS_token = 0
dp_pattern = list()
w_pattern  = list()
ot_pattern = list()

def check_dp(pattern):
    udp_list = ["rcmod", "xsubj", "acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc", "ccomp", "clf", "compound", "conj", "cop", "csubj", "dep", "det", "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "list", "mark", "nmod", "nsubj", "nummod", "obj", "obl", "orphan", "parataxis", "punct", "reparandum", "root", "vocative", "xcomp"]
    for dp in udp_list:
        if dp in pattern:
            return True
    return False

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

def train(input_tensor, label_tensor, cause_pos, effect_pos, rule_info, gold, edge_index, encoder, classifier, 
    decoder, encoder_optimizer, classifier_optimizer, decoder_optimizer):

    criterion2 = nn.NLLLoss()
    criterion1 = nn.BCELoss()

    encoder.train()
    classifier.train()
    decoder.train()

    encoder_optimizer.zero_grad()
    classifier_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0

    encoder_outputs, encoder_hidden, cause_vec, effect_vec, cw, ew = encoder(input_tensor, cause_pos, effect_pos, edge_index)
    
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
            lsb  = False # left square bracket [
            part = 'trigger' # rule start with trigger
            prev = None
            # Without teacher forcing: use its own predictions as the next input
            for di in range(rule_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, cause_vec, effect_vec, pg_mat)
                # topi, decoded, lsb, part = get_topi(decoder_output, rule_lang, id2source, lsb, part, prev)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                loss += criterion2(decoder_output, rule_tensor[di])
                if decoder_input.item() == EOS_token:
                    break
    
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

def top_skipIds(topis, skids):
    for id in topis[0]:
        if id.item() not in skids:
            return id

def get_topi(decoder_output, rule_lang, id2source, lsb, part, prev):
    # topvs, topis = decoder_output.data.topk(decoder_output.size(1))
    # if topis[0][0].item() == EOS_token:
    #     # decoded_rule.append('<EOS>')
    #     return topis[0][0], None, None, part
    # topi = topis[0][0]
    # lsb_id = rule_lang.word2index['[']
    # rsb_id = rule_lang.word2index[']']
    # l_w_id = [rule_lang.word2index['lemma'], rule_lang.word2index['word']]
    # c_e_id = [rule_lang.word2index['cause: Entity'], rule_lang.word2index['effect: Entity']]
    # eq_id  = rule_lang.word2index['=']

    # skip_ids = list(range(rule_lang.n_words+len(id2source), decoder_output.size(1)))
    # dps      = dp_pattern
    # words    = w_pattern

    # for p in id2source:
    #     if check_dp(id2source[p]):
    #         dps.append(p)
    #     else:
    #         words.append(p)
    
    # if lsb:
    #     skip_ids.append(lsb_id)
    # else:
    #     skip_ids.append(rsb_id)

    # if part == 'word/lemma':
    #     skip_ids += dps
    # elif part == 'effect/cause':
    #     skip_ids += words

    # topi = top_skipIds(topis, skip_ids)

    # if topi.item() == rsb_id:
    #     lsb = False
    # if topi.item() == lsb_id:
    #     lsb = True
    # if topi.item() == eq_id and  prev in c_e_id :
    #     part = 'cause/effect'
    # if topi.item() == eq_id and  prev in l_w_id:
    #     part = 'word/lemma'

    topv, topi = decoder_output.topk(1)

    if topi.item() == EOS_token or topi.item() == prev:
        # decoded_rule.append('<EOS>')
        return topi, None, None, part
    elif topi.item() in rule_lang.index2word:
        decoded = rule_lang.index2word[topi.item()]
    elif topi.item() in id2source:
        decoded = id2source[topi.item()]+'_from_source'
    else:
        decoded = 'UNK'
        # decoded_rule.append('UNK')

    return topi, decoded, lsb, part


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
            # edge_index   = torch.tensor(datapoint[-1], dtype=torch.long, device=device)

            input = makeIndexes(input_lang, datapoint[2])
            input_tensor   = tensorFromIndexes(input)
            if datapoint[1]!='not_causal' and len(datapoint) <= 5:
                t += 1
            if datapoint[1] == 'not_causal':
                label = 0
            else:
                label = 1

            rule_ids, pg_mat, id2source = makeOutputIndexes(rule_lang, [], datapoint[2])
            pg_mat = torch.tensor(pg_mat, dtype=torch.float, device=device)

            cause_pos, effect_pos = datapoint[3], datapoint[4]

            with torch.no_grad():
                input_length = input_tensor.size(0)


                encoder_outputs, encoder_hidden, cause_vec, effect_vec, cw, ew = encoder(input_tensor, cause_pos, effect_pos)
                classify_output = classifier(cause_vec, effect_vec)
                classify_output = classify_output.detach()

                decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
                decoder_hidden = encoder_hidden
                decoded_rule = []

                lsb  = False # left square bracket [
                part = 'trigger' # rule start with trigger
                prev = None

                for di in range(220):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                            decoder_input, decoder_hidden, encoder_outputs, cause_vec, effect_vec, pg_mat)
                    
                    topi, decoded, lsb, part = get_topi(decoder_output, rule_lang, id2source, lsb, part, prev)

                    if decoded is not None:
                        decoded_rule.append(decoded)
                        prev = topi.item()
                        decoder_input = topi.squeeze().detach()
                    else:
                        break
                gold = True
                if len(datapoint)>6:
                    if len(datapoint)>7:
                        gold = datapoint[6]
                    else:
                        gold = False
                    rule = datapoint[5]
                    # decoded_rule.reverse()
                    decoded_rule = [token.replace('_from_source', '') for token in decoded_rule]
                    candidates.append(decoded_rule)
                    references.append([rule])
                
                if gold and np.round(classify_output).item() == 1:
                    p += 1
                    if (np.round(classify_output).item()==label):
                        tp += 1

    print ('result', tp, p, t, eval_rules(references, candidates))

def eval_rules(references, candidates):
    c = 0.0
    s = 0.0
    print (len(candidates), len(references))
    for i, r in enumerate(candidates):
        if r == references[i][0]:
            c += 1
        print ("cand", r)
        print ("ref ", references[i][0])
        s += sentence_bleu(references[i][0], r)
        print (sentence_bleu(references[i][0], r))
        # print('Cumulative 1-gram: %f' % sentence_bleu(references[i][0], r, weights=(1, 0, 0, 0)))
        # print('Cumulative 2-gram: %f' % sentence_bleu(references[i][0], r, weights=(0.5, 0.5, 0, 0)))
        # print('Cumulative 3-gram: %f' % sentence_bleu(references[i][0], r, weights=(0.33, 0.33, 0.33, 0)))
        # print('Cumulative 4-gram: %f' % sentence_bleu(references[i][0], r, weights=(0.25, 0.25, 0.25, 0.25)))
        print ()
    return c/len(candidates), s/len(candidates), corpus_bleu(references, candidates) #, weights=(1, 0, 0, 0)), corpus_bleu(references, candidates, weights=(0.5, 0.5, 0, 0)), corpus_bleu(references, candidates, weights=(0.33, 0.33, 0.33, 0)), corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25))

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument('lr')
    args = parser.parse_args()

    input_lang = Lang("input")
    rule_lang  = Lang("rule")
    trainning_set = list()

    with open('train_GCN.json') as f:
        raw_train = json.load(f)
    with open('test_GCN.json') as f:
        raw_test = json.load(f)

    for datapoint in raw_train:
        input_lang.addSentence(datapoint[2])
        if len(datapoint) > 6 and datapoint[5]:
            rule_lang.addSentence(datapoint[5])
    for pattern in rule_lang.word2index:
        if check_dp(pattern):
            dp_pattern.append(rule_lang.word2index[pattern])
        elif pattern.isalnum() and pattern.lower() == pattern and pattern not in ['outgoing', 'incoming', 'word', 'lemma', 'tag', 'trigger']:
            w_pattern.append(rule_lang.word2index[pattern])
        else:
            ot_pattern.append(rule_lang.word2index[pattern])
    
    with open("lang.pickle", "wb") as f:
        pickle.dump((input_lang, rule_lang, raw_test), f)

    for datapoint in raw_train:
        if len(datapoint[2]) < 512 and datapoint[1] != 'hastopic':
            input = makeIndexes(input_lang, datapoint[2])
            input_tensor   = tensorFromIndexes(input)
            if len(datapoint) > 6 and datapoint[5]:
                rule_ids, pg_mat, id2source = makeOutputIndexes(rule_lang, datapoint[5], datapoint[2])
                pg_mat = torch.tensor(pg_mat, dtype=torch.float, device=device)
                rule_tensor = tensorFromIndexes(rule_ids)
                rule = [rule_tensor, pg_mat, id2source]
                if len(datapoint)>7:
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

            label_tensor = torch.tensor([label], dtype=torch.float, device=device)
            edge_index   = torch.tensor(datapoint[-1], dtype=torch.long, device=device)
            trainning_set.append((input_tensor, label_tensor, datapoint[3], datapoint[4], rule, gold, edge_index))
    embeds = torch.FloatTensor(load_embeddings("glove.840B.300d.txt", input_lang))
    learning_rate = float(args.lr)
    hidden_size = 100
    
    encoder    = EncoderRNN(input_lang.n_words, hidden_size, embeds).to(device)
    classifier = Classifier(4 * hidden_size, hidden_size, 1).to(device)
    decoder    = AttnDecoderRNN(hidden_size, rule_lang.n_words, dropout_p=0.1).to(device)

    # encoder.cuda()
    # classifier.cuda()
    # decoder.cuda()

    encoder_optimizer    = optim.Adam(encoder.parameters(), lr=learning_rate)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    decoder_optimizer    = optim.Adam(decoder.parameters(), lr=learning_rate)

    for epoch in range(100):

        random.shuffle(trainning_set)
        total_loss = 0
        for i, data in enumerate(trainning_set):
            loss = train(data[0], data[1], data[2], data[3],
                     data[4], data[5], data[6],
                     encoder, classifier, decoder, encoder_optimizer, 
                     classifier_optimizer, decoder_optimizer)
            total_loss += loss

        print (total_loss)

        evaluate(encoder, classifier, decoder, raw_test, input_lang, rule_lang)
        os.mkdir("model_cause_GCN/%d"%epoch)
        PATH = "model_cause_GCN/%d"%epoch
        torch.save(encoder, PATH+"/encoder")
        torch.save(classifier, PATH+"/classifier")
        torch.save(decoder, PATH+"/decoder")