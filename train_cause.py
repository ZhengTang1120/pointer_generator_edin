from model import *
from utils import *
import json
import os
import argparse

def makeIndexes(lang, seq):
    indexes = [lang.word2index[word] if word in lang.word2index else 1 for word in seq]
    indexes.append(EOS_token)
    return indexes

def tensorFromIndexes(indexes):
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def train(input_tensor, label_tensor, cause_pos, effect_pos, rule_info, gold, encoder, classifier, 
    decoder, encoder_optimizer, classifier_optimizer, decoder_optimizer):

    criterion2 = nn.NLLLoss()
    criterion1 = nn.BCELoss()

    encoder.train()
    classifier.train()
    decoder.train()

    encoder_optimizer.zero_grad()
    classifier_optimizer.zero_grad()

    # input_length = input_tensor.size(0)

    loss = 0

    encoder_outputs, encoder_hidden, cause_vec, effect_vec, cw, ew = encoder(input_tensor, cause_pos, effect_pos)
    # encoder_outputs  = encoder_output.view(input_length, -1)
    # cause_vec        = encoder_outputs[cause_pos[0]:cause_pos[-1]+1]
    # effect_vec       = encoder_outputs[effect_pos[0]:effect_pos[-1]+1]
    
    if gold:
        classify_output = classifier(cause_vec, effect_vec)
        loss += criterion1(classify_output, label_tensor)

    if len(rule_info)!=0:
        rule_tensor    = rule_info[0]#, pg_mat, id2source = rule_info
        rule_length    = rule_tensor.size(0)
        decoder_input  = torch.tensor([[0]], device=device)
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < 0.5 else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(rule_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, cause_vec, effect_vec)
                loss += criterion2(decoder_output, rule_tensor[di])
                topv, topi = decoder_output.topk(1)
                print (topi)
                decoder_input = rule_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(rule_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, cause_vec, effect_vec)
                topv, topi = decoder_output.topk(1)
                print (topi)
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

    for datapoint in test:
        if len(datapoint[2]) < 512 and datapoint[1] != 'hastopic':
            input = makeIndexes(input_lang, datapoint[2])
            input_tensor   = tensorFromIndexes(input)
            if datapoint[1]=='not_causal':
                label = 0
            else:
                label = 1
                t+=1

            cause_pos, effect_pos = datapoint[3], datapoint[4]

            with torch.no_grad():
                input_length = input_tensor.size(0)


                encoder_outputs, encoder_hidden, cause_vec, effect_vec, cw, ew = encoder(input_tensor, cause_pos, effect_pos)
                # encoder_outputs  = encoder_output.view(input_length, -1)
                # cause_vec        = encoder_outputs[cause_pos[0]:cause_pos[-1]+1]
                # effect_vec       = encoder_outputs[effect_pos[0]:effect_pos[-1]+1]
                classify_output = classifier(cause_vec, effect_vec)

                decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
                decoder_hidden = encoder_hidden
                decoded_rule = []
                for di in range(50):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                            decoder_input, decoder_hidden, encoder_outputs, cause_vec, effect_vec)
                    topv, topi = decoder_output.data.topk(1)
                    if topi.item() == EOS_token:
                        # decoded_rule.append('<EOS>')
                        break
                    else:
                        if topi.item() in rule_lang.index2word:
                            decoded_rule.append(rule_lang.index2word[topi.item()])
                        # elif topi.item() in id2source:
                        #     decoded_rule.append(id2source[topi.item()])
                        else:
                            decoded_rule.append('UNK')

                    decoder_input = topi.squeeze().detach()
                rule = datapoint[5] if len(datapoint)>5 else None
                print (decoded_rule)
                print (rule)
                
                # print (cw, datapoint[2][cause_pos[0]:cause_pos[-1]+1])
                # print (ew, datapoint[2][effect_pos[0]:effect_pos[-1]+1])
                # print ()
                if np.round(classify_output).item() == 1:
                    p += 1
                    if (np.round(classify_output).item()==label):
                        tp += 1
    print (tp, t, p)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('train')
    args = parser.parse_args()

    input_lang = Lang("input")
    rule_lang  = Lang("rule")
    trainning_set = list()
    with open('training_data_%s.json'%args.train) as f:
        raw_train1 = json.load(f)
    with open('eidos_training.json') as f:
        raw_train2 = json.load(f)
    with open("rules_cause.json") as f:
        rules = json.load(f)
        for datapoint in raw_train2:
            datapoint[5] = rules[datapoint[5]]
    random.shuffle(raw_train1)
    random.shuffle(raw_train2)
    raw_test  = raw_train1[:3000] + raw_train2[:50]
    # with open('test_%s.json'%args.train, 'w') as f:
    #     f.write(json.dumps(raw_test))
    raw_train = raw_train1[3000:] + raw_train2[50:]
    for datapoint in raw_train:
        input_lang.addSentence(datapoint[2])
        if len(datapoint) > 5:
            rule_lang.addSentence(datapoint[5])
    for datapoint in raw_train:
        if len(datapoint[2]) < 512 and datapoint[1] != 'hastopic':
            input = makeIndexes(input_lang, datapoint[2])
            input_tensor   = tensorFromIndexes(input)
            
            if len(datapoint) > 5:
                rule_tensor    = tensorFromIndexes(makeIndexes(rule_lang, datapoint[5]))
                rule = [rule_tensor]
                gold = True
            else:
                rule = []
                gold = True

            if datapoint[1] == 'not_causal':
                label = 0
            else:
                label = 1
            
            label_tensor = torch.tensor([label], dtype=torch.float, device=device)
            trainning_set.append((input_tensor, label_tensor, datapoint[3], datapoint[4], rule, gold))
    embeds = torch.FloatTensor(load_embeddings("glove.840B.300d.txt", input_lang))
    learning_rate = 0.001
    hidden_size = 100
    print (rule_lang.n_words)
    encoder    = EncoderRNN(input_lang.n_words, hidden_size, embeds).to(device)
    classifier = Classifier(4 * hidden_size, hidden_size, 1).to(device)
    decoder    = AttnDecoderRNN(hidden_size, rule_lang.n_words, dropout_p=0.1).to(device)

    encoder_optimizer    = optim.Adam(encoder.parameters(), lr=learning_rate)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    decoder_optimizer    = optim.Adam(decoder.parameters(), lr=learning_rate)

    for epoch in range(20):

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
        os.mkdir("model_cause_d%s/%d"%(args.train, epoch))
        PATH = "model_cause_d%s/%d"%(args.train, epoch)
        torch.save(encoder, PATH+"/encoder")
        torch.save(classifier, PATH+"/classifier")
        torch.save(decoder, PATH+"/decoder")
