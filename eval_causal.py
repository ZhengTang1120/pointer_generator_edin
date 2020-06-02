from model import *
from utils import *
from train_cause import *
import json

if __name__ == '__main__':

    input_lang = Lang("input")
    rule_lang  = Lang("rule")
    trainning_set = list()
    with open('LDC_training.json') as f:
        raw_train1 = json.load(f)[:7000]
    with open('eidos_training.json') as f:
        raw_train2 = json.load(f)
    with open('eidos_extra.json') as f:
        raw_train2 = json.load(f)[:7000]
    rd = defaultdict(int)
    temp = list()
    with open("rules_cause.json") as f:
        rules = json.load(f)
        for datapoint in raw_train2:
            try:
                rd[datapoint[5]] = len(rules[datapoint[5]])
                datapoint[5] = rules[datapoint[5]]
                temp.append(datapoint)
            except KeyError:
                pass

    raw_train2 = temp[:]
    del temp
    random.shuffle(raw_train1)
    random.shuffle(raw_train2)
    raw_test  = raw_train1[:300]+raw_train2[:300]
    raw_train = raw_train1[300:]+raw_train2[300:]

    for datapoint in raw_train:
        input_lang.addSentence(datapoint[2])
        if len(datapoint) > 5 and datapoint[5]:
            rule_lang.addSentence(datapoint[5])
    for pattern in rule_lang.word2index:
        if check_dp(pattern):
            dp_pattern.append(rule_lang.word2index[pattern])
        elif pattern.isalnum() and pattern.lower() == pattern and pattern not in ['outgoing', 'incoming', 'word', 'lemma', 'tag', 'trigger']:
            w_pattern.append(rule_lang.word2index[pattern])
        else:
            ot_pattern.append(rule_lang.word2index[pattern])

    # with open('test_x.json'%args.train, 'w') as f:
    #     f.write(json.dumps(raw_test))

    epoch = 5
    PATH = "model_cause/%d"%int(epoch)
    encoder = torch.load(PATH+"/encoder")
    classifier = torch.load(PATH+"/classifier")
    decoder = torch.load(PATH+'/decoder')

    encoder.eval()
    classifier.eval()
    decoder.eval()

    evaluate(encoder, classifier, decoder, raw_test, input_lang, rule_lang)