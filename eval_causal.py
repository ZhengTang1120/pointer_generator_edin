from model import *
from utils import *
from train_cause import *
import json

if __name__ == '__main__':

    input_lang = Lang("input")
    rule_lang  = Lang("rule")
    trainning_set = list()
    
    with open('train.json') as f:
        raw_train = json.load(f)
    with open('test.json') as f:
        raw_test = json.load(f)

    for datapoint in raw_train:
        input_lang.addSentence(datapoint[2])
        if len(datapoint) > 5 and datapoint[5]:
            rule_lang.addSentence(datapoint[5])
    for pattern in rule_lang.word2index:
        if check_dp(pattern):
            dp_pattern.append(rule_lang.word2index[pattern])
        elif pattern == '${ trigger }' or (pattern.isalnum() and pattern.lower() == pattern and pattern not in ['outgoing', 'incoming', 'word', 'lemma', 'tag', 'trigger']):
            w_pattern.append(rule_lang.word2index[pattern])
        else:
            ot_pattern.append(rule_lang.word2index[pattern])

    # with open('test_x.json'%args.train, 'w') as f:
    #     f.write(json.dumps(raw_test))

    epoch = 15
    PATH = "model_cause_new3/%d"%int(epoch)
    encoder = torch.load(PATH+"/encoder")
    classifier = torch.load(PATH+"/classifier")
    decoder = torch.load(PATH+'/decoder')

    encoder.eval()
    classifier.eval()
    decoder.eval()

    evaluate(encoder, classifier, decoder, raw_test, input_lang, rule_lang)