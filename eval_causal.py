from model import *
from utils import *
from train_cause import *
import json

if __name__ == '__main__':

    input_lang = Lang("input")
    rule_lang  = Lang("rule")
    trainning_set = list()
    with open('training_data_x.json') as f:
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
    # with open('test_x.json'%args.train, 'w') as f:
    #     f.write(json.dumps(raw_test))
    raw_train = raw_train1[3000:] + raw_train2[50:]
    for datapoint in raw_train:
        input_lang.addSentence(datapoint[2])
        if len(datapoint) > 5:
            rule_lang.addSentence(datapoint[5])

    epoch = 19
    PATH = "model_cause_x/%d"%int(epoch)
    encoder = torch.load(PATH+"/encoder")
    classifier = torch.load(PATH+"/classifier")

    encoder.eval()
    classifier.eval()

    with open('test_x.json') as f:
        sents = json.load(f)


    evaluate(encoder, classifier, decoder, raw_test, input_lang, rule_lang)