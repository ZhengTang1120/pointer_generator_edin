from model_GCN import *
from utils import *
from train_new import *
import json

if __name__ == '__main__':

    input_lang = Lang("input")
    rule_lang  = Lang("rule")
    depen_lang = Lang("depen")
    trainning_set = list()
    
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

    # with open('test_x.json'%args.train, 'w') as f:
    #     f.write(json.dumps(raw_test))
    # for epoch in range(20):
    epoch = 5
    PATH = "model_cause_GCN/%d"%int(epoch)
    encoder = torch.load(PATH+"/encoder")
    classifier = torch.load(PATH+"/classifier")
    decoder = torch.load(PATH+'/decoder')

        print (eval(encoder, classifier, decoder, raw_dev, input_lang, depen_lang, rule_lang))