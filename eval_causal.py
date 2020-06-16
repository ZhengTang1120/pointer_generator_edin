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

    with open("lang.pickle") as f:
        input_lang, rule_lang, raw_test = pickle.load(f)

    # with open('test_x.json'%args.train, 'w') as f:
    #     f.write(json.dumps(raw_test))

    epoch = 6
    PATH = "model_cause_new/%d"%int(epoch)
    encoder = torch.load(PATH+"/encoder")
    classifier = torch.load(PATH+"/classifier")
    decoder = torch.load(PATH+'/decoder')

    encoder.eval()
    classifier.eval()
    decoder.eval()

    evaluate(encoder, classifier, decoder, raw_test, input_lang, rule_lang)