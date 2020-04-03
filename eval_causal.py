from model import *
from utils import *
from train_cause import *
import json

if __name__ == '__main__':

    input_lang = Lang("input")
    trainning_set = list()
    with open('training_data.json') as f:
        raw_train = json.load(f)

    random.shuffle(raw_train)
    test = raw_train[:3000]
    raw_train = raw_train[3000:]
    for datapoint in raw_train:
        input_lang.addSentence(datapoint[2])

    epoch = 9
    PATH = "model_cause/%d"%int(epoch)
    encoder = torch.load(PATH+"/encoder")
    classifier = torch.load(PATH+"/classifier")

    encoder.eval()
    classifier.eval()

    with open('parsed_test.json') as f:
        sents = json.load(f)['sentences']


    t = 0.0
    p = 0.0
    tp = 0.0

    for i, datapoint in enumerate(test):
        src_dict = dict()
        for dp in sents[i]['graphs']['universal-enhanced']['edges']:
            src_dict[dp['destination']] = dp['source']
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


                encoder_output, encoder_hidden = encoder(input_tensor)

                encoder_outputs  = encoder_output.view(input_length, -1)
                cause_vec        = encoder_outputs[cause_pos[0]:cause_pos[-1]+1]
                effect_vec       = encoder_outputs[effect_pos[0]:effect_pos[-1]+1]
                classify_output, cw, ew = classifier(cause_vec, effect_vec)
                cl = list()
                el = list()
                if len(cause_pos) == 1:
                    cl.append(datapoint[2][cause_pos[0]])
                else:
                    for w in cause_pos:
                        if w not in src_dict or src_dict[w] not in cause_pos:
                            cl.append(datapoint[2][w])
                if len(effect_pos) == 1:
                    el.append(datapoint[2][effect_pos[0]])
                else:
                    for w in effect_pos:
                        if w not in src_dict or src_dict[w] not in effect_pos:
                            el.append(datapoint[2][w])
                if cl == [] or el == []:
                    print (i, cause_pos, effect_pos)
                print (cw, datapoint[2][cause_pos[0]:cause_pos[-1]+1], cl)
                print (ew, datapoint[2][effect_pos[0]:effect_pos[-1]+1],  el)
                print ()
                if np.round(classify_output).item() == 1:
                    p += 1
                    if (np.round(classify_output).item()==label):
                        tp += 1
    print (tp, t, p, tp/t, tp/p)