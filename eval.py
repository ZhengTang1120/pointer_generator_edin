from train_bio import *
from sklearn.cluster import AffinityPropagation


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('jfile')
    parser.add_argument('jfile2')
    parser.add_argument('epoch')
    args = parser.parse_args()

    input_lang = Lang("input")
    pl1        = Lang("position")
    char_lang       = Lang("char")
    rule_lang  = Lang("rule")
    raw_train  = list()

    # input_lang, pl1, char, rule_lang, raw_train   = prepare_data(args.datadir, input_lang, pl1, char, rule_lang, raw_train)
    # input_lang, pl1, char, rule_lang, raw_train = prepare_data("pubmed2", input_lang, pl1, char, rule_lang, raw_train, "valids2.json")
    # input2_lang, pl2, char2, rule_lang2, raw_test = prepare_data(args.dev_datadir, valids="valids.json")


    input_lang, pl1, char_lang, rule_lang, raw_train   = prepare_data_from_json(args.jfile, input_lang, pl1, char_lang, rule_lang, raw_train)
    input2_lang, pl2, char_lang2, rule_lang2, raw_test = prepare_data_from_json(args.jfile2)

    deps2id = dict()
    deps = list()
    word2id = dict()
    words = list()
    for datapoint in raw_train:
        for i, word in enumerate(datapoint[0]):
            if i%2 == 0:
                if word not in deps:
                    deps.append(word)
                    deps2id[word] = input_lang.word2index[word]
            else:
                if word not in words:
                    words.append(word)
                    word2id[word] = input_lang.word2index[word]

    epoch = args.epoch
    # for epoch in range(20):
    PATH = "model_new/%d"%int(epoch)
    print (PATH)
    encoder = torch.load(PATH+"/encoder")
    decoder = torch.load(PATH+"/decoder")
    classifier = torch.load(PATH+"/classifier")

    encoder.eval()
    decoder.eval()
    classifier.eval()

    wvs = np.zeros((len(words), 100))
    dvs = np.zeros((len(deps), 100))
    for i, d in enumerate(deps):
        d_tensor = torch.tensor([deps2id[d]], dtype=torch.long, device=device).view(-1, 1)
        dvs[i] = d_tensor.numpy()
    for i, w in enumerate(words):
        w_tensor = torch.tensor([word2id[w]], dtype=torch.long, device=device).view(-1, 1)
        wvs[i] = w_tensor.numpy()

    X = np.concatenate((dvs, wvs), axis=0)
    print (X)
    af = AffinityPropagation(preference=-50).fit(X)
    labels = af.labels_
    print (labels)

    # evaluate(encoder, decoder, classifier, raw_test, input_lang, pl1, char_lang, rule_lang)
