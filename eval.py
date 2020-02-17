from train_bio import *


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


    epoch = args.epoch
    for epoch in range(20):
        PATH = "model_new/%d"%int(epoch)
        print (PATH)
        encoder = torch.load(PATH+"/encoder")
        decoder = torch.load(PATH+"/decoder")
        classifier = torch.load(PATH+"/classifier")

        encoder.eval()
        decoder.eval()
        classifier.eval()

        evaluate(encoder, decoder, classifier, raw_test, input_lang, pl1, char_lang, rule_lang)
