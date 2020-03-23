from model import *
from utils import *
import json
import random

def makeIndexes(lang, seq):
    indexes = [lang.word2index[word] if word in lang.word2index else 1 for word in seq]
    indexes.append(EOS_token)
    return indexes

def tensorFromIndexes(indexes):
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def train(input_tensor, label_tensor, cause_pos, effect_pos, encoder, classifier, 
    encoder_optimizer, classifier_optimizer, criterion):
    encoder.train()
    classifier.train()

    encoder_optimizer.zero_grad()
    classifier_optimizer.zero_grad()

    input_length = input_tensor.size(0)

    loss = 0

    encoder_output, encoder_hidden = encoder(input_tensor)

    encoder_outputs  = encoder_output.view(input_length, -1)
    cause_vec        = encoder_outputs[cause_pos]
    effect_vec       = encoder_outputs[effect_pos]
    classify_output = classifier(encoder_outputs[-1], cause_vec, effect_vec)
    loss = criterion(classify_output, label_tensor)

    loss.backward()

    encoder_optimizer.step()
    classifier_optimizer.step()

    return loss.item()

def evaluate(encoder, classifier, test, input_lang):
    encoder.eval()
    classifier.eval()
    count1 = 0
    count0 = 0
    t1 = 0
    t0 = 0
    for datapoint in test:
        input = makeIndexes(input_lang, datapoint[2])
        input_tensor   = tensorFromIndexes(input)
        if datapoint[1]=='not_causal':
            label = 0
            count0+=1
        else:
            label = 1
            count1+=1

        cause_pos, effect_pos = datapoint[3][0], datapoint[4][0]

        with torch.no_grad():
            input_length = input_tensor.size(0)


            encoder_output, encoder_hidden = encoder(input_tensor)

            encoder_outputs  = encoder_output.view(input_length, -1)
            cause_vec        = encoder_outputs[cause_pos]
            effect_vec       = encoder_outputs[effect_pos]
            classify_output = classifier(encoder_outputs[-1], cause_vec, effect_vec)
            if label == 1:
                if (np.round(classify_output).item()==label):
                    t1 += 1
            else:
                if (np.round(classify_output).item()==label):
                    t0 += 1
    print (count0, t0)
    print (count1, t1)

if __name__ == '__main__':

    input_lang = Lang("input")
    trainning_set = list()
    with open('training_data.json') as f:
        raw_train = json.load(f)

    random.shuffle(raw_train)
    print(len(raw_train))
    raw_test  = raw_train[:3000]
    raw_train = raw_train[3000:]
    for datapoint in raw_train:
        input_lang.addSentence(datapoint[2])
    for datapoint in raw_train:
        input = makeIndexes(input_lang, datapoint[2])
        if datapoint[1] == 'not_causal':
            label = 0
        elif datapoint[1] != 'hastopic':
            label = 1
        input_tensor   = tensorFromIndexes(input)
        label_tensor = torch.tensor([label], dtype=torch.float, device=device)
        trainning_set.append((input_tensor, label_tensor, datapoint[3][0], datapoint[4][0]))
    embeds = torch.FloatTensor(load_embeddings("glove.840B.300d.txt", input_lang))

    learning_rate = 0.001
    hidden_size = 100

    encoder    = EncoderRNN(input_lang.n_words, hidden_size, embeds).to(device)
    classifier = Classifier(6 * hidden_size, hidden_size, 1).to(device)

    encoder_optimizer    = optim.Adam(encoder.parameters(), lr=learning_rate)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    criterion = nn.BCELoss()

    for epoch in range(20):

        random.shuffle(trainning_set)
        total_loss = 0
        for i, data in enumerate(trainning_set):

            loss = train(data[0], data[1], data[2], data[3],
                     encoder, classifier, encoder_optimizer, 
                     classifier_optimizer, criterion)
            total_loss += loss

        print (total_loss)

        evaluate(encoder, classifier, raw_test, input_lang)

