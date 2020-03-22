from collections import defaultdict
import numpy as np
np.random.seed(1)
import math

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"EOS":0,"UNK":1}
        self.index2word = {0: "EOS", 1:"UNK"}
        self.n_words = 2 # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

def load_embeddings(file, lang):
    emb_matrix = None
    emb_dict = dict()
    for line in open(file):
        if not len(line.split()) == 2:
            if "\t" in line:
                delimiter = "\t"
            else:
                delimiter = " "
            line_split = line.rstrip().split(delimiter)
            # extract word and vector
            word = line_split[0]
            x = np.array([float(i) for i in line_split[1:]])
            vector = (x /np.linalg.norm(x))
            embedding_size = vector.shape[0]
            emb_dict[word] = vector
    base = math.sqrt(6/embedding_size)
    emb_matrix = np.random.uniform(-base,base,(lang.n_words, embedding_size))
    for i in range(3, lang.n_words):
        word = lang.index2word[i]
        if word in emb_dict:
            emb_matrix[i] = emb_dict[word]
    return emb_matrix