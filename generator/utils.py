import codecs
import random

from vocabulary import Vocabulary

def make_vocabulary(train_file):
    set_vocab = set()
    [[set_vocab.add(word) for word in word_arr]
     for word_arr in word_list(train_file)]
    n_vocab = len(set_vocab) + 3
    src_vocab = Vocabulary.new(
        word_list(train_file), n_vocab)
    return src_vocab, set_vocab


def word_list(file_name):
    with codecs.open(file_name, "r", encoding="utf-8") as fp:
        for l in fp:
            yield l.split("  ")[:-1]


def denoise_input(vocab, t, noise_rate=0.3):
    for i, t_e in enumerate(t):
        ind_arr = [t_i for t_i in range(1, min(len(t_e), 30))]
        random.shuffle(ind_arr)
        unk_ind_arr = ind_arr[:int(len(ind_arr) * noise_rate)]
        for unk_ind in unk_ind_arr:
            t[i][unk_ind] = vocab.stoi("<unk>")
    return t

