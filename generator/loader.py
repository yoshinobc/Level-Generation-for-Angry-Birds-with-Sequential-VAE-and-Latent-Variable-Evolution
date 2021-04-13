import random

from utils import word_list

def get_batch(vocab, file_name, batch_size, is_shuffle):
    tt_now_list = [[vocab.stoi(char) for char in char_arr]
                   for char_arr in word_list(file_name)]
    ind_arr = list(range(len(tt_now_list)))
    if is_shuffle:
        random.shuffle(ind_arr)
    tt_now = (tt_now_list[ind] for ind in ind_arr)
    tt_gen = batch(tt_now, batch_size)
    for tt in tt_gen:
        if len(tt) == batch_size:
            yield tt


def batch(tt_now, batch_size):
    batch = []
    for l in tt_now:
        is_tuple = isinstance(l, tuple)
        batch.append(l)
        if len(batch) == batch_size:
            yield tuple(list(x) for x in zip(*batch)) if is_tuple else batch
            batch = []
    if batch:
        yield tuple(list(x) for x in zip(*batch)) if is_tuple else batch
