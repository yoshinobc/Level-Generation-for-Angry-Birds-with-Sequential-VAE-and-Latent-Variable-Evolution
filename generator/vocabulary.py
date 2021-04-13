from collections import defaultdict
import codecs


class Vocabulary():
  def __init__(self):
    pass

  def __len__(self):
    return self.__size

  def stoi(self, s):
    return self.__stoi[s]

  def itos(self, i):
    return self.__itos[i]

  def new(list_generator, size):
    self = Vocabulary()
    self.__size = size

    word_freq = defaultdict(lambda: 0)
    for words in list_generator:
      for word in words:
        word_freq[word] += 1

    self.__stoi = defaultdict(lambda: 0)
    self.__stoi['<unk>'] = 0
    self.__stoi['<s>'] = 1
    self.__stoi['</s>'] = 2
    self.__itos = [''] * self.__size
    self.__itos[0] = '<unk>'
    self.__itos[1] = '<s>'
    self.__itos[2] = '</s>'
    for i, (k, v) in zip(range(self.__size - 3), sorted(word_freq.items(), key=lambda x: -x[1])):
      self.__stoi[k] = i + 3
      self.__itos[i + 3] = k

    return self
