from gensim.models import Word2Vec
import codecs

def train_embed(train_file, set_vocab, n_embed):
    textlist = [['<unk>'], ["<s>"], ["</s>"]]
    wordlist = set()
    with codecs.open(train_file, "r", encoding="utf-8") as fp:
        for l in fp:
            for word in l.split("  ")[:-1]:
                wordlist.add(word)
                if word not in set_vocab:
                    print("not vocab", word)
            textlist.append(l.split("  ")[:-1])
    wordlist.add("<unk>")
    wordlist.add("<s>")
    wordlist.add("</s>")
    w2v = Word2Vec(textlist, sg=0, size=n_embed, window=2, min_count=0)
    return w2v
