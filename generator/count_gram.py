def count_gram(t, uni_gram, bi_gram):
    oldl = ""
    for t_ in t:
        if t_ in uni_gram.keys():
            uni_gram[t_] += 1
        else:
            uni_gram[t_] = 1
        cl = oldl + t_
        if cl in bi_gram.keys():
            bi_gram[cl] += 1
        else:
            bi_gram[cl] = 1
        oldl = t_
    return uni_gram, bi_gram

