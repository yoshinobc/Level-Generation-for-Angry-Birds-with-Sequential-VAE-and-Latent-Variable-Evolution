import numpy as np
from chainer import Variable
import chainer.functions as F
from chainer import cuda
from chainer.cuda import cupy as xp

def predict_random(prob):
    probability = cuda.to_cpu(prob.data)[0].astype(np.float64)
    probability /= np.sum(probability)
    index = np.random.choice(range(len(probability)), p=probability)
    return index

def predict(vae, batch_size, z=None, rand_flag=False):
    if z is None:
        z = Variable(xp.random.normal(0, 1, (batch_size, vae.n_latent)).astype(xp.float32))

    vae.dec.hx = F.reshape(vae.ld_h(z), (1, batch_size, 2*vae.hidden_size))
    vae.dec.cx = F.reshape(vae.ld_c(z), (1, batch_size, 2*vae.hidden_size))

    t = [[bi] for bi in [1] * batch_size]
    t = vae.make_embed_batch(t)
    ys_d = vae.dec(t, train=False)
    ys_w = [vae.h2w(y) for y in ys_d]
    name_arr_arr = []
    if rand_flag:
        t = [predict_random(F.softmax(y_each)) for y_each in ys_w]
    else:
        t = [y_each.data[-1].argmax(0) for y_each in ys_w]
    name_arr_arr.append(t)
    t = [vae.embed(xp.array([t_each], dtype=xp.int32)) for t_each in t]
    count_len = 0
    while count_len < 30:
        ys_d = vae.dec(t, train=False)
        ys_w = [vae.h2w(y) for y in ys_d]
        if rand_flag:
            t = [predict_random(F.softmax(y_each)) for y_each in ys_w]
        else:
            t = [y_each.data[-1].argmax(0) for y_each in ys_w]
        name_arr_arr.append(t)
        t = [vae.embed(xp.array([t_each], dtype=xp.int32)) for t_each in t]
        count_len += 1

    tenti = np.array(name_arr_arr, dtype=np.int32).T

    lines = []
    for name in tenti:
        name = [vae.vocab.itos(nint) for nint in name]
        if "</s>" in name:
            line = name[:name.index("</s>")]
            lines.append(line)
        else:
            lines.append(name)
    return lines
