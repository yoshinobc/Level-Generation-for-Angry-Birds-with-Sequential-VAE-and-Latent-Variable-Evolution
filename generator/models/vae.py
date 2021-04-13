from chainer import Chain, Variable
import chainer.links as L
import chainer.functions as F
from chainer.cuda import cupy as xp
import numpy as np

from models.lstm import LSTM
from utils import make_vocabulary, denoise_input
from train_embed import train_embed

class VAE(Chain):
    def __init__(self, n_embed, n_layers, hidden_size, drop_ratio, n_latent, batch_size, train_file, epoch):
        super(VAE, self).__init__()

        self.n_embed = n_embed
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.drop_ratio = drop_ratio
        self.n_latent = n_latent
        self.batch_size = batch_size
        src_vocab, set_vocab, = make_vocabulary(train_file)
        self.vocab = src_vocab
        self.n_vocab = len(self.vocab)
        self.epoch = epoch

        w2v = train_embed(train_file, set_vocab, n_embed)
        with self.init_scope():
            self.embed = L.EmbedID(self.n_vocab, self.n_embed,
                            initialW=w2v.wv.vectors)
            self.enc_f = LSTM(self.n_layers, self.n_embed,
                        self.hidden_size, dropout=self.drop_ratio)
            self.enc_b = LSTM(self.n_layers, self.n_embed,
                        self.hidden_size, dropout=self.drop_ratio)

            self.le2_mu = L.Linear(4*self.hidden_size, self.n_latent)
            self.le2_ln_var = L.Linear(4*self.hidden_size, self.n_latent)
            self.ld_h = L.Linear(self.n_latent, 2*self.hidden_size)
            self.ld_c = L.Linear(self.n_latent, 2*self.hidden_size)

            self.dec = LSTM(self.n_layers, self.n_embed, 2 *
                    self.hidden_size, dropout=self.drop_ratio)
            self.h2w = L.Linear(2*self.hidden_size, self.n_vocab)

    def make_embed_batch(self, xs, reverse=False):
        if reverse:
            xs = [xp.asarray(x[::-1], dtype=xp.int32) for x in xs]
        elif not reverse:
            xs = [xp.asarray(x, dtype=xp.int32) for x in xs]
        section_pre = np.array([len(x) for x in xs[:-1]], dtype=np.int32)
        sections = np.cumsum(section_pre)
        xs = F.split_axis(self.embed(F.concat(xs, axis=0)), sections, axis=0)
        return xs

    def __call__(self, xs):
        mu, ln_var = self.encode(xs)
        z = F.gaussian(mu, ln_var)
        t = [[1]+x for x in xs]
        t_pred = [t_e[1:]+[2] for t_e in t]
        t_pred = [np.asarray(tp_e, dtype=np.int32) for tp_e in t_pred]
        t = denoise_input(self.vocab, t)
        t_vec = self.make_embed_batch(t)

        ys_w, t_all = self.decode(z, t_vec, t_pred)
        return mu, ln_var, ys_w, t_all

    def encode(self, xs):
        xs = [x + [2] for x in xs]
        xs_f = self.make_embed_batch(xs)
        xs_b = self.make_embed_batch(xs, True)
        self.enc_f.reset_state()
        self.enc_b.reset_state()
        ys_f = self.enc_f(xs_f)
        ys_b = self.enc_b(xs_b)
        mu = [self.le2_mu(F.concat((hx_f, cx_f, hx_b, cx_b), axis=1)) for hx_f, cx_f, hx_b, cx_b in
              zip(self.enc_f.hx, self.enc_f.cx, self.enc_b.hx, self.enc_b.cx)][0]
        ln_var = [self.le2_ln_var(F.concat((hx_f, cx_f, hx_b, cx_b), axis=1)) for hx_f, cx_f, hx_b, cx_b in
                  zip(self.enc_f.hx, self.enc_f.cx, self.enc_b.hx, self.enc_b.cx)][0]
        return mu, ln_var

    def decode(self, z, t_vec, t_pred):
        self.dec.hx = F.reshape(
            self.ld_h(z), (1, self.batch_size, 2 * self.hidden_size))
        self.dec.cx = F.reshape(
            self.ld_c(z), (1, self.batch_size, 2 * self.hidden_size))
        ys_d = self.dec(t_vec)
        ys_w = self.h2w(F.concat(ys_d, axis=0))
        t_all = []
        for t_each in t_pred:
            t_each_list = t_each.tolist()
            t_all += t_each_list
        t_all = xp.array(t_all, dtype=xp.int32)
        return ys_w, t_all

