import chainer.functions as F


class CustomLoss():
    def __init__(self, kl_zero_epoch, epoch, batch_size):
        self.kl_zero_epoch = kl_zero_epoch
        self.epoch = epoch
        self.batch_size = batch_size

    def __call__(self, mu, ln_var, ys_w, t_all, now_epoch):
        loss = F.softmax_cross_entropy(ys_w, t_all) / self.batch_size
        C = 0.01 * (now_epoch - self.kl_zero_epoch) / self.epoch
        rec_loss = loss
        if now_epoch > self.kl_zero_epoch:
            loss += C * F.gaussian_kl_divergence(mu, ln_var) / self.batch_size

        return loss, rec_loss.data
