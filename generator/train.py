import sys
import os
import numpy as np
import chainer
from chainer import serializers, optimizers
import chainer.functions as F

from loss import CustomLoss
from loader import get_batch
from predict import predict

sys.path.append("../")
from converter import txt2xml, xml2txt

def train(args, vae, model_name_base):
    vae.to_gpu()
    optimizer = optimizers.Adam()
    optimizer.setup(vae)

    #vae.embed.W.update_rule.enabled = False
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))

    with open(args.result_dir+"/loss_train.txt", "w") as f:
        f.write("loss_train, rec_loss, kl_loss\n")
    with open(args.result_dir+"/loss_valid.txt", "w") as f:
        f.write("loss_test, rec_loss, kl_loss\n")

    deconverter = txt2xml.txt2xml()

    criterion = CustomLoss(args.kl_zero_epoch, args.epoch, args.batch_size)
    print("Epoch train/loss_sum, train/rec_loss train/kl_loss valid/loss_sum valid/rec_loss valid/kl_loss")
    for e_i in range(args.epoch):
        loss_sum = 0
        rec_loss_sum = 0
        kl_loss_sum = 0
        count = 0
        for batch in get_batch(vae.vocab, args.train_file, args.batch_size, True):
            count += 1
            mu, ln_var, ys_w, t_all = vae(batch)
            loss, rec_loss = criterion(mu, ln_var, ys_w, t_all, e_i)
            loss_sum += loss.data
            rec_loss_sum += float(rec_loss)
            kl_loss_sum += (loss.data - rec_loss)
            vae.cleargrads()
            loss.backward()
            optimizer.update()
        train_loss = float(loss_sum/count)
        train_rec_loss = float(rec_loss_sum/count)
        train_kl_loss = float(kl_loss_sum/count)

        with open(args.result_dir+"/loss_train.txt", "a") as f:
            f.write(str(loss_sum/count) + "," + str(rec_loss_sum / count) + "," + str(kl_loss_sum/count) + "\n")

        count = 0
        for batch in get_batch(vae.vocab, args.valid_file, args.batch_size, False):
            count += 1
            mu, ln_var, ys_w, t_all = vae(batch)
            loss_sum += loss.data
            rec_loss_sum += float(rec_loss)
            kl_loss_sum += (loss.data - rec_loss)
            vae.cleargrads()
        valid_loss = float(loss_sum/count)
        valid_rec_loss = float(rec_loss_sum/count)
        valid_kl_loss = float(kl_loss_sum/count)
        print(f"{e_i}   {train_loss:.4f}    {train_rec_loss:.4f}    {train_kl_loss:.4f}    {valid_loss:.4f}    {valid_rec_loss:.4f}    {valid_kl_loss:.4f}")

        with open(args.result_dir+"/loss_valid.txt", "a") as f:
            f.write(str(loss_sum/count) + "," + str(rec_loss_sum / count) + "," + str(kl_loss_sum/count) + "\n")

        if e_i % 50 == 0:
            os.makedirs(args.result_dir+"/valid/sample_levels/" +
                        str(e_i), exist_ok=True)
            i = 0
            for tupl in get_batch(vae.vocab, args.valid_file, args.batch_size, False):
                mu, ln_var = vae.encode(tupl)
                z = F.gaussian(mu, ln_var)
                tenti = predict(vae, args.batch_size, z=z)
                vae.dec.reset_state()
                for _, name in enumerate(tenti):
                    text = deconverter.vector2xml(name)
                    with open(args.result_dir+"/valid/sample_levels/"+str(e_i)+"/level-"+str(i)+".xml", "w") as f:
                        f.write(text)
                    i += 1
            model_name = model_name_base.format(args.dataname, e_i)
            serializers.save_npz(model_name, vae)
