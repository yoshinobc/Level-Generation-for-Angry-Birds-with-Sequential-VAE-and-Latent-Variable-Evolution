import sys
import os
import argparse
import json

import numpy as np
import json
from chainer import serializers
from tqdm import tqdm

from train import train
from models.vae import VAE
from predict import predict

sys.path.append("../")
from converter import txt2xml, xml2txt
#from count_gram import count_gram

def generate(model_dir, model_epoch, is_random, sample_size):
    args = parser.parse_args()
    with open(f"{model_dir}/parameters.json", "r") as f:
        jsn = json.load(f)
    for jsn_key in jsn:
        setattr(args, jsn_key, jsn[jsn_key])
    vae = VAE(args.n_embed, args.n_layers, args.hidden_size, args.drop_ratio,
              args.n_latent, args.batch_size, args.train_file, args.epoch)
    vae.to_gpu()
    model_name = f"./{model_dir}/models/aibirds_word_{model_epoch}"
    serializers.load_npz(model_name, vae)
    deconverter = txt2xml.txt2xml()
    os.makedirs(f"{model_dir}/make_levels", exist_ok=True)
    bi_gram, uni_gram = {}, {}
    t = []
    for i in tqdm(range(sample_size)):
        tenti = predict(vae, 1, rand_flag=is_random)
        t.extend(tenti[0])
        vae.dec.reset_state()
        text = deconverter.vector2xml(tenti[0])
        with open(f"{model_dir}/make_levels/level-" + str(i) + ".xml", "w") as f:
            f.write(text)
    #uni_gram, bi_gram = count_gram(t, uni_gram, bi_gram)
    #print(len(uni_gram), len(bi_gram))

def main(args):
    vae = VAE(args.n_embed, args.n_layers, args.hidden_size, args.drop_ratio,
              args.n_latent, args.batch_size, args.train_file, args.epoch)
    os.makedirs(args.result_dir, exist_ok=True)
    with open(args.result_dir+"/parameters.json", mode="w") as f:
        json.dump(args.__dict__, f, indent=4)
    os.makedirs(args.result_dir+"/models", exist_ok=True)
    train(args, vae, args.result_dir+"/models/{}_{}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #training parameter
    parser.add_argument("--dataname", default = "aibirds_word", type=str, help="dataset name")
    parser.add_argument("--train_file", default="aibirds_word/train.txt", type=str, help="training file name")
    parser.add_argument("--valid_file", default="aibirds_word/valid.txt", type=str, help="validation file name")
    parser.add_argument("--epoch", default=501, type=int, help="training epoch")
    parser.add_argument("--n_embed", default=50,type=int, help="embedding dim")
    parser.add_argument("--hidden_size", default=400, type=int, help="hidden layer neurons")
    parser.add_argument("--n_latent", default=60, type=int, help="latent vector dim")
    parser.add_argument("--n_layers", default=1, type=int, help="layer num")
    parser.add_argument("--batch_size", default=20, type=int, help="batch size")
    parser.add_argument("--kl_zero_epoch", default=251, type=int, help="kl zero epoch")
    parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
    parser.add_argument("--drop_ratio", default=0.3, type=float, help="word drop rate")
    parser.add_argument("--grad_clip", default=3, type=int, help="gradient clipping threshold")
    #generate parameter
    parser.add_argument("--generate", default=False, action="store_true", help="if you want to generate level, add this flag")
    parser.add_argument("--model_dir", default="", type=str, help="the directory containing the model")
    parser.add_argument("--model_epoch", default=500, type=int, help="epoch number of the model to be generate")
    parser.add_argument("--is_random", default=False, action="store_true", help="if you want to make it a stochastic generator, add this flag")
    parser.add_argument("--sample_size", default=20, type=int, help="number of levels to generate")
    args = parser.parse_args()

    args.result_dir = f"{args.dataname}_{args.lr}_{args.n_layers}_{args.n_embed}_{args.hidden_size}_{args.n_latent}_{args.batch_size}_{args.drop_ratio}_{args.grad_clip}"

    if args.generate:
        generate(args.model_dir, args.model_epoch, args.is_random, args.sample_size)
    else:
        print("="*10, "start training", "="*10)
        main(args)
        print("="*10, "finish training", "="*10)
        print("="*10, "start generating", "="*10)
        generate(args.result_dir, 500, False, 20)
