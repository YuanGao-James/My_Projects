"""
@Author     dizs@using.ai
@Date       2022-04-27
@Describe   MAS 生成器 + Signature 判别器 pipline
"""
import argparse
import logging
import random
from typing import List, Tuple

import torch
from clearml import Task

def setup_seed(seed):
    import numpy as np
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(100)

# pylint: disable = wrong-import-position, logging-fstring-interpolation
from model.dataset import MASDataset, get_dataset
from model.massignature import MASignature

logger = logging.getLogger(__name__)

def main():
    """main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-shape", type=Tuple, default=(10, 1), help="input shape of generator")
    parser.add_argument("--output-shape", type=Tuple, default=(10, 1), help="output shape of generator")
    parser.add_argument("--batch-size", type=int, default=1000, help="batch size of dataset")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--epochs", type=int, default=500, help="train epochs")
    parser.add_argument("--device", type=str, default="cuda", help="the device model train on")
    parser.add_argument("--logs-dir", type=str, default="logs", help="logs dir")
    parser.add_argument("--num-samples", type=List, default=[1, 5, 3], help="the num of traders. [noise, technical, fundamental]")
    parser.add_argument("--sigma-p", type=int, default=1, help="noise trader individual signal param")
    parser.add_argument("--sigma-i", type=int, default=9, help="noise trader public signal param")
    parser.add_argument("--mu1", type=float, default=100., help="fundamental trader Heston model param")
    parser.add_argument("--mu2", type=float, default=0.01, help="fundamental trader Heston model param")
    parser.add_argument("--nu1", type=float, default=0.1, help="fundamental trader Heston model param")
    parser.add_argument("--nu2", type=float, default=0.1, help="fundamental trader Heston model param")
    parser.add_argument("--price-init", type=float, default=100., help="inital price")
    parser.add_argument("--price-star-init", type=float, default=100., help="inital price-star")
    parser.add_argument("--rho", type=float, default=0.15, help="The coefficient between price and demand")
    parser.add_argument("--param-b", type=float, default=1.0, help="the param b of mas to fit")
    parser.add_argument("--param-c", type=float, default=1.0, help="the param c of mas to fit")
    parser.add_argument("--param-d", type=float, default=0.5, help="the param d of mas to fit")
    parser.add_argument("--param-init", type=bool, default=True, help="init param or random")
    parser.add_argument("--param-b-init", type=float, default=0.6, help="the init param b of mas")
    parser.add_argument("--param-c-init", type=float, default=0.6, help="the init param c of mas")
    parser.add_argument("--param-d-init", type=float, default=0.1, help="the init param d of mas")
    # parser.add_argument("--param-b-init", type=float, default=1.0, help="the init param b of mas")
    # parser.add_argument("--param-c-init", type=float, default=1.0, help="the init param c of mas")
    # parser.add_argument("--param-d-init", type=float, default=0.5, help="the init param d of mas")
    parser.add_argument("--project", type=str, default="MAS", help="the project of log save to clearml")
    parser.add_argument("--name", type=str, default="test", help="he name of log save to clearml")

    args = parser.parse_args()
    in_price, in_price_star, in_sigma, out_price = get_dataset(args)
    logger.info(f"[*] generate dataset done! dataset length: {len(in_price)}")

    model = MASignature(args=args)
    # sigs_pred = model.get_lr_sigs_pred(in_price, out_price)
    # logger.info("[*] train LinearRegression done!")

    dataset = MASDataset(in_price, in_price_star, in_sigma, out_price)
    model.train_GAN(dataset)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

# GY: 可视化
# import matplotlib.pyplot as plt
# ax = plt.subplot(2, 1, 1)
# plt.sca(ax)
# plt.plot(out[0])
# ax = plt.subplot(2, 1, 2)
# plt.sca(ax)
# plt.plot(gen_price.detach().numpy()[0])
# plt.show()