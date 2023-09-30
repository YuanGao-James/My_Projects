"""
@Author     dizs@using.ai
@Date       2022-04-27
@Describe   MASignature
"""
import logging
import signatory
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
# pylint: disable = wrong-import-position, logging-fstring-interpolation
from model.dataset import MASDataset
from model.maslayer import Generator, Discriminator

torch.autograd.set_detect_anomaly(True)
logger = logging.getLogger(__name__)


class MASignature:
    """MASignature"""
    def __init__(self, args) -> None:
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.device = args.device
        self.discriminator = Discriminator()
        if self.device == "cuda":
            if torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                logger.info("cuda is not avalible! now use cpu.")
                self.device = "cpu"
        self.generator = Generator(args.num_samples, args.output_shape[0], args.rho, args.sigma_i,\
            args.sigma_p, args.mu1, args.mu2, args.nu1, args.nu2).to(self.device)
        if args.param_init:
            for name, _param in self.generator.named_parameters():
                if name == "mas_cell.noise_traders.param_b":
                    torch.nn.init.constant_(_param, args.param_b_init)
                if name == "mas_cell.noise_traders.param_c":
                    torch.nn.init.constant_(_param, args.param_c_init)
                if name == "mas_cell.technical_traders.param_d":
                    torch.nn.init.constant_(_param, args.param_d_init)
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.99)
        self.lr_model = LinearRegression()

    def signature(self, data: Tensor, scale: float=0.2):
        """计算路径签名"""
        # !!! 研究更好的路径签名方法
        # (batch_size, input_length, 1)
        # 1. scale
        data = data * scale
        # 2. cumsum
        data = data.cumsum(dim=1)
        # 3. cat lags
        data_lift = []
        for i in range(2):
            data_lift.append(data[:, i:i+9])
        data = torch.cat(data_lift, dim=-1)
        # 4. repeat
        # data = data.repeat((1, 2, 1))

        #原始
        # data = torch.concat([data[:, :-1, :], data[:, 1:, :]], dim=2)
        # data = torch.mean(data, dim=-1, keepdim=True)
        # @yangjn
        x_rep = torch.repeat_interleave(data, repeats=2, dim=1)
        data = torch.cat([x_rep[:, :-1], x_rep[:, 1:]], dim=2)

        return signatory.signature(data, 3, basepoint=False)

    def get_lr_sigs_pred(self, x: Tensor, y: Tensor, signature: bool=True):
        """
        Args:
            x (Tensor): past price
            y (Tensor): future price
            signature (bool, optional): 是否使用signature. Defaults to True.

        Returns:
            Tensor: 将x输入到(x, y)训练的线性回归模型得到sigs_pred
        """
        device = x.device
        if signature:
            x = self.signature(x)
            y = self.signature(y)
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        self.lr_model.fit(x, y)
        sigs_pred = torch.tensor(self.lr_model.predict(x)).float().to(device)
        return sigs_pred

    def train(self, dataset: MASDataset):
        """train"""
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        self.generator.train()
        train_loss = []
        param = []
        for epoch in range(self.epochs):
            _loss = []
            for _, sample in enumerate(dataloader):
                (in_price, in_price_star, in_sigma), out_price = sample
                in_price = in_price.to(self.device)
                in_price_star = in_price_star.to(self.device)
                in_sigma = in_sigma.to(self.device)
                out_price = out_price.to(self.device)
                gen_price, _, _ = self.generator(in_price, in_price_star, in_sigma)
                sigs_gen_price = self.signature(gen_price)
                # sigs_out_price = self.signature(out_price)
                loss = torch.norm(sigs_gen_price - out_price, p=2, dim=1).mean()
                return loss
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()
                logger.info(f"[*] epoch: {epoch}/{self.epochs}, loss: {loss}")
                _loss.append(np.mean(loss.item()))

            logger.info(f"[*] epoch: {epoch}/{self.epochs}, loss: {np.mean(_loss)}")
            train_loss.append(np.mean(_loss))
            for n in self.generator.parameters():
                param.append(float(n))

        # GY: 可视化参数（b，c，d）的结果
        param = np.array(param)
        param = np.reshape(param, (self.epochs, 3))
        import matplotlib.pyplot as plt
        ax = plt.subplot(2, 1, 1)
        plt.sca(ax)
        plt.plot(param[:, 0], label='b')
        plt.plot(param[:, 1], label='c')
        plt.plot(param[:, 2], label='d')
        plt.legend()
        ax = plt.subplot(2, 1, 2)
        plt.sca(ax)
        plt.plot(train_loss, label='loss')
        plt.legend()
        plt.show()

    def train_GAN(self, dataset: MASDataset):
        """train GAN"""
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        BCE_loss = torch.nn.BCELoss()

        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)

        Tensor = torch.FloatTensor

        G_loss = []
        D_loss = []
        param = []
        for epoch in range(self.epochs):
            _loss_g = []
            _loss_d = []
            for _, sample in enumerate(dataloader):
                (in_price, in_price_star, in_sigma), out_price = sample
                in_price = in_price.to(self.device)
                in_price_star = in_price_star.to(self.device)
                in_sigma = in_sigma.to(self.device)
                out_price = out_price.to(self.device)
                out_price = torch.reshape(out_price, (1000, 10))

                # labels
                from torch.autograd import Variable
                valid = Variable(Tensor(self.batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(self.batch_size, 1).fill_(0.0), requires_grad=False)
                fake = torch.reshape(fake, (1000, 1))

                gen_price, _, _ = self.generator(in_price, in_price_star, in_sigma)
                gen_price = torch.reshape(gen_price, (1000, 10))


                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()
                D_gen_price = self.discriminator(gen_price.detach())
                D_gen_price = torch.reshape(D_gen_price, (1000, 1))
                D_out_price = self.discriminator(out_price)
                D_out_price = torch.reshape(D_out_price, (1000, 1))
                real_loss = BCE_loss(D_out_price, valid)
                fake_loss = BCE_loss(D_gen_price, fake)
                print(real_loss, fake_loss)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward() #retain_graph=True
                optimizer_D.step()

                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()
                D_gen_price = self.discriminator(gen_price)
                D_gen_price = torch.reshape(D_gen_price, (1000, 1))
                g_loss = BCE_loss(D_gen_price, valid)
                g_loss.backward()
                optimizer_G.step()

                print("[Epoch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, self.epochs, d_loss.item(), g_loss.item()))
                # print("[Epoch %d/%d] [D loss: %f]" % (epoch, self.epochs, d_loss.item()))

                _loss_d.append(np.mean(d_loss.item()))
                _loss_g.append(np.mean(g_loss.item()))

            G_loss.append(np.mean(_loss_g))
            D_loss.append(np.mean(_loss_d))

            for n in self.generator.parameters():
                param.append(float(n))

        # GY: 可视化参数（b，c，d）的结果
        param = np.array(param)
        param = np.reshape(param, (self.epochs, 3))
        import matplotlib.pyplot as plt
        ax = plt.subplot(2, 1, 1)
        plt.sca(ax)
        plt.plot(param[:, 0], label='b')
        plt.plot(param[:, 1], label='c')
        plt.plot(param[:, 2], label='d')
        plt.legend()
        ax = plt.subplot(2, 1, 2)
        plt.sca(ax)
        plt.plot(G_loss, label='g_loss')
        plt.plot(D_loss, label='d_loss')
        plt.legend()
        plt.show()
