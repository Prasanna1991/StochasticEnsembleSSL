from config import config
import torch
import os
from torchvision import datasets, transforms
from dataloader import get_dataLoaderVAE, get_dataLoaderVAEEnsemble
from models import VAE, EnsembleClassifier
from torch.nn import functional as F
from utils import save_org_recon
import numpy as np
from torch.autograd import Variable
from sklearn.metrics.ranking import roc_auc_score

class VAETrainer:
    def __init__(self, config):
        self.config = config
        self.lr = config.lr
        self.batchsize = config.batch
        self.dataRoot = config.data_root
        self.z_dim = config.nz
        self.lr = config.lr

        if torch.cuda.is_available():
            self.use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            torch.cuda.set_device(config.deviceId)
        else:
            self.use_cuda = False
            torch.set_default_tensor_type('torch.FloatTensor')

        #Transforms for the data
        transformList = []
        transformList.append(transforms.ToTensor())
        transformSequence = transforms.Compose(transformList)

        self.dataLoaderTrain_L, self.dataLoaderTrain_U, self.dataLoaderVal, self.dataLoaderTest = \
            get_dataLoaderVAE(self.dataRoot, transformSequence, batch_size=self.batchsize)

        self.model = VAE(zdim=self.z_dim)
        if self.use_cuda: self.model = self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        recon_x = recon_x.view(-1, 3 * 128 * 128)
        x = x.view(-1, 3 * 128 * 128)
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + beta * KLD, BCE, KLD

    def train(self):
        prevLoss = 20000
        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            train_loss = 0
            bce_loss = 0
            kl_loss = 0

            for batch_idx, (data, _, _) in enumerate(self.dataLoaderTrain_U):
                if self.use_cuda:
                    data = data.cuda()
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data)
                loss, bce, kl = self.loss_function(recon_batch, data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                bce_loss += bce.item()
                kl_loss += kl.item()
                self.optimizer.step()

            print('====> Epoch: {} Average loss: {:.4f}   Recon: {:.4f}   KL: {:.4f}'.format(
                epoch, train_loss / len(self.dataLoaderTrain_U.dataset),
                       bce_loss / len(self.dataLoaderTrain_U.dataset),
                       kl_loss / len(self.dataLoaderTrain_U.dataset)))

            self.model.eval()
            reconstruction, _, _ = self.model(data)
            save_org_recon(data.data[0], reconstruction.data[0], epoch, "vae")

            test_loss = 0
            bce_val = 0
            kl_val = 0
            with torch.no_grad():
                for i, (data, _, _) in enumerate(self.dataLoaderVal):
                    if self.use_cuda:
                        data = data.cuda()
                    recon_batch, mu, logvar = self.model(data)
                    loss, bce, kl = self.loss_function(recon_batch, data, mu, logvar)
                    test_loss += loss.item()
                    bce_val += bce.item()
                    kl_val += kl.item()

            print('[Val]  ====> Epoch: {} Average loss: {:.4f}   Recon: {:.4f}   KL: {:.4f}'.format(
                epoch, test_loss / len(self.dataLoaderVal.dataset),
                       bce_val / len(self.dataLoaderVal.dataset),
                       kl_val / len(self.dataLoaderVal.dataset)))

            if test_loss < prevLoss:
                print('saving checkpoint .. for loss: {}'.format(test_loss))
                prevLoss = test_loss
                torch.save(self.model, 'VAE_CHX8.pt')

class EnsembleTrainer:
    def __init__(self, config):
        self.config = config
        self.lr = config.lr
        self.batchsize = config.batch
        self.dataRoot = config.data_root
        self.z_dim = config.nz
        self.lr = config.lr
        self.n_classes = config.n_classes

        if torch.cuda.is_available():
            self.use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            torch.cuda.set_device(config.deviceId)
        else:
            self.use_cuda = False
            torch.set_default_tensor_type('torch.FloatTensor')

        self.model = EnsembleClassifier(classCount=self.n_classes, zdim=self.z_dim)
        if self.use_cuda: self.model = self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.99), eps=1e-05, weight_decay=1e-5)

        self.dataLoaderTrain, self.dataLoaderTest = get_dataLoaderVAEEnsemble(labelled=500, batch_size=self.batchsize)
        self.ntrain = len(self.dataLoaderTrain.dataset)
        self.n_labeled = 7311

    def reparameterize(self,mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def ramp_up(self, epoch, max_epochs, max_val, mult):
        if epoch == 0:
            return 0.
        elif epoch >= max_epochs:
            return max_val
        return max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)

    def weight_schedule(self, epoch, max_epochs, max_val, mult, n_labeled, n_samples):
        max_val = max_val * (float(n_labeled) / n_samples)
        if epoch >= self.config.cut_off_epoch:
            return self.config.cut_off_value
        else:
            return self.ramp_up(epoch, max_epochs, max_val, mult)

    def temporal_loss(self, out1, out2, w, labels, labels_unit):
        def mse_loss(out1, out2):
            return F.mse_loss(out1, out2)

        def masked_crossentropy(out, labels, labels_unit):
            cond = (labels_unit[:, 0] >= 0)
            nnz = torch.nonzero(cond)
            nbsup = len(nnz)
            # check if labeled samples in batch, return 0 if none
            if nbsup > 0:
                masked_outputs = torch.index_select(out, 0, nnz.view(nbsup))
                masked_labels = labels[cond]
                loss = F.binary_cross_entropy(masked_outputs, masked_labels)
                # loss = F.cross_entropy(masked_outputs, masked_labels)
                return loss, nbsup
            return Variable(torch.FloatTensor([0.]).cuda(), requires_grad=False), 0

        sup_loss, nbsup = masked_crossentropy(out1, labels, labels_unit)
        unsup_loss = mse_loss(out1, out2)
        return sup_loss + w * unsup_loss, sup_loss, unsup_loss, nbsup

    def train(self):
        """
        Placeholder for prev epoch temporal outputs
        """
        Z = torch.zeros(self.ntrain, self.n_classes).float().cuda()  # intermediate values
        z = torch.zeros(self.ntrain, self.n_classes).float().cuda()  # temporal outputs
        outputs = torch.zeros(self.ntrain, self.n_classes).float().cuda()


        for epoch in range(self.config.epochs_ensemble):
            self.model.train()

            # evaluate unsupervised cost weight
            w = self.weight_schedule(epoch, self.config.max_epochs, self.config.max_val, self.config.ramp_up_mult, self.n_labeled, self.ntrain)
            print('unsupervised loss weight : {}'.format(w))

            # turn it into a usable pytorch object
            w = torch.autograd.Variable(torch.FloatTensor([w]).cuda(), requires_grad=False)

            l = []
            supl = []
            unsupl = []

            total_loss = 0
            for i, (x_m, x_lv, y, y_unit) in enumerate(self.dataLoaderTrain):

                x_m, x_lv, y = Variable(x_m.float()), Variable(x_lv.float()), Variable(y)
                y_unit = Variable(y_unit)

                if self.use_cuda:
                    x_m, x_lv, y = x_m.cuda(), x_lv.cuda(), y.cuda()
                    y_unit = y_unit.cuda()

                x = self.reparameterize(x_m, x_lv)
                self.optimizer.zero_grad()
                logits = self.model(x)

                """
                Temporal ensembling
                """
                zcomp = Variable(z[i * self.batchsize: (i + 1) * self.batchsize], requires_grad=False)
                loss, suploss, unsuploss, nbsup = self.temporal_loss(logits, zcomp, w, y, y_unit)

                # save outputs and losses
                outputs[i * self.batchsize: (i + 1) * self.batchsize] = logits.data.clone()
                l.append(loss.data[0])
                total_loss += loss.data[0]
                supl.append(nbsup * suploss.item())
                unsupl.append(unsuploss.item())

                # backprop
                loss.backward()
                self.optimizer.step()

                if i % 100 == 0:
                    print('[Ep: %d] Step [%d/%d], Temp loss: %.6f, Sup: %.6f, UnSup: %.6f' % (
                    epoch, i, len(self.dataLoaderTrain), loss.item(), suploss.item(), unsuploss.item()))

            m = len(self.dataLoaderTrain)
            print('Epoch {} report: Temp Loss: {}'.format(epoch, total_loss / m))

            # update temporal ensemble
            Z = self.config.alpha * Z + (1. - self.config.alpha) * outputs
            z = Z * (1. / (1. - self.config.alpha ** (epoch + 1)))

        torch.save(self.model, 'VAE_Ensemble_CHX8.pt')

    def test(self):
        self.model.eval()
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
        for i, (x_m, x_lv, y, y_unit) in enumerate(self.dataLoaderTest):

            x_m, x_lv, y = Variable(x_m.float()), Variable(x_lv.float()), Variable(y)

            if self.use_cuda:
                x_m, x_lv, y = x_m.cuda(), x_lv.cuda(), y.cuda()

            logits = self.model(x_m)

            outGT = torch.cat((outGT, y.detach()), 0)
            outPRED = torch.cat((outPRED, logits.detach()), 0)

        aurocIndividual = self.computeAUROC(outGT, outPRED, 14)
        aurocMean = np.array(aurocIndividual).mean()

        print("[Test]\t AUROC mean: {:.4f} \n".format(aurocMean))

    def computeAUROC(self, dataGT, dataPRED, classCount):
        outAUROC = []

        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()

        for i in range(classCount):
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))

        return outAUROC



if __name__ == '__main__':
    print('----------------- configuration -----------------')
    for k, v in vars(config).items():
        print('  {}: {}'.format(k, v))
    print('-------------------------------------------------')
    torch.backends.cudnn.benchmark = True           # boost speed.

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if config.mode == 1: #train VAE
        trainer = VAETrainer(config)
        trainer.train()

    if config.mode == 2: #train and test ensemble SSL
        trainer = EnsembleTrainer(config)
        trainer.train()
        trainer.test()