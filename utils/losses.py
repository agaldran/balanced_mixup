import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as stats2
import sys
# try:
#     from kornia.losses import FocalLoss
# except:
#     sys.exit('install kornia please')

def get_gauss_label(label, n_classes, amplifier, noise=0):
    n = n_classes*amplifier
    half_int = amplifier/2
    label_noise = np.random.uniform(low=-noise, high=noise)
    if label == 0:
        label_noise = np.abs(label_noise)
    if label == 4:
        label_noise = -np.abs(label_noise)
    label += label_noise
    label_new = half_int + label*amplifier
    gauss_label = stats2.norm.pdf(np.arange(n), label_new, half_int/2)
    gauss_label/=np.sum(gauss_label)
    return gauss_label

def get_gaussian_label_distribution(n_classes, std=0.5):
    cls = []
    for n in range(n_classes):
        cls.append(stats2.norm.pdf(range(n_classes), n, std))
    dists = np.stack(cls, axis=0)
    return dists
    # if n_classes == 3:
    #     CL_0 = stats2.norm.pdf([0, 1, 2], 0, std)
    #     CL_1 = stats2.norm.pdf([0, 1, 2], 1, std)
    #     CL_2 = stats2.norm.pdf([0, 1, 2], 2, std)
    #     dists = np.stack([CL_0, CL_1, CL_2], axis=0)
    #     return dists
    # if n_classes == 5:
    #     CL_0 = stats2.norm.pdf([0, 1, 2, 3, 4], 0, std)
    #     CL_1 = stats2.norm.pdf([0, 1, 2, 3, 4], 1, std)
    #     CL_2 = stats2.norm.pdf([0, 1, 2, 3, 4], 2, std)
    #     CL_3 = stats2.norm.pdf([0, 1, 2, 3, 4], 3, std)
    #     CL_4 = stats2.norm.pdf([0, 1, 2, 3, 4], 4, std)
    #     dists = np.stack([CL_0, CL_1, CL_2, CL_3, CL_4], axis=0)
    #     return dists
    # if n_classes == 6:
    #     CL_0 = stats2.norm.pdf([0, 1, 2, 3, 4, 5], 0, std)
    #     CL_1 = stats2.norm.pdf([0, 1, 2, 3, 4, 5], 1, std)
    #     CL_2 = stats2.norm.pdf([0, 1, 2, 3, 4, 5], 2, std)
    #     CL_3 = stats2.norm.pdf([0, 1, 2, 3, 4, 5], 3, std)
    #     CL_4 = stats2.norm.pdf([0, 1, 2, 3, 4, 5], 4, std)
    #     CL_5 = stats2.norm.pdf([0, 1, 2, 3, 4, 5], 5, std)
    #     dists = np.stack([CL_0, CL_1, CL_2, CL_3, CL_4, CL_5], axis=0)
    #     return dists
    # else:
    #     raise NotImplementedError

def cross_entropy_loss_one_hot(logits, target, reduction='mean'):
    logp = F.log_softmax(logits, dim=1)
    loss = torch.sum(-logp * target, dim=1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(
            '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')

def one_hot_encoding(label, n_classes):
    return torch.zeros(label.size(0), n_classes).to(label.device).scatter_(1, label.view(-1, 1), 1)

def label_smoothing_criterion(alpha=0.1, distribution='uniform', std=0.5, reduction='mean'):
    def _label_smoothing_criterion(logits, labels):
        n_classes = logits.size(1)
        device = logits.device
        # manipulate labels
        one_hot = one_hot_encoding(labels, n_classes).float().to(device)
        if distribution == 'uniform':
            uniform = torch.ones_like(one_hot).to(device)/n_classes
            soft_labels = (1 - alpha)*one_hot + alpha*uniform
        elif distribution == 'gaussian':
            dist = get_gaussian_label_distribution(n_classes, std=std)
            soft_labels = torch.from_numpy(dist[labels.cpu().numpy()]).to(device)
        else:
            raise NotImplementedError

        loss = cross_entropy_loss_one_hot(logits, soft_labels.float(), reduction)

        return loss

    return _label_smoothing_criterion

class CostSensitiveRegularizedLoss(nn.Module):
    def __init__(self,  n_classes=5, exp=2, normalization='softmax', reduction='mean', base_loss='gls', lambd=10):
        super(CostSensitiveRegularizedLoss, self).__init__()
        if normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        elif normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = None
        self.reduction = reduction
        x = np.abs(np.arange(n_classes, dtype=np.float32))
        M = np.abs((x[:, np.newaxis] - x[np.newaxis, :])) ** exp
        M /= M.max()
        if n_classes == 6: # if we have unclassifiable class, we do not penalize it
            M[:,-1] = 0
            M[-1, :] = 0

        self.M = torch.from_numpy(M)
        self.lambd = lambd
        self.base_loss = base_loss

        if self.base_loss == 'ce':
            self.base_loss = torch.nn.CrossEntropyLoss(reduction=reduction)
        elif self.base_loss == 'ls':
            self.base_loss = label_smoothing_criterion(distribution='uniform', reduction=reduction)
        elif self.base_loss == 'gls':
            self.base_loss = label_smoothing_criterion(distribution='gaussian', reduction=reduction)
        elif self.base_loss == 'focal_loss':
            kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": reduction}
            self.base_loss = FocalLoss(**kwargs)
        else:
            sys.exit('not a supported base_loss')

    def cost_sensitive_loss(self, input, target):
        if input.size(0) != target.size(0):
            raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                             .format(input.size(0), target.size(0)))
        device = input.device
        M = self.M.to(device)
        return (M[target, :] * input.float()).sum(axis=-1)

    def forward(self, logits, target):
        base_l = self.base_loss(logits, target)
        if self.lambd == 0:
            return self.base_loss(logits, target)
        else:
            preds = self.normalization(logits)
            loss = self.cost_sensitive_loss(preds, target)
            if self.reduction == 'none':
                return base_l + self.lambd*loss
            elif self.reduction == 'mean':
                return base_l + self.lambd*loss.mean()
            elif self.reduction == 'sum':
                return base_l + self.lambd*loss.sum()
            else:
                raise ValueError('`reduction` must be one of \'none\', \'mean\', or \'sum\'.')


import torch.nn.functional as F

class MixUpCELoss(torch.nn.Module):
    def __init__(self, n_classes, input_only=False, alpha=0.2, reduction='none'):
        super(MixUpCELoss, self).__init__()
        self.n_classes = n_classes
        self.input_only = input_only
        self.alpha = alpha
        self.reduction = reduction

    def partial_mixup(self, input, gamma, indices):
        if input.size(0) != indices.size(0):
            raise RuntimeError("Size mismatch!")
        perm_input = input[indices]
        return input.mul(gamma).add(perm_input, alpha=1 - gamma)

    def mixup(self, input, target, gamma):
        indices = torch.randperm(input.size(0), device=input.device, dtype=torch.long)
        return self.partial_mixup(input, gamma, indices), self.partial_mixup(target, gamma, indices)

    def forward(self, logits, labels):

        labels = F.one_hot(labels, self.n_classes)
        if self.input_only:
            logits = self.partial_mixup(logits, gamma=np.random.beta(self.alpha + 1, self.alpha),
                                  indices = torch.randperm(logits.size(0)))
        else:
            logits, labels= self.mixup(logits, labels, np.random.beta(self.alpha, self.alpha))

        loss = -(logits.log_softmax(dim=-1) * labels).sum(dim=-1)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()

class CDOLoss(torch.nn.Module):
    def __init__(self, base_loss='ce', cdo='l1', alpha=1, beta=1, n_classes=5, reduction='mean', normalization='sigmoid', do_not_add=True):
        super(CDOLoss, self).__init__()
        self.n_classes = n_classes
        self.cdo = cdo
        self.reduction = reduction
        self.normalization = normalization
        self.base_loss = base_loss
        self.alpha = alpha
        self.beta = beta
        self.do_not_add = do_not_add
        if base_loss == 'ce':
            self.base_loss = torch.nn.CrossEntropyLoss(reduction='none')
        elif base_loss == 'fl':
            focal_loss = torch.hub.load('adeelh/pytorch-multi-class-focal-loss', model='FocalLoss', alpha=None, gamma=2, reduction=reduction)
            self.base_loss = focal_loss
        elif base_loss == 'gls':
            self.base_loss = label_smoothing_criterion(distribution='gaussian', reduction='none')
        elif base_loss == 'cs_reg':
            self.base_loss = CostSensitiveRegularizedLoss(n_classes=self.n_classes, reduction='none')
        elif base_loss == 'ce_mixup':
            self.base_loss = MixUpCELoss(n_classes=self.n_classes, input_only=False, alpha=0.1, reduction='none')
        else: raise ValueError('`base_loss` must be \'ce\',\'fl\', \'gls\', or \'cs_reg\'.')

    def dice_loss(self, probs, cum_labels):
        # compute intersection
        intersection = torch.sum(probs * cum_labels, dim=1)
        # compute union
        union = torch.sum(probs * probs, dim=1) + torch.sum(cum_labels * cum_labels, dim=1)
        dice_score = 2. * intersection / union
        return 1 - dice_score

    def bce_loss(self, probs, cum_labels):
        return torch.nn.functional.binary_cross_entropy(probs, cum_labels, reduction='none').mean(dim=1)

    def forward(self, logits, labels):
        ohe_labels = torch.zeros(labels.size(0), self.n_classes).to(labels.device).scatter_(1, labels.view(-1, 1), 1)
        # dist = get_gaussian_label_distribution(n_classes=self.n_classes)
        # ohe_labels = torch.from_numpy(dist[labels.cpu().numpy()]).to(logits.device).float()

        cum_labels = torch.cumsum(ohe_labels, dim=1)
        # the model predicts PDFs
        base_loss = self.base_loss(logits, labels)

        # convert logits to probabilities before lifting to CDF
        if self.normalization == 'sigmoid':
            probs = logits.sigmoid() / (torch.sum(logits.sigmoid(), dim=1, keepdim=True))
        elif self.normalization == 'softmax':
            probs = logits.softmax(dim=1)

        # lift PDF to CDF
        cum_probs = torch.clamp(torch.cumsum(probs, dim=1), 0, 1)

        if self.cdo == 'dice':
            cdo_loss = self.dice_loss(cum_probs, cum_labels)# + self.dice_loss(torch.clamp(1-cum_probs, 0, 1), 1-cum_labels)
        elif self.cdo == 'bce':
            cdo_loss = self.bce_loss(cum_probs, cum_labels)
        elif self.cdo == 'l1':
            cdo_loss = torch.nn.functional.l1_loss(cum_probs, cum_labels, reduction='none').mean(dim=1)/self.n_classes
        elif self.cdo == 'l2':
            cdo_loss = torch.nn.functional.mse_loss(cum_probs, cum_labels, reduction='none').mean(dim=1) / self.n_classes
        elif self.cdo == 'huber':
            cdo_loss = torch.nn.functional.smooth_l1_loss(cum_probs, cum_labels, reduction='none').mean(dim=1) / self.n_classes
        else:
            raise ValueError('`cdo` must be \'dice\',\'bce\', \'l1\', or \'l2\'.')

        if self.reduction == 'none':
            if self.do_not_add:
                return base_loss, cdo_loss
            return self.alpha * base_loss + self.beta * cdo_loss
        elif self.reduction == 'mean':
            if self.do_not_add:
                return base_loss.mean(), cdo_loss.mean()
            return self.alpha * base_loss.mean() + self.beta * cdo_loss.mean()
        else:
            raise ValueError('`reduction` must be \'none\' or \'mean\'.')

class CeDiceLoss(torch.nn.Module):
    def __init__(self, base_loss='ce', alpha=1, n_classes=5, reduction='none',
                 f1_acc=False, normalization='sigmoid', do_not_add=True):
        super(CeDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.reduction = reduction
        self.f1_acc = f1_acc
        self.normalization = normalization
        self.base_loss = base_loss
        self.alpha = alpha
        self.do_not_add = do_not_add
        self.f1_acc = f1_acc

    def acc_dice_loss(self, probs, labels):
        # map everything to [-1,1]
        labels = 2 * labels.float() - 1
        probs = 2 * probs - 1

        # compute intersection
        intersection = torch.sum(probs * labels, dim=1)
        # compute union
        union = torch.sum(probs * probs, dim=1) + torch.sum(labels * labels, dim=1)
        dice_score = 2. * intersection / union
        # map back to [0,1]
        dice_score = 0.5 * (dice_score + 1)

        return 1 - dice_score

    def dice_loss(self, probs, ohe_labels):
        # compute intersection
        intersection = torch.sum(probs * ohe_labels, dim=1)
        # compute union
        union = torch.sum(probs * probs, dim=1) + torch.sum(ohe_labels * ohe_labels, dim=1)
        dice_score = 2. * intersection / union
        return 1 - dice_score

    def forward(self, logits, labels):
        # base loss
        if self.base_loss == 'ce':
            base_loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        elif self.base_loss == 'fl':
            focal_loss = torch.hub.load('adeelh/pytorch-multi-class-focal-loss', model='FocalLoss', alpha=None, gamma=2, reduction=self.reduction)
            self.base_loss = focal_loss
        elif self.base_loss == 'gls':
            self.base_loss = label_smoothing_criterion(distribution='gaussian', reduction=self.reduction)
        elif self.base_loss == 'cs_reg':
            self.base_loss = CostSensitiveRegularizedLoss(n_classes=self.n_classes, reduction=self.reduction)
        else:
            raise ValueError('`base_loss` must be \'ce\',\'fl\', \'gls\', or \'cs_reg\'.')

        # F1 loss:
        if self.normalization == 'sigmoid':
            probs = logits.sigmoid() / (torch.sum(logits.sigmoid(), dim=1, keepdim=True))
        elif self.normalization == 'softmax':
            probs = logits.softmax(dim=1)

        ohe_labels = torch.zeros(labels.size(0), self.n_classes).to(labels.device).scatter_(1, labels.view(-1, 1), 1)
        if self.f1_acc:
            d_loss = self.acc_dice_loss(probs, ohe_labels)
        else:
            d_loss = self.dice_loss(probs, ohe_labels)

        if self.reduction == 'none':
            if self.do_not_add:
                return base_loss, d_loss
            return self.alpha * base_loss + d_loss
        elif self.reduction == 'mean':
            if self.do_not_add:
                return base_loss.mean(), d_loss.mean()
            return base_loss.mean() + self.alpha * d_loss.mean()
        else:
            raise ValueError('`reduction` must be \'batch\' or \'mean\'.')

def get_focal_loss(alpha=None, gamma=2.0, reduction='mean'):
    focal_loss = torch.hub.load('adeelh/pytorch-multi-class-focal-loss', model='FocalLoss', alpha=alpha, gamma=gamma, reduction=reduction)
    return focal_loss, focal_loss

def get_ce_dice_criterion(n_classes=5, reduction='mean', base_loss='ce', f1_acc=False, normalization='sigmoid', alpha=1):
    train_criterion = CeDiceLoss(base_loss, alpha=alpha, n_classes=n_classes, reduction=reduction, f1_acc=f1_acc,
                                 normalization=normalization,  do_not_add=True)
    val_criterion = CeDiceLoss(base_loss,  alpha=alpha, n_classes=n_classes, reduction=reduction, f1_acc=f1_acc,
                               normalization=normalization, do_not_add=True)
    return train_criterion, val_criterion

def get_cdo_criterion(n_classes=5, base_loss='ce', cdo='l2', alpha=1, beta=1, normalization='sigmoid', reduction='mean', do_not_add=True):
    train_criterion = CDOLoss(base_loss, cdo=cdo, alpha=alpha, beta=beta, n_classes=n_classes, reduction=reduction, normalization=normalization,  do_not_add=do_not_add)
    val_criterion = CDOLoss(base_loss,  cdo=cdo, alpha=alpha, beta=beta, n_classes=n_classes, reduction=reduction, normalization=normalization, do_not_add=do_not_add)
    return train_criterion, val_criterion

def get_cost_sensitive_regularized_criterion(base_loss='ce', n_classes=6, lambd=10, exp=2, reduction='mean'):
    train_criterion = CostSensitiveRegularizedLoss(n_classes, exp=exp, normalization='softmax', base_loss=base_loss, lambd=lambd, reduction=reduction)
    val_criterion = CostSensitiveRegularizedLoss(n_classes, exp=exp, normalization='softmax', base_loss=base_loss, lambd=lambd, reduction=reduction)
    return train_criterion, val_criterion

class GranularGLS(torch.nn.Module):
    def __init__(self, n_classes=5, amplifier=50, noise=0.25, cdo='dice', normalization='sigmoid', alpha=1, beta=1, reduction='mean'):
        super(GranularGLS, self).__init__()
        self.n_classes = n_classes
        self.amplifier=amplifier
        self.noise = noise
        self.reduction=reduction
        self.cdo = cdo
        self.normalization = normalization
        self.alpha = alpha
        self.beta=beta

    def get_super_noisy_gauss_label(self, label):
        n = self.n_classes*self.amplifier
        half_int = self.amplifier/2
        noisy_int = np.random.uniform(low=half_int/4, high=3*half_int/4)
        label_noise = np.random.uniform(low=-self.noise, high=self.noise)
        label += label_noise
        label_new = half_int + label*self.amplifier
        gauss_label = stats2.norm.pdf(np.arange(n), label_new, noisy_int)
        gauss_label/=np.sum(gauss_label)
        return gauss_label

    def get_all_super_noisy_gauss_labels(self):
        gauss_labels = []
        for label in range(self.n_classes):
            gauss_labels.append(self.get_super_noisy_gauss_label(label))
        distribs = np.stack(gauss_labels, axis=0)
        return distribs

    def dice_loss(self, probs, cum_labels):
        # compute intersection
        intersection = torch.sum(probs * cum_labels, dim=1)
        # compute union
        union = torch.sum(probs * probs, dim=1) + torch.sum(cum_labels * cum_labels, dim=1)
        dice_score = 2. * intersection / union
        return 1 - dice_score

    def forward(self, logits, labels):
        gauss_labels = self.get_all_super_noisy_gauss_labels()
        ohe_labels = torch.from_numpy(gauss_labels[labels.cpu().numpy()]).float().to(logits.device)
        logp = F.log_softmax(logits, dim=1)
        if self.reduction == 'none':
            loss =  torch.sum(-logp * ohe_labels, dim=1)
        elif self.reduction == 'mean':
            loss =  torch.sum(-logp * ohe_labels, dim=1).mean()

        cum_labels = torch.cumsum(ohe_labels, dim=1)
        # convert logits to probabilities before lifting to CDF
        if self.normalization == 'sigmoid':
            probs = logits.sigmoid() / (torch.sum(logits.sigmoid(), dim=1, keepdim=True))
        elif self.normalization == 'softmax':
            probs = logits.softmax(dim=1)
        # lift PDF to CDF
        cum_probs = torch.clamp(torch.cumsum(probs, dim=1), 0, 1)
        if self.cdo == 'dice':
            cdo_loss = self.dice_loss(cum_probs, cum_labels)
        elif self.cdo == 'bce':
            cdo_loss = self.bce_loss(cum_probs, cum_labels)
        elif self.cdo == 'l1':
            cdo_loss = torch.nn.functional.l1_loss(cum_probs, cum_labels, reduction='none').mean(dim=1)/self.n_classes
        elif self.cdo == 'l2':
            cdo_loss = torch.nn.functional.mse_loss(cum_probs, cum_labels, reduction='none').mean(dim=1) / self.n_classes
        else:
            raise ValueError('`cdo` must be \'dice\',\'bce\', \'l1\', or \'l2\'.')

        return self.alpha * loss.mean() + self.beta * cdo_loss.mean()

def get_granular_label_smoothing_criterion(n_classes, amplifier=50, noise=0.25, alpha=1, beta=0, reduction='mean'):
    train_criterion = GranularGLS(n_classes, amplifier, noise, alpha=alpha, beta=beta, reduction=reduction)
    val_criterion = GranularGLS(n_classes, amplifier, noise=0, alpha=alpha, beta=beta, reduction=reduction)

    return train_criterion, val_criterion