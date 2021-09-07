import os.path as osp
import copy
import sys
import numbers
import pandas as pd
from PIL import Image
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as tr
from .combo_loader import ComboLoader

class BinClassDataset(Dataset):
    def __init__(self, csv_path, neg_classes=(0), pos_classes=(1,2), data_path=None, transforms=None, mean=None, std=None, test=False):
        self.csv_path=csv_path
        df = pd.read_csv(self.csv_path)
        self.data_path = data_path
        classes = neg_classes + pos_classes
        col_names = df.columns
        filtered_df = df[df[col_names[-1]].isin(classes)]
        pd.set_option('mode.chained_assignment', None)
        filtered_df[col_names[-1]] = filtered_df[col_names[-1]].replace(neg_classes, 0)
        filtered_df[col_names[-1]] = filtered_df[col_names[-1]].replace(pos_classes, 1)
        self.filtered_df = filtered_df
        self.col_names = col_names
        self.im_list = list(filtered_df.image_id)
        self.has_labels = not test
        if self.has_labels:
            self.dr = filtered_df[col_names[-1]].values
        self.transforms = transforms
        self.normalize = tr.Normalize(mean, std)

    def __getitem__(self, index):
        # load image and labels
        if self.data_path is not None:
            try:
                img = Image.open(osp.join(self.data_path, self.im_list[index]))
            except:
                print(osp.join(self.data_path, self.im_list[index]))
        else:
            img = Image.open(self.im_list[index])

        if self.has_labels:
            dr = self.dr[index]

        if self.transforms is not None:
            img = self.transforms(img)
            img = self.normalize(img)
        if self.has_labels:
            return img, dr, self.im_list[index]
        return img

    def __len__(self):
        return len(self.im_list)

class ClassDataset(Dataset):
    def __init__(self, csv_path, data_path=None, transforms=None, mean=None, std=None, test=False):
        self.csv_path=csv_path
        df = pd.read_csv(self.csv_path)
        self.data_path = data_path
        self.im_list = df.image_id
        self.has_labels = not test
        if self.has_labels:
            self.dr = df[df.columns[-1]].values
        self.transforms = transforms
        self.normalize = tr.Normalize(mean, std)

    def __getitem__(self, index):
        # load image and labels
        if self.data_path is not None:
            img = Image.open(osp.join(self.data_path, self.im_list[index]))
        else:
            img = Image.open(self.im_list[index])

        if self.has_labels:
            dr = self.dr[index]

        if self.transforms is not None:
            img = self.transforms(img)
            img = self.normalize(img)
        if self.has_labels:
            return img, dr, self.im_list[index]
        return img

    def __len__(self):
        return len(self.im_list)

# https://github.com/huanghoujing/pytorch-wrapping-multi-dataloaders/blob/master/wrapping_multi_dataloaders.py
class ComboIter(object):
    """An iterator."""
    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]

    def __iter__(self):
        return self

    def __next__(self):
        # When the shortest loader (the one with minimum number of batches)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        batches = [loader_iter.next() for loader_iter in self.loader_iters]
        return self.my_loader.combine_batch(batches)

    def __len__(self):
        return len(self.my_loader)

class ComboLoader(object):
    """This class wraps several pytorch DataLoader objects, allowing each time
    taking a batch from each of them and then combining these several batches
    into one. This class mimics the `for batch in loader:` interface of
    pytorch `DataLoader`.
    Args:
    loaders: a list or tuple of pytorch DataLoader objects
    """
    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        return ComboIter(self)

    def __len__(self):
        return min([len(loader) for loader in self.loaders])

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        return batches

def get_sampling_probabilities(class_count, mode='instance', ep=None, n_eps=None):
    '''
    Note that for progressive sampling I use n_eps-1, which I find more intuitive.
    If you are training for 10 epochs, you pass n_eps=10 to this function. Then, inside
    the training loop you would have sth like 'for ep in range(n_eps)', so ep=0,...,9,
    and all fits together.
    '''
    if mode == 'instance':
        q = 0
    elif mode == 'class':
        q = 1
    elif mode == 'sqrt':
        q = 0.5 # 1/2
    elif mode == 'cbrt':
        q = 0.125 # 1/8
    elif mode == 'prog':
        assert ep != None and n_eps != None, 'progressive sampling requires to pass values for ep and n_eps'
        relative_freq_imbal = class_count ** 0 / (class_count ** 0).sum()
        relative_freq_bal = class_count ** 1 / (class_count ** 1).sum()
        sampling_probabilities_imbal = relative_freq_imbal ** (-1)
        sampling_probabilities_bal = relative_freq_bal ** (-1)
        return (1 - ep / (n_eps - 1)) * sampling_probabilities_imbal + (ep / (n_eps - 1)) * sampling_probabilities_bal
    else: sys.exit('not a valid mode')

    relative_freq = class_count ** q / (class_count ** q).sum()
    sampling_probabilities = relative_freq ** (-1)

    return sampling_probabilities

def modify_loader(loader, mode, ep=None, n_eps=None):
    class_count = np.unique(loader.dataset.dr, return_counts=True)[1]
    sampling_probs = get_sampling_probabilities(class_count, mode=mode, ep=ep, n_eps=n_eps)
    sample_weights = sampling_probs[loader.dataset.dr]

    mod_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))
    mod_loader = DataLoader(loader.dataset, batch_size = loader.batch_size, sampler=mod_sampler, num_workers=loader.num_workers)
    return mod_loader

def get_combo_loader(loader, base_sampling='instance'):
    if base_sampling == 'instance':
        imbalanced_loader = loader
    else:
        imbalanced_loader = modify_loader(loader, mode=base_sampling)

    balanced_loader = modify_loader(loader, mode='class')
    combo_loader = ComboLoader([imbalanced_loader, balanced_loader])
    return combo_loader


def get_train_val_cls_datasets(csv_path_train, csv_path_val, data_path=None, mean=None, std=None, tg_size=(512, 512), see_classes=True):

    train_dataset = ClassDataset(csv_path=csv_path_train, data_path=data_path, mean=mean, std=std)
    val_dataset = ClassDataset(csv_path=csv_path_val, data_path=data_path, mean=mean, std=std)

    # transforms definition
    # required transforms
    resize = tr.Resize(tg_size)
    tensorizer = tr.ToTensor()
    # geometric transforms
    h_flip = tr.RandomHorizontalFlip()
    v_flip = tr.RandomVerticalFlip()
    rotate = tr.RandomRotation(degrees=45)
    scale = tr.RandomAffine(degrees=0, scale=(0.95, 1.20))
    transl = tr.RandomAffine(degrees=0, translate=(0.05, 0))
    # either translate, rotate, or scale
    scale_transl_rot = tr.RandomChoice([scale, transl, rotate])
    # intensity transforms
    brightness, contrast, saturation, hue = 0.25, 0.25, 0.25, 0.01
    jitter = tr.ColorJitter(brightness, contrast, saturation, hue)
    train_transforms = tr.Compose([resize, scale_transl_rot, jitter, h_flip, v_flip, tensorizer])
    val_transforms = tr.Compose([resize, tensorizer])
    train_dataset.transforms = train_transforms
    val_dataset.transforms = val_transforms
    if see_classes:
        print(20 * '*')
        for c in range(len(np.unique(train_dataset.dr))):
            exs_train = np.count_nonzero(train_dataset.dr== c)
            exs_val = np.count_nonzero(val_dataset.dr == c)
            print('Found {:d}/{:d} train/val examples of class {}'.format(exs_train, exs_val, c))

    return train_dataset, val_dataset

def get_train_val_cls_loaders(csv_path_train, csv_path_val, data_path=None, batch_size=4, tg_size=(512, 512), mean=None, std=None,
                              num_workers=0, see_classes=True):

    train_dataset, val_dataset = get_train_val_cls_datasets(csv_path_train, csv_path_val, data_path=data_path, tg_size=tg_size,
                                                            mean=mean, std=std, see_classes=see_classes)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(), shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True)
    return train_loader, val_loader

def get_train_val_bin_datasets(csv_path_train, csv_path_val, neg_classes=(0), pos_classes=(1,2), data_path=None,
                               mean=None, std=None, tg_size=(512, 512), see_classes=True):

    train_dataset = BinClassDataset(csv_path=csv_path_train, neg_classes=neg_classes, pos_classes=pos_classes, data_path=data_path, mean=mean, std=std)
    val_dataset = BinClassDataset(csv_path=csv_path_val, neg_classes=neg_classes, pos_classes=pos_classes, data_path=data_path, mean=mean, std=std)

    # transforms definition
    # required transforms
    resize = tr.Resize(tg_size)
    tensorizer = tr.ToTensor()
    # geometric transforms
    h_flip = tr.RandomHorizontalFlip()
    v_flip = tr.RandomVerticalFlip()
    rotate = tr.RandomRotation(degrees=45)
    scale = tr.RandomAffine(degrees=0, scale=(0.95, 1.20))
    transl = tr.RandomAffine(degrees=0, translate=(0.05, 0))
    # either translate, rotate, or scale
    scale_transl_rot = tr.RandomChoice([scale, transl, rotate])
    # intensity transforms
    brightness, contrast, saturation, hue = 0.25, 0.25, 0.25, 0.01
    jitter = tr.ColorJitter(brightness, contrast, saturation, hue)
    train_transforms = tr.Compose([resize, scale_transl_rot, jitter, h_flip, v_flip, tensorizer])
    val_transforms = tr.Compose([resize, tensorizer])
    train_dataset.transforms = train_transforms
    val_dataset.transforms = val_transforms
    if see_classes:
        print(20 * '*')
        for c in range(len(np.unique(train_dataset.dr))):
            exs_train = np.count_nonzero(train_dataset.dr== c)
            exs_val = np.count_nonzero(val_dataset.dr == c)
            print('Found {:d}/{:d} train/val examples of class {}'.format(exs_train, exs_val, c))

    return train_dataset, val_dataset

def get_train_val_bin_cls_loaders(csv_path_train, csv_path_val, neg_classes=(0), pos_classes=(1,2), data_path=None, batch_size=4, tg_size=(512, 512), mean=None, std=None,
                              num_workers=0, see_classes=True):

    train_dataset, val_dataset = get_train_val_bin_datasets(csv_path_train, csv_path_val, neg_classes=neg_classes, pos_classes=pos_classes,
                                                            data_path=data_path, tg_size=tg_size, mean=mean, std=std, see_classes=see_classes)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(), shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True)
    return train_loader, val_loader


def get_test_cls_dataset(csv_path_test, data_path=None, mean=None, std=None, tg_size=(512,512), test=False):

    test_dataset = ClassDataset(csv_path_test, data_path=data_path, mean=mean, std=std, test=test)

    size = tg_size
    # required transforms
    resize = tr.Resize(size)
    h_flip = tr.RandomHorizontalFlip(p=0)
    v_flip = tr.RandomVerticalFlip(p=0)
    tensorizer = tr.ToTensor()
    test_transforms = tr.Compose([resize, h_flip, v_flip, tensorizer])
    test_dataset.transforms = test_transforms

    return test_dataset

def get_test_cls_loader(csv_path_test, data_path=None, batch_size=8, tg_size=(512, 512), mean=None, std=None, num_workers=8, test=False):
    test_dataset = get_test_cls_dataset(csv_path_test, data_path=data_path, tg_size=tg_size, mean=mean, std=std, test=test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return test_loader

def modify_dataset_bin(train_loader, csv_train_path, im_interest=None, keep_samples=2000, see_classes=True):
    if list(set(keep_samples))==[1]: return train_loader
    train_loader_new = copy.deepcopy(train_loader)  # note, otherwise we modify underlying dataset
    train_df = copy.deepcopy(train_loader.dataset.filtered_df)
    col_names = train_df.columns
    classes = np.unique(train_df[col_names[-1]])
    sample_spec=False
    if isinstance(keep_samples, numbers.Number):
        keep_samples = len(classes)*[keep_samples] # e.g., get 2,000 samples for each class
        sample_spec = True
    ims_per_class = []
    new_ims_per_class = []

    for c in classes:
        ims_per_class.append(train_df.loc[train_df[col_names[-1]] == c])
        n_ims = ims_per_class[c].shape[0]
        if sample_spec:
            oversample = n_ims < keep_samples[c]
            n_samples = keep_samples[c]
        else:
            oversample = n_ims < keep_samples[c] * n_ims
            n_samples = int(keep_samples[c] * n_ims)

        if oversample:  # when we oversample, we use bootstrapping
            new_ims_per_class.append(ims_per_class[c].sample(n=n_samples, replace=oversample))
            duplicate = new_ims_per_class[c][new_ims_per_class[c].duplicated()].shape[0]
            if see_classes:
                # print('Class {}: nr samples (%duplicated): {:d} ({:.2%})'.format(c, new_ims_per_class[c].shape[0], duplicate/new_ims_per_class[c].shape[0]))
                print('Class {}: nr samples (%duplicated): {:d} ({:d})'.format(c, new_ims_per_class[c].shape[0], duplicate))
        elif keep_samples[c]==1: # neither oversample nor undersample
            new_ims_per_class.append(ims_per_class[c])
            duplicate = 0
            if see_classes:
                print('Class {}: nr samples (%duplicated): {:d} ({:.0%})'.format(c, new_ims_per_class[c].shape[0], duplicate/new_ims_per_class[c].shape[0]))
        else:  # when we undersample is when we want to choose the best examples
            if im_interest is None: # if no interest is provided we fall back to random sampling, but no replacement
                new_ims_per_class.append(ims_per_class[c].sample(n=n_samples, replace=False))
            else: # if interest is provided, we discard less interesting images
                interesting_examples = pd.merge(ims_per_class[c], im_interest, on='image_id').sort_values(by='interest', ascending=False, inplace=False)
                if not discard_top_losers:
                    interesting_examples = interesting_examples.head(n=n_samples)  # keep exs with greater losses
                else:
                    to_be_kept = n_samples
                    added_slack = int(1.05 * n_samples)
                    interesting_examples = interesting_examples.head(n=added_slack)  # retain 105% interesting images
                    interesting_examples = interesting_examples.tail(n=to_be_kept)   # discard top 5%
                new_ims_per_class.append(interesting_examples)
            duplicate = new_ims_per_class[c][new_ims_per_class[c].duplicated()].shape[0]
            if see_classes:
                # print('Class {}: nr samples (%duplicated): {:d} ({:.2%})'.format(c, new_ims_per_class[c].shape[0], duplicate/new_ims_per_class[c].shape[0]))
                print('Class {}: nr samples (%duplicated): {:d} ({:d})'.format(c, new_ims_per_class[c].shape[0], duplicate))
    train_df_under_oversampled = pd.concat(new_ims_per_class)

    train_loader_new.dataset.im_list = train_df_under_oversampled['image_id'].values
    train_loader_new.dataset.dr = train_df_under_oversampled[col_names[-1]].values

    return train_loader_new

def modify_dataset(train_loader, csv_train_path, im_interest=None, keep_samples=2000, see_classes=True, discard_top_losers=True):
    if list(set(keep_samples))==[1]: return train_loader
    train_loader_new = copy.deepcopy(train_loader)  # note, otherwise we modify underlying dataset
    train_df = pd.read_csv(csv_train_path)

    col_names = train_df.columns
    classes = np.unique(train_df[col_names[-1]])
    sample_spec=False
    if isinstance(keep_samples, numbers.Number):
        keep_samples = len(classes)*[keep_samples] # e.g., get 2,000 samples for each class
        sample_spec = True
    ims_per_class = []
    new_ims_per_class = []

    for c in classes:
        ims_per_class.append(train_df.loc[train_df[col_names[-1]] == c])
        n_ims = ims_per_class[c].shape[0]
        if sample_spec:
            oversample = n_ims < keep_samples[c]
            n_samples = keep_samples[c]
        else:
            oversample = n_ims < keep_samples[c] * n_ims
            n_samples = int(keep_samples[c] * n_ims)

        if oversample:  # when we oversample, we use bootstrapping
            new_ims_per_class.append(ims_per_class[c].sample(n=n_samples, replace=oversample))
            duplicate = new_ims_per_class[c][new_ims_per_class[c].duplicated()].shape[0]
            if see_classes:
                # print('Class {}: nr samples (%duplicated): {:d} ({:.2%})'.format(c, new_ims_per_class[c].shape[0], duplicate/new_ims_per_class[c].shape[0]))
                print('Class {}: nr samples (%duplicated): {:d} ({:d})'.format(c, new_ims_per_class[c].shape[0], duplicate))
        elif keep_samples[c]==1: # neither oversample nor undersample
            new_ims_per_class.append(ims_per_class[c])
            duplicate = 0
            if see_classes:
                print('Class {}: nr samples (%duplicated): {:d} ({:.0%})'.format(c, new_ims_per_class[c].shape[0], duplicate/new_ims_per_class[c].shape[0]))
        else:  # when we undersample is when we want to choose the best examples
            if im_interest is None: # if no interest is provided we fall back to random sampling, but no replacement
                new_ims_per_class.append(ims_per_class[c].sample(n=n_samples, replace=False))
            else: # if interest is provided, we discard less interesting images
                interesting_examples = pd.merge(ims_per_class[c], im_interest, on='image_id').sort_values(by='interest', ascending=False, inplace=False)
                if not discard_top_losers:
                    interesting_examples = interesting_examples.head(n=n_samples)  # keep exs with greater losses
                else:
                    to_be_kept = n_samples
                    added_slack = int(1.05 * n_samples)
                    interesting_examples = interesting_examples.head(n=added_slack)  # retain 105% interesting images
                    interesting_examples = interesting_examples.tail(n=to_be_kept)   # discard top 5%
                new_ims_per_class.append(interesting_examples)
            duplicate = new_ims_per_class[c][new_ims_per_class[c].duplicated()].shape[0]
            if see_classes:
                # print('Class {}: nr samples (%duplicated): {:d} ({:.2%})'.format(c, new_ims_per_class[c].shape[0], duplicate/new_ims_per_class[c].shape[0]))
                print('Class {}: nr samples (%duplicated): {:d} ({:d})'.format(c, new_ims_per_class[c].shape[0], duplicate))
    train_df_under_oversampled = pd.concat(new_ims_per_class)

    train_loader_new.dataset.im_list = train_df_under_oversampled['image_id'].values
    train_loader_new.dataset.dr = train_df_under_oversampled['dr'].values

    return train_loader_new