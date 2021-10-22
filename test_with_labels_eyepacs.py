import argparse
import pandas as pd
from models.get_model import get_arch
from utils.get_loaders import get_test_cls_loader
from sklearn.metrics import roc_auc_score as roc
from scipy.stats import kendalltau
from utils.evaluation import evaluate_multi_cls
from utils.reproducibility import set_seeds
from utils.model_saving_loading import load_model
from tqdm import trange
import numpy as np
import torch
import torchvision
import torch.nn.functional as F

import os.path as osp
import os
import sys
from utils.gmean import geometric_mean_score

from sklearn.metrics import confusion_matrix


def get_aca(y_true, y_pred):
    cm = confusion_matrix(y_true,
                          y_pred,
                          labels=[0, 1, 2, 3, 4], normalize='true')
    return np.mean(np.diag(np.divide(cm, np.sum(cm, axis=1)[:, None])))


def str2bool(v):
    # as seen here: https://stackoverflow.com/a/43357954/3208255
    if isinstance(v, bool):
       return v
    if v.lower() in ('true','yes'):
        return True
    elif v.lower() in ('false','no'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/eyepacs_all_ims/', help='path data')
parser.add_argument('--csv_val', type=str,  default='data/val_eyepacs.csv', help='path to val data csv')
parser.add_argument('--csv_test', type=str, default='data/test_eyepacs.csv', help='path to test data csv')
parser.add_argument('--model_name', type=str, default='bit_resnext50_1', help='selected architecture')
parser.add_argument('--load_path', type=str, default='experiments/resnext50_MS/', help='path to saved model')
parser.add_argument('--dihedral_tta', type=int, default=1, help='dihedral group cardinality (0)')
parser.add_argument('--im_size', help='delimited list input, could be 500, or 600,400', type=str, default='512,512')
parser.add_argument('--n_classes', type=int, default=5, help='number of target classes (6)')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--results_path', type=str, default='results/', help='path to output csv')
parser.add_argument('--csv_out_val', type=str, default='results_val.csv', help='path to output csv')
parser.add_argument('--csv_out_test', type=str, default='results_test.csv', help='path to output csv')

args = parser.parse_args()


def run_one_epoch_cls(loader, model, optimizer=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = optimizer is not None
    model.train() if train else model.eval()
    probs_all, preds_all, labels_all = [], [], []
    with trange(len(loader)) as t:
        for i_batch, (inputs, labels, _) in enumerate(loader):
            if loader.dataset.has_labels:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            else:
                inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            probs = torch.nn.Softmax(dim=1)(logits)
            _, preds = torch.max(probs, 1)
            probs_all.extend(probs.detach().cpu().numpy())
            preds_all.extend(preds.detach().cpu().numpy())
            if loader.dataset.has_labels:
                labels_all.extend(labels.detach().cpu().numpy())
            run_loss = 0
            t.set_postfix(vl_loss="{:.4f}".format(float(run_loss)))
            t.update()
    if loader.dataset.has_labels:
        return np.stack(preds_all), np.stack(probs_all), np.stack(labels_all)
    return np.stack(preds_all), np.stack(probs_all), None

def test_cls_tta_dihedral(model, test_loader, n=3):
    probs_tta = []
    prs = [0, 1]

    test_loader.dataset.transforms.transforms.insert(-1, torchvision.transforms.RandomRotation(0))
    rotations = np.array([i * 360 // n for i in range(n)])
    for angle in rotations:
        for p2 in prs:
            test_loader.dataset.transforms.transforms[2].p = p2  # pr(vertical flip)
            test_loader.dataset.transforms.transforms[-2].degrees = [angle, angle]
            # validate one epoch, note no optimizer is passed
            with torch.no_grad():
                test_preds, test_probs, test_labels = run_one_epoch_cls(test_loader, model)
                probs_tta.append(test_probs)

    probs_tta = np.mean(np.array(probs_tta), axis=0)
    preds_tta = np.argmax(probs_tta, axis=1)

    del model
    torch.cuda.empty_cache()
    return probs_tta, preds_tta, test_labels

def test_cls(model, test_loader):
    # validate one epoch, note no optimizer is passed
    with torch.no_grad():
        test_preds, test_probs, test_labels = run_one_epoch_cls(test_loader, model)

    del model
    torch.cuda.empty_cache()
    return test_probs, test_preds, test_labels


if __name__ == '__main__':
    '''
    Example:
    python test_with_labels.py --csv_val data/eyepacs/val_same_fov.csv --csv_test data/eyepacs/test_eyepacs.csv --batch_size 8
    --load_path experiments/eyepacs/GLS_alone/cycle_05_F1_82.81_K_80.31_MCC_58.93 --csv_out_val borrar_val.csv --csv_out_test borrar_test.csv
    '''
    data_path = 'data'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # reproducibility
    seed_value = 0
    set_seeds(seed_value, use_cuda)

    # gather parser parameters
    args = parser.parse_args()
    data_path = args.data_path
    model_name = args.model_name
    load_path = args.load_path
    results_path = osp.join(args.results_path, load_path.split('/')[1], load_path.split('/')[2])
    os.makedirs(results_path, exist_ok=True)
    bs = args.batch_size
    csv_test = args.csv_test
    csv_val = args.csv_val
    n_classes = args.n_classes
    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')
    dihedral_tta = args.dihedral_tta
    csv_out_val = args.csv_out_val
    csv_out_test = args.csv_out_test

    print('* Loading model {} from {}'.format(model_name, load_path))
    model, mean, std = get_arch(model_name, n_classes=n_classes)
    model, stats = load_model(model, load_path, device='cpu')
    model = model.to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    class_names = ['DR0', 'DR1', 'DR2', 'DR3', 'DR4']


    print('* Creating Test Dataloaders, batch size = {:d}'.format(bs))
    test_loader = get_test_cls_loader(csv_path_test=csv_test, data_path=data_path, batch_size=bs, mean=mean, std=std, tg_size=tg_size, test=False)

    if dihedral_tta==0:
        probs, preds, labels = test_cls(model, test_loader)
    elif dihedral_tta>0:
        probs, preds, labels = test_cls_tta_dihedral(model, test_loader, n=dihedral_tta)
    else: sys.exit('dihedral_tta must be >=0')

    print_conf = True
    text_file = osp.join(results_path, 'performance_test.txt')
    test_auc, test_k, test_mcc, test_f1, test_bacc, test_auc_all ,test_f1_all = evaluate_multi_cls(labels, preds, probs, print_conf=True,
                                                          class_names=class_names, text_file=text_file)

    print('Test - K: {:.2f} - mAUC: {:.2f}  - MCC: {:.2f} - F1: {:.2f} - BalAcc: {:.2f}'.format(100*test_k, 100*test_auc, 100*test_mcc, 100*test_f1, 100*test_bacc))



