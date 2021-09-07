from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.metrics import balanced_accuracy_score as bal_acc
import numpy as np
from scipy.stats import rankdata
import sys

def iou_score(target, prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / (np.sum(union) + 1e-6)
    return iou_score

def dice_score(actual, predicted):
    actual = np.asarray(actual).astype(np.bool)
    predicted = np.asarray(predicted).astype(np.bool)
    im_sum = actual.sum() + predicted.sum()
    if im_sum == 0: return 1
    intersection = np.logical_and(actual, predicted)
    return 2. * intersection.sum() / im_sum


def fast_auc(actual, predicted):
    r = rankdata(predicted)
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(r[actual==1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None, text_file=None):
    """
    pretty print for confusion matrixes
    https://gist.github.com/zachguo/10296432
    """
    if text_file is None: print("\n", end=" ")
    else: print("\n", end=" ", file=open(text_file, "a"))

    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    if text_file is None: print("    " + fst_empty_cell, end=" ")
    else: print("    " + fst_empty_cell, end=" ", file = open(text_file, "a"))

    for label in labels:
        if text_file is None: print("%{0}s".format(columnwidth) % label, end=" ")
        else: print("%{0}s".format(columnwidth) % label, end=" ", file = open(text_file, "a"))
    if text_file is None: print()
    else: print(' ', file = open(text_file, "a"))
    # Print rows
    for i, label1 in enumerate(labels):
        if text_file is None: print("    %{0}s".format(columnwidth) % label1, end=" ")
        else: print("    %{0}s".format(columnwidth) % label1, end=" ", file = open(text_file, "a"))
        for j in range(len(labels)):
            cell = "%{}d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            if text_file is None: print(cell, end=" ")
            else: print(cell, end=" ", file = open(text_file, "a"))
        if text_file is None: print()
        else: print(' ', file = open(text_file, "a"))

def get_one_hot_np(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def evaluate_bin_cls(y_true, y_pred, y_proba, print_conf=True, text_file=None, class_names=None, loss=0):
    classes, _ = np.unique(y_true, return_counts=True)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)

    acc = accuracy_score(y_true, y_pred)

    if text_file is not None:
        print("AUC={:.2f} - ACC={:.2f} - F1={:.2f} - Loss={:.4f}\n".format(100*auc, 100*acc, 100*f1, loss), end=" ", file=open(text_file, "a"))
        if print_conf:
            cm = confusion_matrix(y_true, y_pred, labels=classes)
            print_cm(cm, class_names, text_file=text_file)

    return auc, acc, f1

def evaluate_multi_cls(y_true, y_pred, y_proba, print_conf=True, text_file=None, class_names=None, loss=0, lr=None):
    # preds
    f1 = f1_score(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    k = kappa(y_true, y_pred, weights='quadratic')
    bacc = bal_acc(y_true, y_pred)
    # probs - handle case of not every label in y_true
    present_classes, _ = np.unique(y_true, return_counts=True)
    present_classes = list(present_classes)
    classes = list(range(y_proba.shape[1]))
    if present_classes != classes:
        y_proba = y_proba[:, present_classes]
        y_proba /=  y_proba.sum(axis=1)[:, np.newaxis]
        f1_all = len(present_classes) * [0]
        auc_all = len(present_classes) * [0]
    else:
        y_true_ohe = get_one_hot_np(y_true, len(present_classes)).astype(int)
        y_pred_ohe = get_one_hot_np(y_pred, len(present_classes)).astype(int)
        f1_all = [f1_score(y_true_ohe[:, i], y_pred_ohe[:, i]) for i in range(len(present_classes))]
        auc_all = [roc_auc_score(y_true_ohe[:, i], y_proba[:, i]) for i in range(len(present_classes))]
    if len(classes)==2:
        mean_auc = roc_auc_score(y_true, y_proba[:,1])
    else:
        mean_auc = roc_auc_score(y_true, y_proba, multi_class='ovr') # equivalent to np.mean(auc_all)

        # mean_auc = roc_auc_score(y_true, y_proba, average='weighted', multi_class='ovr')
        # ovo should be better, but average is not clear from docs
        # mean_auc = roc_auc_score(y_true, y_proba, average='macro', multi_class='ovo')

    if class_names is None:
        class_names = [str(n) for n in present_classes]

    if print_conf:
        if text_file is not None:
            if lr is None:
                print("K={:.2f} - BACC={:.2f} - MCC={:.2f}- F1={:.2f} - AUC={:.2f} - Loss={:.4f}".format(100*k, 100*bacc, 100*mcc, 100*f1, 100*mean_auc, loss),
                      end=" ", file=open(text_file, "a"))
            else:
                print("K={:.2f} - BACC={:.2f} - MCC={:.2f} - F1={:.2f} - AUC={:.2f} - Loss={:.4f} - LR={:.5f}".format(100*k, 100*bacc, 100*mcc, 100*f1, 100*mean_auc, loss, lr),
                    end=" ", file=open(text_file, "a"))

            if len(class_names)==3:
                print('\nAUC: No={:.2f} - cDME={:.2f} - DME={:.2f}'.format(
                    100 * auc_all[0], 100 * auc_all[1], 100 * auc_all[2]), end=" ", file=open(text_file, "a"))
            elif len(class_names)==5:
                print('\nAUC: DR0={:.2f} - DR1={:.2f} - DR2={:.2f} - DR3={:.2f} - DR4={:.2f}'.format(
                    100 * auc_all[0], 100 * auc_all[1], 100 * auc_all[2], 100 * auc_all[3], 100 * auc_all[4]), end=" ", file=open(text_file, "a"))
            elif len(class_names)==6:
                print('\nF1: DR0={:.2f} - DR1={:.2f} - DR2={:.2f} - DR3={:.2f} - DR4={:.2f} - U={:.2f}'.format(
                    100 * f1_all[0], 100 * f1_all[1], 100 * f1_all[2], 100 * f1_all[3], 100 * f1_all[4], 100 * f1_all[5]), end=" ", file=open(text_file, "a"))
                print('\nAUC: DR0={:.2f} - DR1={:.2f} - DR2={:.2f} - DR3={:.2f} - DR4={:.2f} - U={:.2f}'.format(
                    100 * auc_all[0], 100 * auc_all[1], 100 * auc_all[2], 100 * auc_all[3], 100 * auc_all[4], 100 * auc_all[5]), end=" ", file=open(text_file, "a"))
            else:
                return mean_auc, k, mcc, f1, bacc, auc_all, f1_all
                #sys.exit('invalid number of clases in print conf')
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        print_cm(cm, class_names, text_file=text_file)
    return mean_auc, k, mcc, f1, bacc, auc_all, f1_all


# def evaluate_multi_cls_Q(y_true, y_pred, y_proba):
#     f1 = f1_score(y_true, y_pred, average='micro')
#     mcc = matthews_corrcoef(y_true, y_pred)
#     mean_auc = roc_auc_score(y_true, y_proba, average='weighted', multi_class='ovo')
#     acc = accuracy_score(y_true, y_pred)
#     k = kappa(y_true, y_pred, weights='quadratic')
#
#     return mean_auc, k, mcc, f1, acc