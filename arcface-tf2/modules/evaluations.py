"""
This script was modified from https://github.com/ZhaoJ9014/face.evoLVe.PyTorch
"""
import os
import cv2
import bcolz
import numpy as np
import tqdm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from .utils import l2_norm


def get_val_pair(path, name, binary=True): #LYJ
    img_dir = os.path.join(path, name)
    if binary:
        arr = bcolz.carray(rootdir=img_dir, mode='r')
    else:
        arr = np.empty((0, 3, 112, 112))
        img_list = sorted([file for file in os.listdir(img_dir) if not file.startswith('.')])
        for img_name in tqdm.tqdm(img_list):
            img_path = os.path.join(img_dir, img_name)
            # print(img_path)
            try:
                img_arr = cv2.resize(cv2.imread(img_path), (112, 112)).swapaxes(0, 2).swapaxes(1, 2) #차원, 행, 열
            except:
                print('failed to read: ', img_path)
            img_arr = np.expand_dims(img_arr, 0)
            # img_arr_toshow = cv2.cvtColor(img_arr[0].swapaxes(0, 2).swapaxes(0, 1), cv2.COLOR_BGR2RGB)
            # plt.imshow(img_arr_toshow)
            # plt.show()
            # print(arr.shape, img_arr.shape)
            if img_arr.shape != (1, 3, 112, 112):
                print('shape differs: ', img_path)
            arr = np.concatenate([arr, img_arr], axis=0)
    issame = np.load('{}/{}_list.npy'.format(path, name))

    return arr, issame


def get_val_data(data_path):
    """get validation data"""
    lfw, lfw_issame = get_val_pair(data_path, 'lfw_align_112/lfw')
    agedb_30, agedb_30_issame = get_val_pair(data_path,
                                             'agedb_align_112/agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_align_112/cfp_fp')

    return lfw, agedb_30, cfp_fp, lfw_issame, agedb_30_issame, cfp_fp_issame


def ccrop_batch(imgs):
    assert len(imgs.shape) == 4
    resized_imgs = np.array([cv2.resize(img, (128, 128)) for img in imgs])
    ccropped_imgs = resized_imgs[:, 8:-8, 8:-8, :]

    return ccropped_imgs


def hflip_batch(imgs):
    assert len(imgs.shape) == 4
    return imgs[:, :, ::-1, :]


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame),
                               np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame,
                  nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    print('distance is: ', dist)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)

        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = \
                calculate_accuracy(threshold,
                                   dist[test_set],
                                   actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index],
            dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)

    vis_ROC_curve(tprs, fprs, (nrof_folds, nrof_thresholds), thresholds)
    return tpr, fpr, accuracy, best_thresholds


def evaluate(embeddings, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2] # 0, 2, 4, 6, 8 LYJ
    embeddings2 = embeddings[1::2] # 1, 3, 5, 7, 9 LYJ
    if embeddings1.shape != embeddings2.shape:
        print('emb 1 and 2 shape different')
    tpr, fpr, accuracy, best_thresholds = calculate_roc(
        thresholds, embeddings1, embeddings2, np.asarray(actual_issame),
        nrof_folds=nrof_folds)

    return tpr, fpr, accuracy, best_thresholds


def perform_val(embedding_size, batch_size, model,
                carray, issame, nrof_folds=10, is_ccrop=False, is_flip=True):
    """perform val"""
    embeddings = np.zeros([len(carray), embedding_size])

    for idx in tqdm.tqdm(range(0, len(carray), batch_size)):
        batch = carray[idx:idx + batch_size]
        batch = np.transpose(batch, [0, 2, 3, 1]) * 0.5 + 0.5
        batch = batch[:, :, :, ::-1]  # convert BGR to RGB

        if is_ccrop:
            batch = ccrop_batch(batch)
        if is_flip:
            fliped = hflip_batch(batch)
            emb_batch = model(batch) + model(fliped)
            embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
        else:
            emb_batch = model(batch)
            embeddings[idx:idx + batch_size] = l2_norm(emb_batch)

    tpr, fpr, accuracy, best_thresholds = evaluate(
        embeddings, issame, nrof_folds)

    return accuracy.mean(), best_thresholds.mean()

def test_registration(registered_embed, test_embed, threshold): #LYJ
    # 1. get model embedding of each img
    # 2. get diff
    assert (registered_embed.shape[0] == test_embed.shape[0])
    assert (registered_embed.shape[1] == test_embed.shape[1])

    diff = np.subtract(registered_embed, test_embed)
    dist = np.sum(np.square(diff), 1)

    # 3. calculate the result
    predict_issame = np.less(dist, threshold)

    return predict_issame, dist #Boolean, Probability
    
def vis_ROC_curve(tprs, fprs, tprs_shape, thresholds):
    #1. get ROC information from multiple threshold results
    print(tprs_shape)
    print(tprs.shape, fprs.shape, thresholds.shape)
    # threshold: 0 to 4, 400 threshold
    # tprs, fprs: K-fold * 400 thres

    #2. plot ROC curve
    for k_tprs, k_fprs in zip(tprs, fprs):
        for tpr, fpr in zip(k_tprs, k_fprs):
            plt.plot(tpr, fpr)
    return