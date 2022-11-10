import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from config.config import (
    cols,
    rows,
    MultiModal_T1_data_path,
    MultiModal_T2_data_path,
    MultiModal_T1_label_path,
    MultiModal_T2_label_path,
    random_list,
)

R25 = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 25, 1)


class PancreasCancerDatasetAugmentationFold(Dataset):
    def __init__(self, fold, im_size=None):
        if im_size is None:
            im_size = [rows, cols]

        train_list = random_list.copy()

        if fold == 0:
            del train_list[fold * 16 : (fold + 1) * 16]
        else:
            del train_list[fold * 17 - 1 : (fold + 1) * 17 - 1]

        self.im_ht = im_size[0]
        self.im_wd = im_size[1]
        self.patient_index = []
        self.count = []

        for i in range(len(train_list)):
            temp_label = np.load(
                MultiModal_T1_label_path + "{}".format(train_list[i]) + ".npy"
            )

            for j in range(temp_label.shape[0]):
                self.patient_index.append(int(train_list[i]))
                self.count.append(j)

    def __getitem__(self, index):
        img_t1 = np.zeros((1, self.im_ht, self.im_wd))
        mask_t1 = np.zeros((1, self.im_ht, self.im_wd))

        img_t2 = np.zeros((1, self.im_ht, self.im_wd))
        mask_t2 = np.zeros((1, self.im_ht, self.im_wd))

        temp_img = np.load(
            MultiModal_T1_data_path + "{}".format(self.patient_index[index]) + ".npy"
        )
        temp_label = np.load(
            MultiModal_T1_label_path + "{}".format(self.patient_index[index]) + ".npy"
        )

        img_t1[0] = temp_img[self.count[index]]
        mask_t1[0] = temp_label[self.count[index]]

        temp_img = np.load(
            MultiModal_T2_data_path + "{}".format(self.patient_index[index]) + ".npy"
        )
        temp_label = np.load(
            MultiModal_T2_label_path + "{}".format(self.patient_index[index]) + ".npy"
        )

        img_t2[0] = temp_img[self.count[index]]
        mask_t2[0] = temp_label[self.count[index]]

        img_t1 = (img_t1 - 0.11273727) / 0.11880316
        img_t2 = (img_t2 - 0.10151927) / 0.11612002

        rand_augmentation = random.random()

        if 0.25 <= rand_augmentation < 0.5:
            rotation = random.random()
            img_t1[0] = cv2.warpAffine(
                img_t1[0],
                cv2.getRotationMatrix2D(
                    ((cols - 1) / 2.0, (rows - 1) / 2.0), rotation * 25, 1
                ),
                (cols, rows),
            )
            mask_t1[0] = cv2.warpAffine(
                mask_t1[0],
                cv2.getRotationMatrix2D(
                    ((cols - 1) / 2.0, (rows - 1) / 2.0), rotation * 25, 1
                ),
                (cols, rows),
            )

            img_t2[0] = cv2.warpAffine(
                img_t2[0],
                cv2.getRotationMatrix2D(
                    ((cols - 1) / 2.0, (rows - 1) / 2.0), rotation * 25, 1
                ),
                (cols, rows),
            )
            mask_t2[0] = cv2.warpAffine(
                mask_t2[0],
                cv2.getRotationMatrix2D(
                    ((cols - 1) / 2.0, (rows - 1) / 2.0), rotation * 25, 1
                ),
                (cols, rows),
            )

        elif 0.5 <= rand_augmentation < 0.75:
            img_t1[0] = cv2.flip(img_t1[0], 1)
            mask_t1[0] = cv2.flip(mask_t1[0], 1)

            img_t2[0] = cv2.flip(img_t2[0], 1)
            mask_t2[0] = cv2.flip(mask_t2[0], 1)

        elif 0.75 <= rand_augmentation:
            img_t1[0] = cv2.flip(img_t1[0], 0)
            mask_t1[0] = cv2.flip(mask_t1[0], 0)

            img_t2[0] = cv2.flip(img_t2[0], 0)
            mask_t2[0] = cv2.flip(mask_t2[0], 0)

        data_t1 = torch.from_numpy(img_t1).clone()
        label_t1 = torch.from_numpy(mask_t1).clone()

        data_t2 = torch.from_numpy(img_t2).clone()
        label_t2 = torch.from_numpy(mask_t2).clone()

        return data_t1, label_t1, data_t2, label_t2

    def __len__(self):
        return len(self.count)


class PancreasCancerDatasetAugmentationFoldReverse(Dataset):
    def __init__(self, fold, im_size=None):
        if im_size is None:
            im_size = [rows, cols]

        train_list = random_list.copy()

        if fold == 0:
            del train_list[fold * 16 : (fold + 1) * 16]
        else:
            del train_list[fold * 17 - 1 : (fold + 1) * 17 - 1]

        self.im_ht = im_size[0]
        self.im_wd = im_size[1]
        self.patient_index = []
        self.count = []

        for i in range(len(train_list)):
            temp_label = np.load(
                MultiModal_T1_label_path + "{}".format(train_list[i]) + ".npy"
            )

            for j in range(temp_label.shape[0]):
                self.patient_index.append(int(train_list[i]))
                self.count.append(j)

    def __getitem__(self, index):
        img_t1 = np.zeros((1, self.im_ht, self.im_wd))
        mask_t1 = np.zeros((1, self.im_ht, self.im_wd))

        img_t2 = np.zeros((1, self.im_ht, self.im_wd))
        mask_t2 = np.zeros((1, self.im_ht, self.im_wd))

        temp_img = np.load(
            MultiModal_T1_data_path + "{}".format(self.patient_index[index]) + ".npy"
        )
        temp_label = np.load(
            MultiModal_T1_label_path + "{}".format(self.patient_index[index]) + ".npy"
        )

        img_t1[0] = temp_img[self.count[index]]
        mask_t1[0] = temp_label[self.count[index]]

        temp_img = np.load(
            MultiModal_T2_data_path + "{}".format(self.patient_index[index]) + ".npy"
        )
        temp_label = np.load(
            MultiModal_T2_label_path + "{}".format(self.patient_index[index]) + ".npy"
        )

        img_t2[0] = temp_img[self.count[index]]
        mask_t2[0] = temp_label[self.count[index]]

        img_t1 = (img_t1 - 0.11273727) / 0.11880316
        img_t2 = (img_t2 - 0.10151927) / 0.11612002

        rand_augmentation = random.random()

        if 0.25 <= rand_augmentation < 0.5:
            rotation = random.random()
            img_t1[0] = cv2.warpAffine(
                img_t1[0],
                cv2.getRotationMatrix2D(
                    ((cols - 1) / 2.0, (rows - 1) / 2.0), rotation * 25, 1
                ),
                (cols, rows),
            )
            mask_t1[0] = cv2.warpAffine(
                mask_t1[0],
                cv2.getRotationMatrix2D(
                    ((cols - 1) / 2.0, (rows - 1) / 2.0), rotation * 25, 1
                ),
                (cols, rows),
            )

            img_t2[0] = cv2.warpAffine(
                img_t2[0],
                cv2.getRotationMatrix2D(
                    ((cols - 1) / 2.0, (rows - 1) / 2.0), rotation * 25, 1
                ),
                (cols, rows),
            )
            mask_t2[0] = cv2.warpAffine(
                mask_t2[0],
                cv2.getRotationMatrix2D(
                    ((cols - 1) / 2.0, (rows - 1) / 2.0), rotation * 25, 1
                ),
                (cols, rows),
            )

        elif 0.5 <= rand_augmentation < 0.75:
            img_t1[0] = cv2.flip(img_t1[0], 1)
            mask_t1[0] = cv2.flip(mask_t1[0], 1)

            img_t2[0] = cv2.flip(img_t2[0], 1)
            mask_t2[0] = cv2.flip(mask_t2[0], 1)

        elif 0.75 <= rand_augmentation:
            img_t1[0] = cv2.flip(img_t1[0], 0)
            mask_t1[0] = cv2.flip(mask_t1[0], 0)

            img_t2[0] = cv2.flip(img_t2[0], 0)
            mask_t2[0] = cv2.flip(mask_t2[0], 0)

        data_t1 = torch.from_numpy(img_t1).clone()
        label_t1 = torch.from_numpy(mask_t1).clone()

        data_t2 = torch.from_numpy(img_t2).clone()
        label_t2 = torch.from_numpy(mask_t2).clone()

        return data_t2, label_t2, data_t1, label_t1

    def __len__(self):
        return len(self.count)
