# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple

import albumentations as A
import cv2
import torch

from dataset.base_dataset import _BaseSODDataset
from dataset.transforms.resize import ms_resize, ss_resize
from dataset.transforms.rotate import UniRotate
from utils.builder import DATASETS
from utils.io.genaral import get_datasets_info_with_keys
from utils.io.image import read_color_array, read_gray_array
import numpy as np


@DATASETS.register(name="msi_cod_te")
class MSICOD_TestDataset(_BaseSODDataset):
    def __init__(self, root: Tuple[str, dict], shape: Dict[str, int], interp_cfg: Dict = None):
        super().__init__(base_shape=shape, interp_cfg=interp_cfg)
        # self.datasets = get_datasets_info_with_keys(dataset_infos=[root], extra_keys=["mask", "depth"])
        self.datasets = get_datasets_info_with_keys(dataset_infos=[root], extra_keys=["mask", ])
        self.total_image_paths = self.datasets["image"]
        self.total_mask_paths = self.datasets["mask"]
        # self.total_depth_paths = self.datasets["depth"]
        self.image_norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __getitem__(self, index):
        image_path = self.total_image_paths[index]
        mask_path = self.total_mask_paths[index]
        # depth_path = self.total_depth_paths[index]

        image = read_color_array(image_path)
        # mask_r = np.ones((image.shape[0], image.shape[1], image.shape[2])) * 255
        # image_r = (mask_r - image).astype(np.float32)
        # image_r = self.image_norm(image=image_r)["image"]

        image = self.image_norm(image=image)["image"]
        # depth = read_gray_array(depth_path, to_normalize=True)


        base_h = self.base_shape["h"]
        base_w = self.base_shape["w"]


        images = ms_resize(image, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)
        # depth = ss_resize(depth, scale=1.0, base_h=base_h, base_w=base_w)
        # depth_1_0 = torch.from_numpy(depth).unsqueeze(0)
        # images_r = ms_resize(image_r, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)
        # mask_rs = ms_resize(mask_r, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)

        # image_0_5 = torch.from_numpy((mask_rs[0] - images[0]).astype(np.float32)).permute(2, 0, 1)
        image1 = images[0]
        # image1[0] = 255 - image1[0]
        # image_0_5 = torch.from_numpy(images[0]).permute(2, 0, 1)
        image_0_5 = torch.from_numpy(image1).permute(2, 0, 1)
        # image_0_5_r = torch.from_numpy(images_r[0]).permute(2, 0, 1)
        # image_0_5 = image_0_5 + image_0_5_r.sigmoid() * image_0_5
        image1 = images[1]
        # image1[1] = 255 - image1[1]
        # image_1_0 = torch.from_numpy(images[1]).permute(2, 0, 1)
        image_1_0 = torch.from_numpy(image1).permute(2, 0, 1)
        # image_1_0_r = torch.from_numpy(images_r[1]).permute(2, 0, 1)
        # image_1_0 = image_1_0 + image_1_0_r.sigmoid() * image_1_0
        # image_1_0 = torch.from_numpy(image).permute(2, 0, 1)
        # image_1_0_r = torch.from_numpy(image_r).permute(2, 0, 1)
        # image_1_5 = torch.from_numpy((mask_rs[2] - images[2]).astype(np.float32)).permute(2, 0, 1)
        image1 = images[2]
        # image1[2] = 255 - image1[2]
        # image_1_5 = torch.from_numpy(images[2]).permute(2, 0, 1)
        image_1_5 = torch.from_numpy(image1).permute(2, 0, 1)
        # image_1_5_r = torch.from_numpy(images_r[2]).permute(2, 0, 1)
        # image_1_5 = image_1_5 + image_1_5_r.sigmoid() * image_1_5

        return dict(
            data={
                # "image1.5": image_1_5,
                "image1.0": image_1_0,
                # "depth": depth_1_0,
                # "image0.5": image_0_5,
                # "image1.0_r": image_1_0_r,
                # "image0.5": image_0_5,
            },
            info=dict(
                mask_path=mask_path,
            ),
        )

    def __len__(self):
        return len(self.total_image_paths)


@DATASETS.register(name="msi_cod_tr")
class MSICOD_TrainDataset(_BaseSODDataset):
    def __init__(
        self, root: List[Tuple[str, dict]], shape: Dict[str, int], extra_scales: List = None, interp_cfg: Dict = None
    ):
        super().__init__(base_shape=shape, extra_scales=extra_scales, interp_cfg=interp_cfg)
        self.datasets = get_datasets_info_with_keys(dataset_infos=root, extra_keys=["mask", "edge", "depth"])
        self.total_image_paths = self.datasets["image"]
        self.total_mask_paths = self.datasets["mask"]
        self.total_edge_paths = self.datasets["edge"]
        self.total_depth_paths = self.datasets["depth"]
        self.joint_trans = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                UniRotate(limit=10, interpolation=cv2.INTER_LINEAR, p=0.5),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ],
        )
        self.image_norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.reszie = A.Resize

    def __getitem__(self, index):
        image_path = self.total_image_paths[index]
        mask_path = self.total_mask_paths[index]
        # print(index)
        edge_path = self.total_edge_paths[index]
        depth_path = self.total_depth_paths[index]


        image = read_color_array(image_path)
        mask = read_gray_array(mask_path, to_normalize=True, thr=0.5)
        edge = read_gray_array(edge_path, to_normalize=True, thr=0.5)
        depth = read_gray_array(depth_path, to_normalize=True)
        # edge = mask
        # edge = edge.astype(np.float32)


        # kernel = np.ones((17, 17), np.uint8)

        # 图像膨胀处理
        # edge = cv2.dilate(edge, kernel)

        # mask_r = np.ones((image.shape[0], image.shape[1], image.shape[2])) * 255
        # image_r = (mask_r - image).astype(np.float32)

        transformed = self.joint_trans(image=image, masks=[mask, edge, depth])
        # transformed_e = self.joint_trans(image=image, mask=edge)
        image = transformed["image"]
        image = self.image_norm(image=image)["image"]
        mask = transformed["masks"][0]
        edge = transformed["masks"][1]
        depth = transformed["masks"][2]



        # transformed_r = self.joint_trans(image=image_r, mask=mask)
        # image_r = transformed_r["image"]




        base_h = self.base_shape["h"]
        base_w = self.base_shape["w"]
        images = ms_resize(image, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)
        # images_r = ms_resize(image_r, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)
        # print(images[0].dtype)
        # mask_rs = ms_resize(mask_r, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)
        # print((mask_rs[0] - images[0]).dtype)
        # image_0_5 = torch.from_numpy((mask_rs[0] - images[0]).astype(np.float32)).permute(2, 0, 1)
        image1 = images[0]
        # image1[0] = 255 - image1[0]
        # image_0_5 = torch.from_numpy(images[0]).permute(2, 0, 1)
        image_0_5 = torch.from_numpy(image1).permute(2, 0, 1)
        # image_0_5_r = torch.from_numpy(images_r[0]).permute(2, 0, 1)
        # image_0_5 = image_0_5 + image_0_5_r.sigmoid() * image_0_5
        image1 = images[1]
        # image1[1] = 255 - image1[1]
        # image_1_0 = torch.from_numpy(images[1]).permute(2, 0, 1)
        image_1_0 = torch.from_numpy(image1).permute(2, 0, 1)
        # image_1_0_r = torch.from_numpy(images_r[1]).permute(2, 0, 1)
        # image_1_0 = image_1_0 + image_1_0_r.sigmoid() * image_1_0
        # image_1_0 = torch.from_numpy(image).permute(2, 0, 1)
        # image_1_0_r = torch.from_numpy(image_r).permute(2, 0, 1)
        # image_1_5 = torch.from_numpy((mask_rs[2] - images[2]).astype(np.float32)).permute(2, 0, 1)
        image1 = images[2]
        # image1[2] = 255 - image1[2]
        # image_1_5 = torch.from_numpy(images[2]).permute(2, 0, 1)
        image_1_5 = torch.from_numpy(image1).permute(2, 0, 1)
        # image_1_5_r = torch.from_numpy(images_r[2]).permute(2, 0, 1)
        # image_1_5 = image_1_5 + image_1_5_r.sigmoid() * image_1_5

        mask = ss_resize(mask, scale=1.0, base_h=base_h, base_w=base_w)
        mask_1_0 = torch.from_numpy(mask).unsqueeze(0)

        edge = ss_resize(edge, scale=1.0, base_h=base_h, base_w=base_w)
        edge_1_0 = torch.from_numpy(edge).unsqueeze(0)

        depth = ss_resize(depth, scale=1.0, base_h=base_h, base_w=base_w)
        depth_1_0 = torch.from_numpy(depth).unsqueeze(0)

        # mask1 = ms_resize(mask, scale=(0.5, 0.25, 0.125, 0.0625, 0.03125), base_h=base_h, base_w=base_w)
        # mask_0_5 = torch.from_numpy(mask1[0]).unsqueeze(0)
        # mask_0_25 = torch.from_numpy(mask1[1]).unsqueeze(0)
        # mask_0_125 = torch.from_numpy(mask1[2]).unsqueeze(0)
        # mask_0_0625 = torch.from_numpy(mask1[3]).unsqueeze(0)
        # mask_0_03125 = torch.from_numpy(mask1[4]).unsqueeze(0)


        return dict(
            data={
                # "image1.5": image_1_5,
                "image1.0": image_1_0,
                # "image0.5": image_0_5,
                # "image1.0_r": image_1_0_r,
                "mask": mask_1_0,
                "edge": edge_1_0,
                "depth": depth_1_0,
                # "mask_0_5": mask_0_5,
                # "mask_0_25": mask_0_25,
                # "mask_0_125": mask_0_125,
                # "mask_0_0625": mask_0_0625,
                # "mask_0_03125": mask_0_03125,

            }
        )

    def __len__(self):
        return len(self.total_image_paths)

@DATASETS.register(name="cross_val_msi_cod_tr")
class Cross_MSICOD_TrainDataset(_BaseSODDataset):
    def __init__(
        self, root: List[Tuple[str, dict]], shape: Dict[str, int], extra_scales: List = None, interp_cfg: Dict = None
    ):
        super().__init__(base_shape=shape, extra_scales=extra_scales, interp_cfg=interp_cfg)
        self.datasets = get_datasets_info_with_keys(dataset_infos=root, extra_keys=["mask", "edge"])
        self.total_image_paths = self.datasets["image"]
        self.total_mask_paths = self.datasets["mask"]
        self.total_edge_paths = self.datasets["edge"]
        self.joint_trans = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                UniRotate(limit=10, interpolation=cv2.INTER_LINEAR, p=0.5),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ],
        )
        self.image_norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.reszie = A.Resize

    def __getitem__(self, index):
        image_path = self.total_image_paths[index]
        mask_path = self.total_mask_paths[index]
        # print(index)
        edge_path = self.total_edge_paths[index]
        # depth_path = self.total_depth_paths[index]


        image = read_color_array(image_path)
        mask = read_gray_array(mask_path, to_normalize=True, thr=0.5)
        edge = read_gray_array(edge_path, to_normalize=True, thr=0.5)

        transformed = self.joint_trans(image=image, masks=[mask, edge])
        # transformed_e = self.joint_trans(image=image, mask=edge)
        image = transformed["image"]
        image = self.image_norm(image=image)["image"]
        mask = transformed["masks"][0]
        edge = transformed["masks"][1]
        # depth = transformed["masks"][2]



        # transformed_r = self.joint_trans(image=image_r, mask=mask)
        # image_r = transformed_r["image"]




        base_h = self.base_shape["h"]
        base_w = self.base_shape["w"]
        images = ms_resize(image, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)

        image1 = images[0]
        image_0_5 = torch.from_numpy(image1).permute(2, 0, 1)
        image1 = images[1]

        image_1_0 = torch.from_numpy(image1).permute(2, 0, 1)

        image1 = images[2]

        image_1_5 = torch.from_numpy(image1).permute(2, 0, 1)


        mask = ss_resize(mask, scale=1.0, base_h=base_h, base_w=base_w)
        mask_1_0 = torch.from_numpy(mask).unsqueeze(0)

        edge = ss_resize(edge, scale=1.0, base_h=base_h, base_w=base_w)
        edge_1_0 = torch.from_numpy(edge).unsqueeze(0)

        # depth = ss_resize(depth, scale=1.0, base_h=base_h, base_w=base_w)
        # depth_1_0 = torch.from_numpy(depth).unsqueeze(0)

        # mask1 = ms_resize(mask, scale=(0.5, 0.25, 0.125, 0.0625, 0.03125), base_h=base_h, base_w=base_w)
        # mask_0_5 = torch.from_numpy(mask1[0]).unsqueeze(0)
        # mask_0_25 = torch.from_numpy(mask1[1]).unsqueeze(0)
        # mask_0_125 = torch.from_numpy(mask1[2]).unsqueeze(0)
        # mask_0_0625 = torch.from_numpy(mask1[3]).unsqueeze(0)
        # mask_0_03125 = torch.from_numpy(mask1[4]).unsqueeze(0)


        return dict(
            data={
                # "image1.5": image_1_5,
                "image1.0": image_1_0,
                # "image0.5": image_0_5,
                # "image1.0_r": image_1_0_r,
                "mask": mask_1_0,
                "edge": edge_1_0,
                # "depth": depth_1_0,
                # "mask_0_5": mask_0_5,
                # "mask_0_25": mask_0_25,
                # "mask_0_125": mask_0_125,
                # "mask_0_0625": mask_0_0625,
                # "mask_0_03125": mask_0_03125,

            }
        )

    def __len__(self):
        return len(self.total_image_paths)
