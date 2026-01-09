# Modified by Qianyu Zhou and Lu He
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask
from .coco_video_parser import CocoVID
from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms_multi as T
from torch.utils.data.dataset import ConcatDataset
from collections import defaultdict

import random
import copy

class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, num_frames = 4,
        is_train = True,  filter_key_img=True,  cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        # self.prepare_seq = ConvertCocoSeqPolysToMask(return_masks)
        self.ann_file = ann_file
        self.frame_range = [-2, 2]
        self.num_ref_frames = num_frames - 1
        self.cocovid = CocoVID(self.ann_file)
        self.is_train = is_train
        self.filter_key_img = filter_key_img

        """
        img_folder:图片根目录（COCO JSON 的 file_name 会在这个根下找）
        ann_file：COCO 风格标注 JSON 路径
        transforms：多帧一致的数据增强/归一化方法流水线
        return_masks：是否生成 instance masks（分割才用；检测一般 False）
        num_frames：总帧数（当前帧 + 参考帧），代码里 self.num_ref_frames = num_frames - 1
        is_train：训练/验证开关，决定参考帧怎么取（训练随机、验证规则
        filter_key_img：是否把当前帧从参考帧候选里剔除（避免采到自己）
        cache_mode, local_rank, local_size：分布式/缓存相关，传给父类
        """

    def __getitem__(self, idx):
        """
        每产生一个样本就调用一次__getitem__
        Args:
            idx (int): Index,每个idx返回的是一个clip（当前帧 + 参考帧）.一个样本 == 一个clip
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        imgs = []
        tgts = []

        coco = self.coco
        # 选中第 idx 个“关键帧”作为当前样本的主帧，再去采参考帧组成一个 clip
        # 原代码中没有对采样范围做限制，所以self.ids相当于train.json里images字段包含的所有 image_id 列表
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        video_id = img_info['video_id']
        img = self.get_image(path)
        target = {'image_id': img_id,'video_id': video_id, 'annotations': target}
        img, target = self.prepare(img, target)
        imgs.append(img)
        tgts.append(target)
        if video_id == -1:
            for i in range(self.num_ref_frames):
                imgs.append(copy.deepcopy(img))
                tgts.append(copy.deepcopy(target))
        else:
            img_ids = self.cocovid.get_img_ids_from_vid(video_id) 
            #print("length", len(img_ids))
            ref_img_ids = []
            if self.is_train:
                interval = 5 # *20
                left = max(img_ids[0], img_id - interval)
                right = min(img_ids[-1], img_id + interval)
                sample_range = list(range(left, right))
                if self.filter_key_img and img_id in sample_range:
                    sample_range.remove(img_id)
                if self.num_ref_frames >= 10:
                    sample_range = img_ids
                while self.num_ref_frames > len(sample_range):
                    sample_range.extend(sample_range)
                ref_img_ids = random.sample(sample_range, self.num_ref_frames)

            else:
                #print("------------------------------")i
                ref_img_ids = []
                Len = len(img_ids)
                interval  = max(int(Len // 15), 1)  #
                left_indexs = int((img_id - img_ids[0]) // interval)
                right_indexs = int((img_ids[-1] - img_id) // interval)
                if left_indexs < self.num_ref_frames:
                   for i in range(self.num_ref_frames):
                       ref_img_ids.append(min(img_id + (i+1)*interval, img_ids[-1]))
                else:
                   for i in range(self.num_ref_frames):
                       ref_img_ids.append(max(img_id - (i+1)* interval, img_ids[0]))

                # print("ref_img_ids", ref_img_ids)
            for ref_img_id in ref_img_ids:
                ref_ann_ids = coco.getAnnIds(imgIds=ref_img_id)
                ref_img_info = coco.loadImgs(ref_img_id)[0]
                ref_img_path = ref_img_info['file_name']
                ref_img = self.get_image(ref_img_path)
                ref_target = coco.loadAnns(ref_ann_ids)
                ref_target = {'image_id': ref_img_id, 'video_id': video_id, 'annotations': ref_target}
                ref_img, ref_target = self.prepare(ref_img, ref_target)
                imgs.append(ref_img)
                tgts.append(ref_target)

        if self._transforms is not None:
            imgs, target = self._transforms(imgs, tgts) 


        """
        torch.cat(imgs, dim=0):返回拼接后的多帧图像张量,imgs 原本是长度 num_frames 的 list，每个元素是 3×H×W（ToTensor 后）。
                                torch.cat(imgs, dim=0) 变成 (3*num_frames) × H × W。
        target:单帧标注经 ConvertCocoPolysToMask 后至少包含：boxes（xyxy）、labels、image_id、area、iscrowd、orig_size、size
                                （若 return_masks=True 还有 masks）
        """
        return  torch.cat(imgs, dim=0),  target


class CocoDetectionRandomSampling(TvCocoDetection):
    def __init__(
        self, img_folder, ann_file, transforms, return_masks, num_frames=4,
        is_train=True, filter_key_img=True,
        cache_mode=False, local_rank=0, local_size=1,
        samples_per_video=0, ref_interval=5
    ):
        """
        随机采样新增
        :param samples_per_video: 每个 epoch 每个视频抽取多少个样本（关键帧），0 表示关闭随机关键帧采样，恢复原逻辑（遍历 self.ids），把训练集长度从“所有帧总数”改成“视频数 × samples_per_video”
        :param ref_interval: 参考帧只在关键帧附近的时间窗口内采样（以帧序列下标为准）
        """
        super(CocoDetectionRandomSampling, self).__init__(
            img_folder, ann_file, cache_mode=cache_mode,
            local_rank=local_rank, local_size=local_size
        )
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.ann_file = ann_file
        self.num_ref_frames = num_frames - 1
        self.cocovid = CocoVID(self.ann_file)
        self.is_train = is_train
        self.filter_key_img = filter_key_img

        # === NEW:random key-frame sampling controls ===
        self.samples_per_video = int(samples_per_video)  # 0 表示关闭（保持原逻辑）
        self.ref_interval = int(ref_interval)
        self._epoch = 0  # 用于做可复现随机：同一个 epoch 内，同一个 idx 采样结果稳定；epoch 变了采样也跟着变

        # 训练模式 + samples_per_video > 0，才会走“按视频随机抽 key frame”的路线
        self._use_random_key = self.is_train and (self.samples_per_video > 0)
        if self._use_random_key:
            video2img = defaultdict(list)
            # self.coco.imgs: {img_id: {"file_name":..., "video_id":..., ...}, ...}
            for img_id, info in self.coco.imgs.items():
                vid = info.get("video_id", -1)
                if vid != -1:
                    video2img[vid].append(img_id)

            self.video_ids = sorted(video2img.keys())  # 一个列表，包含所有训练视频的video_id（anti_uav这里是 160 个
            self.video2img_ids = {vid: sorted(ids) for vid, ids in video2img.items()}   # 这个视频下所有帧的 img_id 列表

            # 如果标注里没有 video_id，就自动回退到原逻辑
            if len(self.video_ids) == 0:
                self._use_random_key = False


    def set_epoch(self, epoch: int):
        # 在训练每个 epoch 开始时调用 dataset.set_epoch(epoch)，这样随机种子会随 epoch 变化
        self._epoch = int(epoch)

    def __len__(self): # 数据量
        # 现在训练时 len(dataset) = 160 × samples_per_video
        # 训练：每个视频每个 epoch 只采 samples_per_video 个 key frame
        if getattr(self, "_use_random_key", False):
            return len(self.video_ids) * self.samples_per_video
        return len(self.ids)

    def __getitem__(self, idx):  # 按索引返回一条样本
        imgs, tgts = [], []
        coco = self.coco

        # ===== 1) 选择关键帧 img_id（训练时随机；否则原逻辑遍历 self.ids）=====
        if getattr(self, "_use_random_key", False):

            # e.g.：idx=0~19 → 取 video_ids[0]（第 1 个视频）
            # idx=20~39 → 取 video_ids[1]（第 2 个视频）
            vid = self.video_ids[idx // self.samples_per_video]
            rng = random.Random(self._epoch * 1000003 + idx)  # 可复现
            img_ids_this_vid = self.video2img_ids[vid]
            img_id = rng.choice(img_ids_this_vid)
        else:
            rng = random
            img_id = self.ids[idx]
            img_ids_this_vid = None

        # ===== 2) 读取关键帧与标注 =====
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        video_id = img_info.get('video_id', -1)

        img = self.get_image(path)
        target = {'image_id': img_id, 'video_id': video_id, 'annotations': target}
        img, target = self.prepare(img, target)

        imgs.append(img)
        tgts.append(target)

        # ===== 3) 采样参考帧 ref_img_ids 并读取 =====
        if video_id == -1:
            # 不是视频帧：用同一帧复制补齐
            for _ in range(self.num_ref_frames):
                imgs.append(copy.deepcopy(img))
                tgts.append(copy.deepcopy(target))
        else:
            # 优先用缓存的该视频帧列表（随机 key 模式下）
            if img_ids_this_vid is not None and video_id in getattr(self, "video2img_ids", {}):
                img_ids = self.video2img_ids[video_id]
            else:
                img_ids = self.cocovid.get_img_ids_from_vid(video_id)
                img_ids = sorted(img_ids)

            if self.is_train:
                # 训练：在 key 附近 +/- ref_interval 的“下标窗口”内随机采样
                try:
                    pos = img_ids.index(img_id)
                except ValueError:
                    pos = 0

                left = max(0, pos - self.ref_interval)
                right = min(len(img_ids), pos + self.ref_interval + 1)
                sample_range = img_ids[left:right]

                if self.filter_key_img:
                    sample_range = [x for x in sample_range if x != img_id]

                # 原代码逻辑：num_ref_frames 很大时，允许全局采样
                if self.num_ref_frames >= 10:
                    sample_range = [x for x in img_ids if (not self.filter_key_img or x != img_id)]

                while self.num_ref_frames > len(sample_range):
                    sample_range = sample_range + sample_range

                ref_img_ids = rng.sample(sample_range, self.num_ref_frames)

            else:
                # 验证：按序列下标做“向前/向后”取样（不依赖 image_id 连续性）
                Len = len(img_ids)
                step = max(int(Len // 15), 1)
                try:
                    # 找到关键帧在该视频帧列表里的位置
                    pos = img_ids.index(img_id)
                except ValueError:
                    pos = 0

                ref_pos = []
                if pos < self.num_ref_frames:
                    for i in range(self.num_ref_frames):
                        ref_pos.append(min(pos + (i + 1) * step, Len - 1))
                else:
                    for i in range(self.num_ref_frames):
                        ref_pos.append(max(pos - (i + 1) * step, 0))

                ref_img_ids = [img_ids[p] for p in ref_pos]

            for ref_img_id in ref_img_ids:
                ref_ann_ids = coco.getAnnIds(imgIds=ref_img_id)
                ref_img_info = coco.loadImgs(ref_img_id)[0]
                ref_img_path = ref_img_info['file_name']

                ref_img = self.get_image(ref_img_path)
                ref_target = coco.loadAnns(ref_ann_ids)
                ref_target = {'image_id': ref_img_id, 'video_id': video_id, 'annotations': ref_target}
                ref_img, ref_target = self.prepare(ref_img, ref_target)

                imgs.append(ref_img)
                tgts.append(ref_target)

        # ===== 4) transforms（多帧同步增强）+ 返回 =====
        if self._transforms is not None:
            imgs, target = self._transforms(imgs, tgts)

        return torch.cat(imgs, dim=0), target



def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # classes = [obj["category_id"] for obj in anno]
        # 只有一个类，所以减一
        classes = [obj["category_id"] -1 for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        
        return image, target


def make_coco_transforms(image_set):
    """

    :param image_set: 是一个String类型的

    针对train/val/test，构造不同的数据增强流水线,

    返回的是一套数据变化的方法，（也就也是几种处理图片的函数的不同组合）
    """

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train_vid' or image_set == "train_det" or image_set == "train_joint":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize([600], max_size=1000),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([600], max_size=1000),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.vid_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train_det": [(root / "Data" / "DET", root / "annotations" / 'imagenet_det_30plus1cls_vid_train.json')],
        "train_vid": [(root / "Data" / "VID", root / "annotations" / 'imagenet_vid_train.json')],
        # "train_joint": [(root / "Data" , root / "annotations" / 'imagenet_vid_train_joint_30.json')],
        # "val": [(root / "Data" / "VID", root / "annotations" / 'imagenet_vid_val.json')],
        # 改成我自己的命名
        "train_joint": [(root / "Data" , root / "annotations" / 'anti_uav_train.json')],
        "val":          [(root / "Data", root / "annotations" / 'anti_uav_val.json')],
    }
    datasets = []
    for (img_folder, ann_file) in PATHS[image_set]:

        """
        CocoDetection 新增随机采样
        """
        # dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), is_train =(not args.eval), return_masks=args.masks, cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(), num_frames=args.num_frames)
        dataset = CocoDetectionRandomSampling(img_folder, ann_file, transforms=make_coco_transforms(image_set), is_train =(not args.eval), return_masks=args.masks, cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(), num_frames=args.num_frames)
        dataset = CocoDetectionRandomSampling(img_folder,ann_file,transforms=make_coco_transforms(image_set),
                                              is_train =(not args.eval), return_masks=args.masks,
                                              cache_mode=args.cache_mode,samples_per_video=args.samples_per_video)
        datasets.append(dataset)
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)

    
