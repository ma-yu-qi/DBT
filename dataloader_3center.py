from pathlib import Path
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    RandomSwap,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)
from torchio.data import UniformSampler, LabelSampler
from torchio import ScalarImage, LabelMap, Subject, SubjectsDataset, Queue
import torchio
from torchio import AFFINE, DATA
import torchio as tio
import torch
import sys

sys.path.append("./")


def get_subjects(config):
    """
    @description: get the subjects for normal training
    """
    subjects = []
    if "predict" in config.job_name:
        img_path = Path(config.pred_data_path)
        gt_path = Path(config.pred_gt_path)
    else:
        img_path = Path(config.data_path)
        gt_path = Path(config.gt_path)
    x_generator = sorted(img_path.glob("*.mhd"))
    gt_generator = sorted(gt_path.glob("*.mhd"))
    for i, (source, gt) in enumerate(zip(x_generator, gt_generator)):
        subject = tio.Subject(
            source=tio.ScalarImage(source),
            gt=tio.LabelMap(gt),
        )
        subjects.append(subject)
    return subjects


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.subjects = []

        queue_length = 10
        samples_per_volume = 10

        self.subjects = self.load_samples(config)

        self.transforms = self.transform(config)

        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)

        self.queue_dataset = tio.Queue(
            self.training_set, queue_length, samples_per_volume, tio.UniformSampler(patch_size=config.patch_size), num_workers=16
        )

    def load_samples(self, config):
        img_path = Path(config.data_path)
        gt_path = Path(config.gt_path)
        
        subjects = []
        x_generator = sorted(img_path.glob("*.mhd"))
        gt_generator = sorted(gt_path.glob("*.mhd"))
        
        for i, (source, gt) in enumerate(zip(x_generator, gt_generator)):
            hospital_id = self.get_hospital_id(source.name)  # 调用 get_hospital_id 方法
            subject = tio.Subject(
                source=tio.ScalarImage(source),
                gt=tio.LabelMap(gt),
                hospital_id=hospital_id  # 为 Subject 附加医院 ID 标签
            )
            subjects.append(subject)
        return subjects

    def get_hospital_id(self, filename):
        if "ADAM" in filename:
            return 0  # 医院 1 的标签
        elif "MRA" in filename:
            return 1  # 医院 2 的标签
        elif "LocH1" in filename:
            return 2  # 医院 3 的标签
        else:
            raise ValueError(f"未知的医院标签：{filename}")

    def __getitem__(self, idx):
        # 修改 __getitem__ 方法，使其包含 hospital_id 信息
        subject = self.subjects[idx]
        data = subject['source'][tio.DATA]  # 获取数据
        label = subject['gt'][tio.DATA]     # 获取标签
        hospital_id = subject['hospital_id']  # 获取医院标签

        return {"data": data, "label": label, "hospital_id": hospital_id}


    def transform(self, config):
        if config.aug:
            training_transform = Compose(
                [
                    # CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                    # RandomMotion(),
                    RandomBiasField(),
                    ZNormalization(),
                    RandomNoise(),
                    RandomFlip(axes=(0,)),
                    OneOf(
                        {
                            RandomAffine(): 0.8,
                            RandomElasticDeformation(): 0.2,
                        }
                    ),
                ]
            )
        else:
            if "train" in config.job_name:
                training_transform = Compose(
                    [
                        # CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                        # RandomAffine(degrees=20),
                        # RandomNoise(std=0.0001),
                        ZNormalization(),
                    ]
                )
            elif "predict" in config.job_name:
                training_transform = Compose(
                    [
                        ZNormalization(),
                    ]
                )
            else:
                training_transform = Compose(
                    [
                        # CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                        # RandomAffine(degrees=20),
                        # RandomNoise(std=0.0001),
                        ZNormalization(),
                    ]
                )
        return training_transform
