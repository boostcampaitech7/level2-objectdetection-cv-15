from typing import Optional
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Subset

from src.data.base_datamodule import BaseDataModule
from src.data.collate_fns.recycle_collate_fn import recycle_collate_fn
from src.data.datasets.recycle_dataset import CocoDetection
from src.data.processors.auto_image_processor import get_auto_image_processor
from src.utils.data_utils import load_yaml_config 


class RecycleDataModule(BaseDataModule):
    def __init__(self, data_config_path: str, augmentation_config_path: str, processor_config_path: str, seed: int=42):
        self.data_config = load_yaml_config(data_config_path)
        self.augmentation_config = load_yaml_config(augmentation_config_path)
        self.processor_config = load_yaml_config(processor_config_path)
        self.seed = self.data_config['seed']
        self.collate_fn = None
        super().__init__(self.data_config)

    def setup(self, stage: Optional[str] = None):
        # Load datasets
        if self.augmentation_config["augmentation"]["use_augmentation"]:
            train_transforms = self._get_augmentation_transforms()
        else:
            train_transforms = A.Compose(
                [ToTensorV2()]
            )

        test_transforms = A.Compose(
            [ToTensorV2()]
        )

        processor = get_auto_image_processor(self.processor_config)
        self.collate_fn = lambda batch: recycle_collate_fn(batch, processor)

        raw_data_path = self.config["data"]["raw_data_path"]
        train_ann_file = self.config["data"]["train_ann_file"]
        test_ann_file = self.config["data"]["test_ann_file"]

        full_dataset = CocoDetection(
            img_folder=raw_data_path, 
            ann_file=train_ann_file, 
            transform=train_transforms, 
            processor=processor, 
            train=True
        )

        # Split train dataset into train and validation
        full_size = int(
            len(full_dataset)
        )

        indices = list(range(full_size))
        train_indices, val_indices = train_test_split(
            indices, 
            train_size=self.config["data"]["train_val_split"], 
            random_state=self.seed
        )

        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(
            CocoDetection(
                img_folder=raw_data_path,
                ann_file=train_ann_file,
                transform=test_transforms,
                processor=processor,
                train=False
            ),
            val_indices
        )

        self.test_dataset = CocoDetection(
            img_folder=raw_data_path, 
            ann_file=test_ann_file, 
            transform=test_transforms, 
            processor=processor, 
            train=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["data"]["train"]["batch_size"],
            num_workers=self.config["data"]["train"]["num_workers"],
            shuffle=True,
            collate_fn=self.collate_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["data"]["val"]["batch_size"],
            num_workers=self.config["data"]["val"]["num_workers"],
            shuffle=False,
            collate_fn=self.collate_fn,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["data"]["test"]["batch_size"],
            num_workers=self.config["data"]["test"]["num_workers"],
            shuffle=False,
            collate_fn=self.collate_fn,
            persistent_workers=True,
        )
    
    def _get_augmentation_transforms(self):
        transform_list = []
        for transform_config in self.augmentation_config["augmentation"]["transforms"]:
            transform_name = transform_config["name"]
            
            if transform_name == "SomeOf":
                # SomeOf 변환을 따로 처리
                sub_transforms = []
                for sub_transform in transform_config["params"]["transforms"]:
                    transform_class = getattr(A, sub_transform["name"])
                    sub_transforms.append(transform_class(**sub_transform["params"]))
                
                # SomeOf 변환 추가
                transform_list.append(A.SomeOf(
                    sub_transforms, 
                    p=transform_config["params"]["p"],
                    n=transform_config["params"]["n"],
                    ))
                
            else:
                transform_class = getattr(A, transform_name)
                transform_list.append(transform_class(**transform_config["params"]))

        transform_list.append(ToTensorV2())
        
        return A.Compose(
            transform_list,
            bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'])
        )
