"""
PyTorch Lightning DataModule for RatSLAM bag-extracted places.

Wraps BagPlacesDataset for use with PyTorch Lightning Trainer.
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as T

from .BagPlacesDataset import BagPlacesDataset

IMAGENET_MEAN_STD = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}


class BagPlacesDataModule(pl.LightningDataModule):
    """
    DataModule for training SALAD on RatSLAM bag data.
    
    Args:
        train_csv: Path to training places.csv
        val_csv: Optional path to validation places.csv (if None, no validation)
        batch_size: Number of places per batch
        img_per_place: Images to sample per place
        min_img_per_place: Minimum images required per place
        image_size: Target image size (H, W)
        num_workers: DataLoader workers
        random_sample: Randomly sample images from each place
        use_augmentation: Apply RandAugment during training
    """
    
    def __init__(self,
                 train_csv: str,
                 val_csv: str = None,
                 batch_size: int = 32,
                 img_per_place: int = 4,
                 min_img_per_place: int = 4,
                 image_size: tuple = (322, 322),
                 num_workers: int = 4,
                 random_sample: bool = True,
                 use_augmentation: bool = True):
        super().__init__()
        
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.batch_size = batch_size
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.image_size = image_size
        self.num_workers = num_workers
        self.random_sample = random_sample
        self.use_augmentation = use_augmentation
        
        self.mean = IMAGENET_MEAN_STD['mean']
        self.std = IMAGENET_MEAN_STD['std']
        
        # Training transform with optional augmentation
        if use_augmentation:
            self.train_transform = T.Compose([
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ])
        else:
            self.train_transform = T.Compose([
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ])
        
        # Validation transform (no augmentation)
        self.val_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ])
        
        self.save_hyperparameters()
    
    def setup(self, stage: str = None):
        """Setup datasets for training/validation"""
        if stage == 'fit' or stage is None:
            self.train_dataset = BagPlacesDataset(
                csv_path=self.train_csv,
                img_per_place=self.img_per_place,
                min_img_per_place=self.min_img_per_place,
                random_sample=self.random_sample,
                transform=self.train_transform
            )
            
            # Validation dataset (optional)
            if self.val_csv is not None:
                self.val_dataset = BagPlacesDataset(
                    csv_path=self.val_csv,
                    img_per_place=self.img_per_place,
                    min_img_per_place=self.min_img_per_place,
                    random_sample=False,  # No random sampling for validation
                    transform=self.val_transform
                )
            else:
                self.val_dataset = None
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers // 2,
            pin_memory=True,
            drop_last=False
        )
    
    def print_stats(self):
        """Print dataset statistics"""
        print("\n" + "="*50)
        print("Training Dataset Statistics")
        print("="*50)
        print(f"  CSV: {self.train_csv}")
        print(f"  Places: {len(self.train_dataset)}")
        print(f"  Images: {self.train_dataset.total_nb_images}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Images per place: {self.img_per_place}")
        print(f"  Image size: {self.image_size}")
        print(f"  Iterations per epoch: {len(self.train_dataset) // self.batch_size}")
        
        if self.val_dataset is not None:
            print(f"\nValidation Dataset:")
            print(f"  CSV: {self.val_csv}")
            print(f"  Places: {len(self.val_dataset)}")
            print(f"  Images: {self.val_dataset.total_nb_images}")
        print("="*50 + "\n")
