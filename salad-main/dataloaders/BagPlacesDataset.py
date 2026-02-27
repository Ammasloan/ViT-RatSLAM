"""
Custom PyTorch Dataset for loading RatSLAM bag-extracted places.

Follows the same interface as GSVCitiesDataset for compatibility with VPRModel.
"""

import pandas as pd
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

default_transform = T.Compose([
    T.Resize((322, 322), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class BagPlacesDataset(Dataset):
    """
    Dataset for loading images extracted from ROS bags, organized by place.
    
    Expected directory structure:
        images_root/
        ├── place_0000/
        │   ├── img_000.jpg
        │   ├── img_001.jpg
        │   └── ...
        ├── place_0001/
        │   └── ...
        
    CSV format (places.csv):
        place_id,img_path,timestamp,target_timestamp
        0,images/place_0000/img_000.jpg,10.501,10.5
        0,images/place_0000/img_001.jpg,10.534,10.5
        ...
    
    Args:
        csv_path: Path to places.csv
        images_root: Root directory containing images (parent of 'images/' folder)
        img_per_place: Number of images to sample per place
        min_img_per_place: Minimum images required per place (places with fewer are dropped)
        random_sample: Whether to randomly sample images from each place
        transform: Image transforms to apply
    """
    
    def __init__(self,
                 csv_path: str,
                 images_root: str = None,
                 img_per_place: int = 4,
                 min_img_per_place: int = 4,
                 random_sample: bool = True,
                 transform=default_transform):
        super().__init__()
        
        self.csv_path = Path(csv_path)
        
        # If images_root not specified, assume it's the parent of csv file
        if images_root is None:
            self.images_root = self.csv_path.parent
        else:
            self.images_root = Path(images_root)
        
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.random_sample = random_sample
        self.transform = transform
        
        # Load and filter dataframe
        self.dataframe = self._load_dataframe()
        
        # Get unique place IDs
        self.place_ids = self.dataframe['place_id'].unique()
        self.total_nb_images = len(self.dataframe)
        
        print(f"[BagPlacesDataset] Loaded {len(self.place_ids)} places, {self.total_nb_images} images")
    
    def _load_dataframe(self) -> pd.DataFrame:
        """Load CSV and filter places with insufficient images"""
        df = pd.read_csv(self.csv_path)
        
        # Count images per place
        place_counts = df.groupby('place_id').size()
        valid_places = place_counts[place_counts >= self.min_img_per_place].index
        
        # Filter to valid places only
        df = df[df['place_id'].isin(valid_places)]
        
        dropped = len(place_counts) - len(valid_places)
        if dropped > 0:
            print(f"[BagPlacesDataset] Dropped {dropped} places with < {self.min_img_per_place} images")
        
        return df
    
    def __len__(self) -> int:
        return len(self.place_ids)
    
    def __getitem__(self, index: int):
        place_id = self.place_ids[index]
        
        # Get all images for this place
        place_df = self.dataframe[self.dataframe['place_id'] == place_id]
        
        # Sample images
        if self.random_sample and len(place_df) > self.img_per_place:
            place_df = place_df.sample(n=self.img_per_place)
        else:
            place_df = place_df.head(self.img_per_place)
        
        # Load images
        imgs = []
        for _, row in place_df.iterrows():
            img_path = self.images_root / row['img_path']
            img = self._load_image(img_path)
            
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        
        # Stack into tensor [K, C, H, W]
        imgs_tensor = torch.stack(imgs)
        labels_tensor = torch.tensor(place_id).repeat(len(imgs))
        
        return imgs_tensor, labels_tensor
    
    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        try:
            return Image.open(path).convert('RGB')
        except Exception as e:
            print(f"[BagPlacesDataset] Failed to load {path}: {e}")
            # Return black image as fallback
            return Image.new('RGB', (322, 322))


if __name__ == '__main__':
    # Quick test
    import sys
    if len(sys.argv) < 2:
        print("Usage: python BagPlacesDataset.py <csv_path>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    ds = BagPlacesDataset(csv_path)
    
    print(f"\nDataset size: {len(ds)} places")
    print(f"Total images: {ds.total_nb_images}")
    
    if len(ds) > 0:
        imgs, labels = ds[0]
        print(f"Sample batch shape: {imgs.shape}")
        print(f"Sample labels: {labels}")
