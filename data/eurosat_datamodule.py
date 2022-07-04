
from torch.utils.data import DataLoader
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule

from data.eurosat_dataset import EurosatDataset,EurosatDataset4Channel


transform_4_channel = T.Normalize((3.0,2.0,0.0,0.0),
                                  (88.0,103.0,129.0,147.0))


class EurosatDataModule(LightningDataModule):

    def __init__(self, data_dir,is_4_channels):
        super().__init__()
        self.data_dir = data_dir
        self.is_4_channels = is_4_channels

    @property
    def num_classes(self):
        return 10

    def setup(self, stage=None):
        if not self.is_4_channels:
            self.train_dataset = EurosatDataset(self.data_dir, split='train', transform=T.ToTensor())
            self.val_dataset = EurosatDataset(self.data_dir, split='val', transform=T.ToTensor())
        else:
            self.train_dataset = EurosatDataset4Channel(self.data_dir, split='train', transform=transform_4_channel)
            self.val_dataset = EurosatDataset4Channel(self.data_dir, split='val', transform=transform_4_channel)
    
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=256,
            shuffle=True,
            num_workers=32,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=32,
            drop_last=True,
            pin_memory=True
        )
