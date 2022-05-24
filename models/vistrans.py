from pathlib import Path
from copy import deepcopy
from argparse import ArgumentParser
import sys
sys.path.append("../")
import torch
from torch import nn, optim
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torchmetrics import Accuracy
from pytorch_lightning.loggers import TensorBoardLogger,WandbLogger

from data.eurosat_datamodule import EurosatDataModule
from timm.models import vision_transformer


class VisTrans(LightningModule):

    def __init__(self, img_size,patch_size,in_chans, num_classes,norm_layer,act_layer):
        super().__init__()
        self.model = vision_transformer.VisionTransformer(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                num_classes=num_classes,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

    def forward(self, x):
        logits = self.model(x)
        return logits

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc', acc, prog_bar=True)
        return loss

    def shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(torch.argmax(logits, dim=1), y)
        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters())
        max_epochs = self.trainer.max_epochs
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6*max_epochs), int(0.8*max_epochs)])
        return [optimizer], [scheduler]


if __name__ == '__main__':
    pl.seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--img_size',type=int,default=64)
    parser.add_argument("--patch_size",type=int,default=4)
    parser.add_argument('--num_classes',type=int,default=10)
    parser.add_argument('--is_4_channels',action="store_true")
    args = parser.parse_args()

    datamodule = EurosatDataModule(args.data_dir,args.is_4_channels)

    datamodule.setup()
    train_loader = datamodule.train_dataloader()

    for idx,data in enumerate(train_loader):
        print(data[0].shape,data[1].shape)
        print(type(data[0]),type(data[1]))
        break

    model = VisTrans(
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_chans=4 if args.is_4_channels else 3,
            num_classes=args.num_classes,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU)

    experiment_name = "VisTrans" + ("_4_channels" if args.is_4_channels else "_3_channels")
    #logger = TensorBoardLogger(save_dir=str(Path.cwd() / 'logs' / 'eurosat'), name=experiment_name)
    wandb_logger = WandbLogger(project="transformers_eurosat",
                                name=experiment_name)
    trainer = Trainer(gpus=args.gpus, logger=wandb_logger, checkpoint_callback=True, max_epochs=100, weights_summary='full')
    trainer.fit(model, datamodule=datamodule)
