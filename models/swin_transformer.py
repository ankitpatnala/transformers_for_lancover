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
from timm.models import swin_transformer
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.strategies import DeepSpeedStrategy
import deepspeed
from fairscale.nn import checkpoint_wrapper,auto_wrap,wrap

class SwinTrans(LightningModule):

    def __init__(self,
            img_size,
            patch_size,
            window_size,
            in_chans,
            num_classes,
            norm_layer):
        super().__init__()
        self.unsharded_model = swin_transformer.SwinTransformer(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=768,
                in_chans=in_chans,
                num_classes=num_classes,
                window_size=window_size,
                norm_layer=nn.LayerNorm,
                ape=True)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
    
    def configure_sharded_model(self):
        #layer = self.unsharded_model.patch_embed
        #self.layer_1 = layer
        #self.layer_2 = torch.nn.Sequential(*self.unsharded_model.layers)
        #self.sharded_model = auto_wrap(checkpoint_wrapper(self.unsharded_model))
        self.sharded_model = self.unsharded_model
        

    def forward(self, x):
        #logits = self.sharded_model(x)
        logits = deepspeed.checkpointing.checkpoint(self.sharded_model,x)
        #x = deepspeed.checkpointing.checkpoint(self.layer_1,x)
        #x = self.layer_2(x)
        #print(1,x.shape)
        #x = self.unsharded_model.norm(x)
        #x = self.unsharded_model.avgpool(x)
        #print(2,x.shape)
        #logits = self.unsharded_model.head(x)
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
        x.requires_grad = True
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(torch.argmax(logits, dim=1), y)
        return loss, acc

    def configure_optimizers(self):
        #optimizer = optim.Adam(self.model.parameters())
        optimizer = optim.AdamW(self.sharded_model.parameters(),lr=0.00025,weight_decay=0.05)
        max_epochs = self.trainer.max_epochs
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6*max_epochs), int(0.8*max_epochs)])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,max_epochs)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    pl.seed_everything(42)

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--img_size',type=int,default=64)
    parser.add_argument("--patch_size",type=int,default=4)
    parser.add_argument("--window_size",type=int,default=4)
    parser.add_argument('--num_classes',type=int,default=10)
    parser.add_argument('--is_4_channels',action="store_true")
    args = parser.parse_args()

    datamodule = EurosatDataModule(args.data_dir,args.is_4_channels)

    model = SwinTrans(
            img_size=args.img_size,
            patch_size=args.patch_size,
            window_size=args.window_size,
            in_chans=4 if args.is_4_channels else 3,
            num_classes=args.num_classes,
            norm_layer=nn.LayerNorm)

    experiment_name = "SwinTrans" + ("_4_channels" if args.is_4_channels else "_3_channels")
    #logger = TensorBoardLogger(save_dir=str(Path.cwd() / 'logs' / 'eurosat'), name=experiment_name)
    wandb_logger = WandbLogger(project="transformers_eurosat")
    trainer = Trainer(logger=wandb_logger,
            checkpoint_callback=True,
            max_epochs=100,
            weights_summary='full',
            gpus=args.gpus,
            accelerator='gpu',
            precision=16,
            devices=4,
            strategy=DeepSpeedStrategy(stage=3,
                                        offload_parameters=False,
                                        offload_optimizer=True,
                                        remote_device='cpu',
                                        reduce_scatter=False) ,
            #strategy='fsdp',
            plugins=[SLURMEnvironment()])
    trainer.fit(model, datamodule=datamodule)
