from argparse import ArgumentParser
from torch.utils.data import DataLoader
import os

import torch
import pytorch_lightning as pl
from change_detection_pytorch.datasets import UCMerced, build_transform
from change_detection_pytorch.encoders import vit_encoders, swin_transformer_encoders
from change_detection_pytorch.encoders._utils import load_pretrained, adjust_state_dict_prefix

from torchmetrics import Accuracy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import torchvision

class Classifier(pl.LightningModule):

    def __init__(self, backbone_name, backbone_weights, in_features, num_classes, lr, sched, checkpoint_path):
        super().__init__()
        self.in_features = in_features
        self.lr = lr
        self.sched = sched

        if 'satlas' in backbone_weights:
            checkpoint = torch.load(checkpoint_path)
            new_state_dict = adjust_state_dict_prefix(checkpoint, 'backbone', 'backbone.', 0)
            self.encoder = torchvision.models.swin_v2_b()
            self.encoder.load_state_dict(new_state_dict)
            self.encoder.head = torch.nn.Linear(in_features, num_classes)
        else:
            self.encoder = self.load_encoder(backbone_name, backbone_weights)
            self.classifier = torch.nn.Linear(in_features, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.backbone_weights = backbone_weights
    
    def load_encoder(self, encoder_name='ibot-B', encoder_weights='imagenet'):
    
        if 'swin' in encoder_name.lower():
            Encoder = swin_transformer_encoders[encoder_name]["encoder"]
            params = swin_transformer_encoders[encoder_name]["params"]
            gap = False if 'satlas' in encoder_weights else True
            params.update(for_cls=True, gap=gap)

            encoder = Encoder(**params)
            settings = swin_transformer_encoders[encoder_name]["pretrained_settings"][encoder_weights]
            checkmoint_model = load_pretrained(encoder, settings["url"], 'cpu')
            msg = encoder.load_state_dict(checkmoint_model, strict=False)

        elif 'ibot' in encoder_name.lower():
            Encoder = vit_encoders[encoder_name]["encoder"]
            params = vit_encoders[encoder_name]["params"]
            params.update(for_cls=True)
            encoder = Encoder(**params)
            settings = vit_encoders[encoder_name]["pretrained_settings"][encoder_weights]
            if 'imagenet' in settings["url"]:
                state_dict = torch.load(settings["url"], map_location=torch.device('cpu'))['state_dict']
            else:
                state_dict = torch.load(settings["url"], map_location=torch.device('cpu'))['teacher']
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = encoder.load_state_dict(state_dict, strict=False)
        print(msg)
        return encoder

    def forward(self, x):
        # with torch.no_grad():
        if 'satlas' in self.backbone_weights:
            return self.encoder(x)
        feats = self.encoder(x)
        logits = self.classifier(feats)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        max_epochs = self.trainer.max_epochs
        if self.sched == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-6)
            monitor = "val/acc"
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6*max_epochs), int(0.8*max_epochs)])
            return [optimizer], [scheduler]


if __name__ == '__main__':
    pl.seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--base_dir', type=str, default='')
    parser.add_argument('--backbone_name', type=str, default='ibot-B')
    parser.add_argument('--encoder_weights', type=str, default='imagenet')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset_name', type=str, default='')
    parser.add_argument('--in_features', type=int, default=768)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--experiment_name', type=str, default='')
    parser.add_argument('--mixup', action="store_true")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--sched', type=str, default='')
    parser.add_argument('--checkpoint_path', type=str, default='')

    args = parser.parse_args()

    tr_transform = build_transform(split='train', mixup=args.mixup)
    val_transform = build_transform(split='val')
    train_dataset = UCMerced(root=args.root, base_dir=args.base_dir, split='train', 
                             transform=tr_transform, dataset_name=args.dataset_name)
    val_dataset = UCMerced(root=args.root, base_dir=args.base_dir, split='val',
                            transform=val_transform, dataset_name=args.dataset_name)
    dataloader_train = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=8)
    dataloader_val = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=8)
    
    model = Classifier(backbone_name=args.backbone_name, backbone_weights=args.encoder_weights,
                       in_features=args.in_features, num_classes=args.num_classes, lr=args.lr, sched=args.sched, checkpoint_path=args.checkpoint_path)
    wandb_logger = WandbLogger(log_model=False, project="classification",
        name=args.experiment_name,config=vars(args))

    checkpoints_dir = f'./checkpoints/{args.experiment_name}'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename='{epoch:02d}',
        save_top_k=-1,
        every_n_epochs=5
    )
    trainer = pl.Trainer(devices=args.device, logger=wandb_logger, max_epochs=args.epoch, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)