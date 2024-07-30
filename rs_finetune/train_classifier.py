from argparse import ArgumentParser
from torch.utils.data import DataLoader
import os
from torchvision.transforms import v2

import torch
import pytorch_lightning as pl
from change_detection_pytorch.datasets import UCMerced, build_transform, BigearthnetDataModule
from change_detection_pytorch.encoders import vit_encoders, swin_transformer_encoders
from change_detection_pytorch.encoders._utils import load_pretrained, adjust_state_dict_prefix

from torchmetrics import Accuracy, AveragePrecision
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

import satlaspretrain_models

import torchvision
import math
torch.set_float32_matmul_precision('medium')

class WarmupCosineAnnealingLR(torch.optim.lr_scheduler.CosineAnnealingLR):
    def __init__(self, optimizer, warmup_epochs, total_epochs, warmup_start_lr=0, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, T_max=total_epochs, eta_min=eta_min, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            cos_val = 0.5 * (1.0 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)))
            return [max(base_lr * (self.eta_min + (1 - self.eta_min) * cos_val), self.eta_min) for base_lr in self.base_lrs]

class LearningRateLogger(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # Get the current learning rate from the optimizer
        lr = float(trainer.optimizers[0].param_groups[0]['lr'])
        # Log the learning rate using your chosen logging framework
        trainer.logger.experiment.log({"learning_rate": lr})


class Classifier(pl.LightningModule):

    def __init__(self, backbone_name, backbone_weights, in_features, num_classes,
                  lr, sched, checkpoint_path, only_head, warmup_steps, eta_min, 
                  warmup_start_lr, weight_decay, mixup, prefix='backbone', multilabel=False):
        super().__init__()
        self.in_features = in_features
        self.lr = lr
        self.sched = sched
        self.only_head = only_head
        self.multilabel = multilabel
        self.backbone_name = backbone_name

        if 'satlas' in backbone_weights and 'ms' not in backbone_weights:
            checkpoint = torch.load(checkpoint_path)
            if prefix == 'encoder':
                new_state_dict = adjust_state_dict_prefix(checkpoint['state_dict'], prefix, f'{prefix}.', 0)
                self.encoder = torchvision.models.swin_v2_b()
                self.encoder.head = torch.nn.Linear(in_features, num_classes)
                self.encoder.load_state_dict(new_state_dict)
            else:
                new_state_dict = adjust_state_dict_prefix(checkpoint, prefix, f'{prefix}.', 0)
                self.encoder = torchvision.models.swin_v2_b()
                self.encoder.load_state_dict(new_state_dict)
                self.encoder.head = torch.nn.Linear(in_features, num_classes)
        else:
            self.encoder = self.load_encoder(backbone_name, backbone_weights)
            self.classifier = torch.nn.Linear(in_features, num_classes)
            if 'ms' in backbone_weights:
                self.global_average_pooling = torch.nn.AdaptiveAvgPool2d(1)
                self.norm_layer = torch.nn.LayerNorm([1024, 4, 4]) 
        if multilabel:
            self.criterion = torch.nn.MultiLabelSoftMarginLoss()
            self.accuracy = AveragePrecision(num_classes=num_classes, average='micro', task='binary')
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
            self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.backbone_weights = backbone_weights
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        self.warmup_start_lr = warmup_start_lr
        self.weight_decay = weight_decay
        
        self.mixup = v2.MixUp(num_classes=num_classes) if mixup else None

    def load_encoder(self, encoder_name='ibot-B', encoder_weights='imagenet'):
    
        if 'swin' in encoder_name.lower():
            if 'satlas_ms' in encoder_weights.lower():
                weights_manager = satlaspretrain_models.Weights()
                encoder = weights_manager.get_pretrained_model(model_identifier="Sentinel2_SwinB_SI_MS")
            else:
                Encoder = swin_transformer_encoders[encoder_name]["encoder"]
                params = swin_transformer_encoders[encoder_name]["params"]
                gap = False if 'satlas' in encoder_weights else True
                params.update(for_cls=True, gap=gap, window_size=8)

                encoder = Encoder(**params)
                settings = swin_transformer_encoders[encoder_name]["pretrained_settings"][encoder_weights]
                checkmoint_model = load_pretrained(encoder, settings["url"], 'cpu')
                msg = encoder.load_state_dict(checkmoint_model, strict=False)
                print(msg)

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
        elif 'dino' in encoder_name.lower():
            encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        elif 'cvit' in encoder_name.lower():
            encoder = torch.hub.load('insitro/ChannelViT', 'so2sat_channelvit_small_p8_with_hcs_random_split_supervised', pretrained=True)

        return encoder

    def forward(self, x, channels = [0, 1, 2]):
        # with torch.no_grad():
        if 'satlas' in self.backbone_weights and 'ms' not in self.backbone_weights:
            return self.encoder(x)
        elif 'cvit' in self.backbone_name.lower():
            channels = torch.tensor([channels]).cuda()
            feats = self.encoder(x, extra_tokens={"channels":channels})
        elif 'ms' in self.backbone_weights:
            feats = self.encoder(x)[-1]
            feats = self.norm_layer(feats)
            feats = self.global_average_pooling(feats)
            feats = torch.flatten(feats, 1)
        else:
            feats = self.encoder(x)
        logits = self.classifier(feats)
        return logits

    def training_step(self, batch, batch_idx):
        mixup=True if self.mixup else False

        loss, acc = self.shared_step(batch, mixup=mixup)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc', acc, prog_bar=True)
        return loss

    def shared_step(self, batch, mixup=False):
        x, y = batch
        if mixup:
            x, y = self.mixup(x, y)
        logits = self(x)
        loss = self.criterion(logits, y)
        if mixup:
            y = torch.argmax(y, dim=1)
        if self.multilabel:
            # probabilities = torch.sigmoid(logits)
            # predictions = (probabilities >= 0.5).float()
            acc = self.accuracy(logits, y.int())
        else:
            acc = self.accuracy(torch.argmax(logits, dim=1), y)
        return loss, acc

    def configure_optimizers(self):
        max_epochs = self.trainer.max_epochs
        if self.only_head:
            if 'satlas' in self.backbone_weights:
                optimizer = torch.optim.Adam(self.encoder.head.parameters())
            else:
                optimizer = torch.optim.Adam(self.classifier.parameters())
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6*max_epochs), int(0.8*max_epochs)])

        else:
            optimizer = torch.optim.AdamW(self.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=self.lr,
                                      weight_decay=self.weight_decay)
            scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=self.warmup_steps, 
                                                total_epochs=max_epochs, eta_min=self.eta_min, warmup_start_lr=self.warmup_start_lr)
        
        return [optimizer], [scheduler]
        

if __name__ == '__main__':

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
    parser.add_argument('--only_head', action="store_true")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--sched', type=str, default='')
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--warmup_steps', type=int, default=20)
    parser.add_argument('--eta_min', type=float, default=1.0e-5)
    parser.add_argument('--warmup_start_lr', type=float, default=1.0e-7)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--splits_dir', type=str, default='')
    parser.add_argument('--fill_zeros', action="store_true")
    parser.add_argument('--seed', type=int, default=42)


    args = parser.parse_args()
    pl.seed_everything(args.seed)

    image_size = 252 if 'dino' in args.backbone_name else 256
    if 'ben' in args.dataset_name.lower():
        datamodule = BigearthnetDataModule(
        data_dir=args.base_dir,
        batch_size=args.batch_size,
        num_workers=24,
        splits_dir=args.splits_dir,
        fill_zeros = args.fill_zeros
        )
        datamodule.setup()

        dataloader_train = datamodule.train_dataloader()
        dataloader_val = datamodule.val_dataloader()
        num_classes= datamodule.num_classes
        multilabel=True
        print(f'BEN num of classes{num_classes}')
    else:
        tr_transform = build_transform(split='train', image_size=image_size, mixup=args.mixup)
        val_transform = build_transform(split='val', image_size=image_size)

        train_dataset = UCMerced(root=args.root, base_dir=args.base_dir, split='train', 
                                transform=tr_transform, dataset_name=args.dataset_name, image_size=image_size)
        val_dataset = UCMerced(root=args.root, base_dir=args.base_dir, split='val',
                                transform=val_transform, dataset_name=args.dataset_name, image_size=image_size)
        dataloader_train = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        dataloader_val = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        num_classes= args.num_classes
        multilabel=False

    print(args.encoder_weights)
    model = Classifier(backbone_name=args.backbone_name, backbone_weights=args.encoder_weights,
                       in_features=args.in_features, num_classes=num_classes,
                         lr=args.lr, sched=args.sched, checkpoint_path=args.checkpoint_path, 
                         only_head=args.only_head, warmup_steps=args.warmup_steps,
                         eta_min=args.eta_min, warmup_start_lr=args.warmup_start_lr, weight_decay=args.weight_decay,
                           mixup=args.mixup,  multilabel=multilabel)
    
    wandb_logger = WandbLogger(log_model=False, project="classification",
        name=args.experiment_name,config=vars(args))

    checkpoints_dir = f'./checkpoints/classification/{args.experiment_name}'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename='{epoch:02d}',
        save_top_k=-1,
        every_n_epochs=20
    )

    trainer = pl.Trainer(devices=args.device, logger=wandb_logger, max_epochs=args.epoch, log_every_n_steps= None, callbacks=[checkpoint_callback, LearningRateLogger()])
    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)