import os
import torch
import torchvision
import math
import pytorch_lightning as pl

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from change_detection_pytorch.datasets import UCMerced, build_transform, BigearthnetDataModule
from change_detection_pytorch.encoders import (vit_encoders, swin_transformer_encoders, prithvi_encoders,
                                               clay_encoders, dinov2_encoders, dofa_encoders, sd_cvit_encoders)
from change_detection_pytorch.encoders._utils import load_pretrained, adjust_state_dict_prefix
from utils import get_band_indices

from torchmetrics import Accuracy, AveragePrecision
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback


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
                  lr, scheduler, checkpoint_path, only_head, warmup_steps, eta_min,
                  warmup_start_lr, weight_decay, mixup, prefix='backbone', optimizer='adamw',
                  enable_sample=False, multilabel=False, bands=['B04', 'B03', 'B02']):
        super().__init__()
        self.in_features = in_features
        self.lr = lr
        self.scheduler = scheduler
        self.only_head = only_head
        self.multilabel = multilabel
        self.backbone_name = backbone_name
        self.bands = bands
        self.enable_sample=enable_sample
        self.optimizer = optimizer
        
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
        # for name, param in self.encoder.named_parameters(): 
        #     if "channel_embed" in name:
        #         param.requires_grad = False
            # if param.requires_grad:
            #     print(name)
        self.mixup = v2.MixUp(num_classes=num_classes) if mixup else None

    def load_encoder(self, encoder_name='ibot-B', encoder_weights='imagenet'):
    
        if 'swin' in encoder_name.lower():
            if 'satlas_ms' in encoder_weights.lower():
                import satlaspretrain_models

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
            if encoder_weights == 'random':
                return encoder
            else:
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
            if 'sat' in encoder_name.lower():
                Encoder = dinov2_encoders[encoder_name]["encoder"]
                params = dinov2_encoders[encoder_name]["params"]
                params.update(classification=True)
                encoder = Encoder(**params).eval()
                # path = '/nfs/ap/mnt/frtn/rs-results/dinov2_sat/SSLhuge_satellite.pth'
                # encoder = SSLAE(pretrained=path, huge=True, classification=True).eval()
            else:
                encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').eval()

        elif 'cvit-pretrained' in encoder_name.lower():
            Encoder = sd_cvit_encoders[encoder_name]["encoder"]
            params = sd_cvit_encoders[encoder_name]["params"]
            params.update(return_feats=False)
            params.update(enable_sample=self.enable_sample)
            encoder = Encoder(**params)
            
            # Load weights
            settings = sd_cvit_encoders[encoder_name]["pretrained_settings"][encoder_weights]
            state_dict = torch.load(settings["url"], map_location=torch.device('cpu'))['teacher']
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = encoder.load_state_dict(state_dict, strict=False)
            print(msg)
        
        elif 'cvit' in encoder_name.lower():
            encoder = torch.hub.load('insitro/ChannelViT', 'so2sat_channelvit_small_p8_with_hcs_random_split_supervised', pretrained=True)

        elif 'prithvi' in encoder_name.lower():
            Encoder = prithvi_encoders[encoder_name]["encoder"]
            params = prithvi_encoders[encoder_name]["params"]
            params.update(for_cls=True)
            encoder = Encoder(**params)
            settings = prithvi_encoders[encoder_name]["pretrained_settings"][encoder_weights]
            state_dict = torch.load(settings["url"], map_location=torch.device('cpu'))
            del state_dict['pos_embed']
            del state_dict['decoder_pos_embed']

            msg = encoder.load_state_dict(state_dict, strict=False)
            print(msg)

        elif 'clay' in encoder_name.lower():
            Encoder = clay_encoders[encoder_name]["encoder"]
            params = clay_encoders[encoder_name]["params"]
            params.update(for_cls=True)
            encoder = Encoder(**params)

        elif 'dofa' in encoder_name.lower():
            Encoder = dofa_encoders[encoder_name]["encoder"]
            params = dofa_encoders[encoder_name]["params"]
            params.update(for_cls=True)
            params.update(global_pool=False)
            encoder = Encoder(**params)

            settings = dofa_encoders[encoder_name]["pretrained_settings"][encoder_weights]
            state_dict = torch.load(settings["url"], map_location=torch.device('cpu'))
            msg = encoder.load_state_dict(state_dict, strict=False)
            print(msg)

        return encoder

    def forward(self, x, metadata=None):
        # with torch.no_grad():
        if 'satlas' in self.backbone_weights:
            if 'ms' in self.backbone_weights:
                feats = self.encoder(x)[-1]
                feats = self.norm_layer(feats)
                feats = self.global_average_pooling(feats)
                feats = torch.flatten(feats, 1)
            else:
                return self.encoder(x)
        elif 'cvit-pretrained' in self.backbone_name.lower():
            feats = self.encoder(x, channel_idxs=get_band_indices(self.bands))
        elif 'cvit' in self.backbone_name.lower():
            channels = torch.tensor([self.channels]).cuda()
            feats = self.encoder(x, extra_tokens={"channels":channels})
        elif 'ms' in self.backbone_weights:
            feats = self.encoder(x)[-1]
            feats = self.norm_layer(feats)
            feats = self.global_average_pooling(feats)
            feats = torch.flatten(feats, 1)
        elif 'clay' in self.backbone_name.lower() or 'dofa' in self.backbone_name.lower():
            feats = self.encoder(x, metadata)
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
        if 'ben' in args.dataset_name.lower():
            x, y, metadata = batch
        else:
            x, y = batch
        if mixup:
            x, y = self.mixup(x, y)
        
        if 'clay' in self.backbone_name.lower():
            logits = self(x, metadata)
        elif 'dofa' in self.backbone_name.lower():
            logits = self(x, metadata[0]['waves'])
        else:
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
            if 'satlas' in self.backbone_weights and 'ms' not in self.backbone_weights:
                parameters = self.encoder.head.parameters()
            else:
                parameters = self.classifier.parameters()
        else:
            parameters = self.parameters()

        if self.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(parameters, eps=1e-8, betas=(0.9, 0.999), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(parameters, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer}")

        if self.scheduler == 'cosine':
            scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=self.warmup_steps, total_epochs=max_epochs, eta_min=self.eta_min, warmup_start_lr=self.warmup_start_lr)
        elif self.scheduler == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6*max_epochs), int(0.8*max_epochs)])
        elif self.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.1*max_epochs), gamma=0.1)
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler}")

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
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--warmup_steps', type=int, default=20)
    parser.add_argument('--eta_min', type=float, default=1.0e-5)
    parser.add_argument('--warmup_start_lr', type=float, default=1.0e-7)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--splits_dir', type=str, default='')
    parser.add_argument('--fill_zeros', action="store_true")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--enable_sample', action='store_true')
    parser.add_argument("--bands", nargs="+", type=str, default=['B04', 'B03', 'B02']) # ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12', 'VH', 'VH','VV', 'VV']

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    image_size =  (args.image_size // 14) * 14 if 'dino' in args.backbone_name else args.image_size

    if 'ben' in args.dataset_name.lower():
        datamodule = BigearthnetDataModule(
        data_dir=args.base_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        splits_dir=args.splits_dir,
        fill_zeros = args.fill_zeros,
        img_size=image_size,
        bands=args.bands
        )
        datamodule.setup()

        dataloader_train = datamodule.train_dataloader()
        dataloader_val = datamodule.val_dataloader()
        num_classes= datamodule.num_classes
        multilabel=True
        print(f'BEN num of classes {num_classes}')
    else:
        tr_transform = build_transform(split='train', image_size=args.image_size, mixup=args.mixup)
        val_transform = build_transform(split='val', image_size=args.image_size)

        train_dataset = UCMerced(root=args.root, base_dir=args.base_dir, split='train', 
                                transform=tr_transform, dataset_name=args.dataset_name, image_size=args.image_size)
        val_dataset = UCMerced(root=args.root, base_dir=args.base_dir, split='val',
                                transform=val_transform, dataset_name=args.dataset_name, image_size=image_size)
        dataloader_train = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        dataloader_val = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        num_classes= args.num_classes
        multilabel=False

    print(args.encoder_weights)
    model = Classifier(backbone_name=args.backbone_name, backbone_weights=args.encoder_weights,
                       in_features=args.in_features, num_classes=num_classes,
                         lr=args.lr, scheduler=args.scheduler, checkpoint_path=args.checkpoint_path, 
                         only_head=args.only_head, warmup_steps=args.warmup_steps,
                         eta_min=args.eta_min, warmup_start_lr=args.warmup_start_lr, 
                         weight_decay=args.weight_decay, enable_sample=args.enable_sample,
                           mixup=args.mixup, multilabel=multilabel, bands=args.bands, optimizer=args.optimizer)
    
    wandb_logger = WandbLogger(log_model=False, project="classification",
        name=args.experiment_name,config=vars(args))

    checkpoints_dir = f'/nfs/h100/raid/rs/checkpoints_anna/checkpoints/classification/{args.experiment_name}'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=checkpoints_dir,
    #     filename='{epoch:02d}',
    #     save_top_k=-1,
    #     every_n_epochs=25
    # )
    best_model_checkpoint = ModelCheckpoint(
        dirpath=checkpoints_dir,
        monitor='val/acc',         
        save_top_k=1,               
        mode='max',                
        filename='best-model',
        verbose=True,
        save_last=True
    )
    trainer = pl.Trainer(devices=args.device, logger=wandb_logger, max_epochs=args.epoch, num_nodes=args.num_nodes,
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         log_every_n_steps=None, callbacks=[best_model_checkpoint, LearningRateLogger()])
    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)