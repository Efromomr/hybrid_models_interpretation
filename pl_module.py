#pl_module.py

import pytorch_lightning as pl

import torchmetrics as met
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class PLModelWrap(pl.LightningModule):
    """
    PyTorch Lightning model wrapper.

    Args:
        model (nn.Module): Model to wrap.
        mad_config (MADConfig): MAD configuration.
        metrics (list, optional): List of metrics to use.
    """

    def __init__(self, model, config, metrics: list=['ppl']):
        super().__init__()
        self.model = model
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()
        #self.instantiate_metrics(metrics=metrics)
        self.save_hyperparameters('config')

    """def instantiate_metrics(self, metrics: list) -> None:
        mets = []
        for m in metrics:
            if m=='acc':
                mets.append(
                    Accuracy(
                        num_classes=2,
                        ignore_index=self.mad_config.target_ignore_index
                    )
                )
            elif isinstance(m, met.Metric):
                mets.append(m)
            else:
                raise ValueError(f"invalid metric: {m}, must be one of 'acc' or a torchmetrics metric instance")

        mets = met.MetricCollection(mets)
        self.train_metrics = mets.clone(prefix='train/')
        self.test_metrics = mets.clone(prefix='test/')"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def step(self,
        batch: tuple,
        batch_idx: int
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        targets = batch['label']
        inputs = batch['input_ids']
        outputs = self(inputs)
        loss = self.loss_fn(
            outputs.view(-1, 2),
            targets.view(-1)
        )
        a, outputs = torch.max(outputs, dim=1)
        acc = accuracy_score(outputs.cpu(), targets.cpu())
        acc = torch.tensor(acc, device = torch.device('cuda'))
        return loss, outputs, targets, acc

    def phase_step(self,
        batch: tuple,
        batch_idx: int,
        phase: str='train'
    ) -> tp.Dict[str, tp.Union[torch.Tensor, tp.Any]]:
        loss, outputs, targets, acc = self.step(batch, batch_idx)
        self.log(f'{phase}/Loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #metrics = getattr(self, f'{phase}_metrics')(outputs, targets)
        self.log(f'{phase}/acc', acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, "outputs": outputs, "targets": targets, "accuracy": acc}

    def training_step(self,
        batch: tuple,
        batch_idx: int
    ) -> tp.Dict[str, tp.Union[torch.Tensor, tp.Any]]:
        return self.phase_step(batch, batch_idx, phase='train')

    def validation_step(self,
        batch: tuple,
        batch_idx: int
    ) -> tp.Dict[str, tp.Union[torch.Tensor, tp.Any]]:
        # We currently do not use any validation data, only train/test
        return self.phase_step(batch, batch_idx, phase='test')

    def test_step(self,
        batch: tuple,
        batch_idx: int
    ) -> tp.Dict[str, tp.Union[torch.Tensor, tp.Any]]:
        return self.phase_step(batch, batch_idx, phase='test')

    def configure_optimizers(self) -> tp.Union[torch.optim.Optimizer, tp.Dict[str, tp.Any]]:
        optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config['lr'],
                weight_decay=self.config['weight_decay']
            )
        scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config['epochs'],
                eta_min=self.config['min_lr'],
                last_epoch=-1
            )
        return {'optimizer': optimizer, 'scheduler': scheduler}