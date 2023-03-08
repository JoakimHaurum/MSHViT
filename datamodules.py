from torch.utils.data import DataLoader
from dataloader import SewerMLDataset
import pytorch_lightning as pl


class SewerMLDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=32, workers=4, ann_root="./annotations_sewerml", data_root="./Data", train_transform = None, eval_transform = None):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.ann_root = ann_root
        self.data_root = data_root

        self.train_transform = train_transform
        self.eval_transform = eval_transform

    def prepare_data(self):
        pass

    def setup(self, stage):
        if stage == 'fit':
            self.train_dataset = SewerMLDataset(self.ann_root, self.data_root, split="Train", transform=self.train_transform)
            self.val_dataset = SewerMLDataset(self.ann_root, self.data_root, split="Val", transform=self.eval_transform)
        if stage == 'test':
            self.test_dataset = SewerMLDataset(self.ann_root, self.data_root, split="Test", transform=self.eval_transform)

        self.num_classes = self.train_dataset.num_classes
        self.LabelNames = self.train_dataset.LabelNames

    def train_dataloader(self):
        train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers = self.workers, pin_memory=True)
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers = self.workers, pin_memory=True)
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers = self.workers, pin_memory=True)
        return test_dl

def get_datamodule(dataset, batch_size, workers, ann_root, data_root, train_transform, eval_transform):

    if dataset == "SewerML":
        return SewerMLDataModule(batch_size, workers, ann_root, data_root, train_transform, eval_transform)
    else:
        raise ValueError("There are no Lightning Datamodules for the supplied dataset: {}".format(dataset))
