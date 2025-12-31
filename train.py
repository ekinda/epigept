import argparse
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model.config import *
from model import EpiGePT, dataset

torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(hparams):
    pl.seed_everything(hparams.seed)

    train_seq_end = hparams.num_sequences - max(0, hparams.val_sequences)
    train_dataset = dataset.EpigeptCorgiDataset(
        dna_path=hparams.dna_path,
        motif_path=hparams.motif_path,
        tf_expression_path=hparams.tf_expression_path,
        experiment_mask_path=hparams.mask_path,
        tissue_dir=hparams.tissue_dir,
        sequence_ids=list(range(train_seq_end)),
        tissue_ids=hparams.tissue_ids,
        output_channels=NUM_SIGNALS,
    )

    val_dataset = None
    if hparams.val_sequences > 0:
        val_dataset = dataset.EpigeptCorgiDataset(
            dna_path=hparams.dna_path,
            motif_path=hparams.motif_path,
            tf_expression_path=hparams.tf_expression_path,
            experiment_mask_path=hparams.mask_path,
            tissue_dir=hparams.tissue_dir,
            sequence_ids=list(range(hparams.num_sequences - hparams.val_sequences, hparams.num_sequences)),
            tissue_ids=hparams.tissue_ids,
            output_channels=NUM_SIGNALS,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        shuffle=True,
        num_workers=hparams.num_workers,
        drop_last=False,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=hparams.batch_size,
            shuffle=False,
            num_workers=hparams.num_workers,
            drop_last=False,
        )

    model = EpiGePT.EpiGePT(WORD_NUM, SEQUENCE_DIM, TF_DIM, hparams.batch_size)

    callbacks = []
    if val_loader is not None:
        callbacks.append(EarlyStopping(monitor='val_loss', mode='min', patience=5))

    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        logger=pl_loggers.TensorBoardLogger(save_dir='logs', name='TensorBoard', version=hparams.run_name),
        callbacks=callbacks,
        default_root_dir=os.getcwd(),
        devices=hparams.gpus if torch.cuda.is_available() else 1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")

    parser.add_argument("--dna_path", type=str, required=True, help="Path to corgi DNA one-hot npy")
    parser.add_argument("--motif_path", type=str, required=True, help="Path to TF binding affinity npy")
    parser.add_argument("--tf_expression_path", type=str, required=True, help="Path to TF expression npy")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to experiment mask npy")
    parser.add_argument("--tissue_dir", type=str, required=True, help="Directory containing tissue_*.npy")
    parser.add_argument("--num_sequences", type=int, required=True, help="Number of genomic regions to use")
    parser.add_argument("--tissue_ids", type=int, nargs="+", required=True, help="Tissue IDs to include")
    parser.add_argument("--val_sequences", type=int, default=0, help="Number of sequences reserved for validation from the end")

    parser.add_argument("--run_name", type=str, default="epigept_corgi", help="Run name for logging")

    args = parser.parse_args()
    main(args)
