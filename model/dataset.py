import math
import os
import random

import numpy as np
import torch
torch.set_default_tensor_type(torch.FloatTensor)
from torch.utils.data import Dataset, Sampler
from .config import LEARNING_RATE, BATCH_SIZE, INPUT_LEN, TARGET_LEN, SEQUENCE_DIM, TF_DIM, NUM_LAYER, NUM_HEAD, NUM_SIGNALS, CHANNEL_SIZE, WORD_NUM

class EpigeptCorgiDataset(Dataset):
    """Corgi-format loader adapted for EpiGePT inputs.

    Returns raw corgi-style tensors:
      dna_seq:      (L_dna=524288, 4)
      tf_expr:      (n_tf,)
      tf_binding:   (L_bins=8192, n_tf)
      padded_label: (L_bins=8192, output_channels)
      exp_mask:     (output_channels, 1)
    Downstream logic in the Lightning module handles quarter splits and pooling.
    """
    def __init__(
        self,
        dna_path: str,
        motif_path: str,
        tf_expression_path: str,
        experiment_mask_path: str,
        tissue_dir: str,
        sequence_ids: list,
        tissue_ids: list,
        output_channels: int = NUM_SIGNALS,
    ):
        super().__init__()
        self.sequence_ids = list(sequence_ids)
        self.tissue_ids = list(tissue_ids)
        self.output_channels = output_channels

        self.dna_sequences = np.load(dna_path, mmap_mode='r')
        self.tf_binding = np.memmap(
            motif_path, 
            dtype=np.float16, 
            mode='r',
            shape=(14395, 4096, 711)
        )
        self.tf_expression = torch.from_numpy(np.load(tf_expression_path)).float()
        
        ## DEBUG ##
        self.tf_expression = self.tf_expression[:, :712]
        ###########
        # Until I supply the real tf exp matrix subsetted to the 711 epigept TFs.
        # Also i think epigept needs 712 numbers so that
        # 712 + 256 (sequence dim) = 968 (transformer input dim) which needs to be a multiple of 8 (heads)????

        self.experiment_mask = np.load(experiment_mask_path)
        self.tissue_dir = tissue_dir

        if len(self.sequence_ids) == 0:
            raise ValueError("sequence_ids must be non-empty")
        if len(self.tissue_ids) == 0:
            raise ValueError("tissue_ids must be non-empty")
        if self.dna_sequences.shape[0] != self.tf_binding.shape[0]:
            raise ValueError("dna and motif arrays must share the first dimension")
        if self.experiment_mask.shape[1] != self.output_channels:
            raise ValueError("experiment_mask second dimension must equal output_channels")

        self._tissue_arrays = {}

    def __len__(self):
        return len(self.sequence_ids) * len(self.tissue_ids)

    def _load_tissue_array(self, tissue_id: int) -> np.memmap:
        if tissue_id not in self._tissue_arrays:
            path = os.path.join(self.tissue_dir, f"tissue_{tissue_id}.npy")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing tissue file: {path}")
            self._tissue_arrays[tissue_id] = np.load(path, mmap_mode='r')
        return self._tissue_arrays[tissue_id]

    def __getitem__(self, index: int):
        seq_offset = index // len(self.tissue_ids)
        tissue_offset = index % len(self.tissue_ids)
        seq_id = self.sequence_ids[seq_offset]
        tissue_id = self.tissue_ids[tissue_offset]

        dna_seq = torch.tensor(self.dna_sequences[seq_id], dtype=torch.float32)              # (524288, 4)
        tf_binding = torch.tensor(self.tf_binding[seq_id], dtype=torch.float32)              # (4096, n_tf)
        
        ## DEBUG: Pad with zero column to match expected 712 TFs ##
        tf_binding = torch.cat(
            [tf_binding, torch.zeros((tf_binding.shape[0], 1), dtype=torch.float32)],
            dim=1
        )  # (4096, 712)
        ###########################################################
        
        tf_expr = self.tf_expression[tissue_id]

        tissue_array = self._load_tissue_array(tissue_id)[seq_id]                # (8192, n_available)
        mask_vec = self.experiment_mask[tissue_id]                               # (C,)
        available_idx = np.where(mask_vec == 1)[0]
        padded_label = torch.zeros((tissue_array.shape[0], self.output_channels), dtype=torch.float32)
        padded_label[:, available_idx] = torch.tensor(tissue_array, dtype=torch.float32)
        exp_mask = torch.from_numpy(mask_vec.astype(np.float32)).unsqueeze(-1)   # (C, 1)

        return dna_seq, tf_expr, tf_binding, padded_label, exp_mask