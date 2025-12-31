from transformers import BertConfig,BertModel
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
import argparse

import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from .config import *

torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Convmodule(nn.Module):
    """Convolution Module.
    The convolution module is made up of "num_cb" conv+pooling blocks.
    """
    def __init__(self,in_channels,out_channels,stride = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels = in_channels,out_channels = 32,kernel_size = 3,padding = 1 , stride = stride)
        self.pool1 = nn.MaxPool1d(kernel_size = 4)

        self.conv2 = nn.Conv1d(in_channels = 32 ,out_channels = 64,kernel_size = 5,padding = 2 , stride = stride)
        self.pool2 = nn.MaxPool1d(kernel_size = 4)

        self.conv3 = nn.Conv1d(in_channels = 64 ,out_channels = 96,kernel_size = 5,padding = 2 , stride = stride)
        self.pool3 = nn.MaxPool1d(kernel_size = 2)

        self.conv4 = nn.Conv1d(in_channels = 96,out_channels = 128,kernel_size = 3,padding = 1 , stride = stride) 
        self.pool4 = nn.MaxPool1d(kernel_size = 2)

        self.conv5 = nn.Conv1d(in_channels = 128 ,out_channels = out_channels,kernel_size = 3,padding = 1 , stride = stride)
        self.pool5 = nn.MaxPool1d(kernel_size = 2)
        self.relu=nn.ReLU()
    def forward(self,x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.pool4(self.relu(self.conv4(x)))
        x = self.pool5(self.relu(self.conv5(x)))
        return x
    
class Multitaskmodule(nn.Module):
    """Multi-task prediction module.
    This module is mainly made up with linear layer.
    """
    def __init__(self,SEQUENCE_DIM,TF_DIM,NUM_SIGNALS):
        super().__init__()
        self.linear = nn.Linear(SEQUENCE_DIM + TF_DIM, NUM_SIGNALS)

    def forward(self,x):
        x = F.relu(self.linear(x))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class EpiGePT(pl.LightningModule):
    """Initialize layers to build EpiGePT model.
        Args:
            word_num: size of the vocabulary of the transformer module.
            sequence_dim: dimension of the token embedding from the output of the Convolution module.
            tf_dim: dimension of the TF embedding.
            batch_size: batch size for training.
    """
    def __init__(self, word_num,sequence_dim,tf_dim,batch_size):
        super().__init__()
        self.word_num = word_num
        self.sequence_dim = sequence_dim
        self.tf_dim = tf_dim
        self.batch_size = batch_size
        self.dna_chunk_len = 128_000
        self.conv_out_len = self._compute_conv_out_len(self.dna_chunk_len)
        self.convmodule = Convmodule(CHANNEL_SIZE,SEQUENCE_DIM)
        self.config_encoder = BertConfig(vocab_size=word_num, hidden_size=SEQUENCE_DIM + TF_DIM,
                                            num_hidden_layers=NUM_LAYER,
                                            num_attention_heads=NUM_HEAD,
                                            intermediate_size=1024,
                                            output_hidden_states=False,
                                            output_attentions=False,
                                            max_position_embeddings=1000)#shape (bs, inp_len, inp_len)

        self.transformermodule = BertModel(config=self.config_encoder)
        # Linear layer for multi-task prediction
        self.multitaskmodule = Multitaskmodule(SEQUENCE_DIM,TF_DIM,NUM_SIGNALS)

    def forward(self,batch_inputs_seq,batch_inputs_tf):
        x = self.convmodule(batch_inputs_seq)
        x = x.transpose(1,2)
        x = torch.cat([x ,batch_inputs_tf],dim=2)
        x = self.transformermodule(inputs_embeds=x)
        output = self.multitaskmodule(x[0])
        return output

    @staticmethod
    def _compute_conv_out_len(length: int) -> int:
        l = length
        for k in [4, 4, 2, 2, 2]:
            l = l // k
        return l

    def _quarter_and_pool_dna(self, dna_seq: torch.Tensor):
        """Split 524288 bp DNA into 4 x 128 kb chunks; no pre-pooling."""
        segment_len = dna_seq.shape[1] // 4
        start = (segment_len - self.dna_chunk_len) // 2
        quarters = []
        for chunk in dna_seq.chunk(4, dim=1):
            quarters.append(chunk[:, start:start + self.dna_chunk_len, :])
        dna_stack = torch.stack(quarters, dim=1).reshape(-1, self.dna_chunk_len, 4)
        return dna_stack.permute(0, 2, 1)  # (B*4, 4, 128000)

    def _quarter_and_pool_tracks(self, tensor: torch.Tensor, target_len: int = 1000) -> torch.Tensor:
        """Split track tensor into quarters, fixed pool/crop to 1000 bins."""
        # tensor: (B, L, C); expected L in {4096 (motifs), 8192 (labels)}
        quarters = []
        for chunk in tensor.chunk(4, dim=1):
            pooled = chunk
            if chunk.shape[1] >= 2000:  # labels quarter = 2048
                pooled = F.avg_pool1d(pooled.permute(0, 2, 1), kernel_size=2, stride=2).permute(0, 2, 1)  # -> 1024
            current = pooled.shape[1]
            if current >= target_len:
                start = (current - target_len) // 2
                pooled = pooled[:, start:start + target_len, :]
            else:
                pad_left = (target_len - current) // 2
                pad_right = target_len - current - pad_left
                pooled = F.pad(pooled, (0, 0, pad_left, pad_right))
            quarters.append(pooled)
        stacked = torch.stack(quarters, dim=1).reshape(-1, target_len, tensor.shape[2])
        return stacked  # (B*4, target_len, C)

    def _prepare_batch(self, batch):
        dna_seq, tf_expr, tf_binding, labels, exp_mask = batch
        dna_seq = dna_seq.float()
        tf_binding = tf_binding.float()
        labels = labels.float()
        exp_mask = exp_mask.float()

        dna_tokens = self._quarter_and_pool_dna(dna_seq)

        tf_binding_proc = self._quarter_and_pool_tracks(tf_binding, target_len=1000)
        labels_proc = self._quarter_and_pool_tracks(labels, target_len=1000)

        tf_expr_expanded = tf_expr.repeat_interleave(4, dim=0)
        tf_feats = tf_binding_proc * tf_expr_expanded.unsqueeze(1)

        # Downsample TF features and labels to the conv output length
        tf_feats = F.adaptive_avg_pool1d(tf_feats.permute(0, 2, 1), output_size=self.conv_out_len).permute(0, 2, 1)
        labels_proc = F.adaptive_avg_pool1d(labels_proc.permute(0, 2, 1), output_size=self.conv_out_len).permute(0, 2, 1)

        exp_mask = exp_mask.repeat_interleave(4, dim=0)
        exp_mask_seq = exp_mask.expand(-1, -1, self.conv_out_len)

        return dna_tokens, tf_feats, labels_proc, exp_mask_seq

    @staticmethod
    def _masked_mse(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = (preds - targets) ** 2
        loss = loss * mask.permute(0, 2, 1)
        denom = mask.sum()
        return loss.sum() / (denom + 1e-8)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr = LEARNING_RATE)
    
    def training_step(self,batch,batch_idx):
        dna_tokens, tf_feats, labels, exp_mask = self._prepare_batch(batch)
        preds = self.forward(dna_tokens, tf_feats)
        loss = self._masked_mse(preds, labels, exp_mask)
        return loss

    def validation_step(self,batch,batch_idx):
        dna_tokens, tf_feats, labels, exp_mask = self._prepare_batch(batch)
        preds = self.forward(dna_tokens, tf_feats)
        loss = self._masked_mse(preds, labels, exp_mask)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        dna_tokens, tf_feats, labels, exp_mask = self._prepare_batch(batch)
        preds = self.forward(dna_tokens, tf_feats)
        loss = self._masked_mse(preds, labels, exp_mask)
        self.log('test_loss', loss)
        return loss

    def setup(self, stage):
        return
    def test_dataloader(self):
        return None 
    