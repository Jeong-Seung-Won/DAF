import random
from argparse import Namespace
from typing import List, Tuple, Optional, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.amp import autocast, GradScaler
from torch import einsum
from torch.nn.utils import weight_norm
from torch.utils.data import (
    DataLoader,
    Dataset,
    ConcatDataset,
    random_split
)
from einops import rearrange, repeat

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random_seed = 1

torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
torch.set_printoptions(precision = 8)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print(device)

TRAIN_META = ""
VAL_META = ""
TEST_META  = ""
STATS_PKL   = ""
SEQ_LEN        = 512
PRED_LEN       = 90
INPUT_DIM      = 6
OUTPUT_DIM     = 6
HIDDEN_DIM     = 512
N_LAYERS       = 2
BATCH_SIZE     = 512
LR             = 1e-4
EPOCHS         = 50
STRIDE         = 512 + 90
EARLY_PATIENCE = 5                     
NUM_WORKERS    = 0
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

with open(STATS_PKL, "rb") as f:
    _stats = pickle.load(f)               

GLOBAL_MEAN = torch.tensor(_stats['means'], dtype=torch.float32).unsqueeze(0)
GLOBAL_STD  = torch.tensor(_stats['stds'] , dtype=torch.float32).unsqueeze(0).clamp(min=1e-8)

class TimeSeriesDataset(Dataset):
    def __init__(self, ts: torch.Tensor, seq_len: int, pred_len: int, stride: int = 1):
        super().__init__()
        self.x, self.L, self.H, self.str = ts, seq_len, pred_len, stride
        self.n = (len(ts) - seq_len - pred_len) // stride + 1

    def __len__(self): return self.n

    def __getitem__(self, idx):
        s = idx * self.str
        return ( self.x[s : s+self.L].contiguous(),
                 self.x[s+self.L : s+self.L+self.H].contiguous() )
    

def pt_to_tensor(fp: str) -> torch.Tensor:
    ts = torch.load(fp, map_location = 'cpu').float()           
    return ts

def pt_to_dataset(fp: str,
                  mean: torch.Tensor = GLOBAL_MEAN,
                  std : torch.Tensor = GLOBAL_STD
                 ) -> TimeSeriesDataset:
    
    ts = pt_to_tensor(fp).float()              
    ts_norm = (ts - mean) / std                
    del ts                                     
    return TimeSeriesDataset(ts_norm, SEQ_LEN, PRED_LEN, STRIDE)

def _build_loader(file_csv: str) -> DataLoader:
    file_list = pd.read_csv(file_csv)["file_path"].tolist()
    ds_list   = [pt_to_dataset(fp) for fp in file_list]          
    full_ds   = ConcatDataset(ds_list)

    return DataLoader(full_ds,
                      batch_size  = BATCH_SIZE,
                      shuffle     = False,
                      drop_last   = True,
                      pin_memory  = True,
                      num_workers = NUM_WORKERS)

def build_train_loader() -> DataLoader:
    return _build_loader(TRAIN_META)

def build_val_loader() -> DataLoader:
    return _build_loader(VAL_META)

def build_test_loader() -> DataLoader:
    return _build_loader(TEST_META)

def train_epoch(model, loader, optimizer, scaler, criterion, criterion_mae):
    model.train()
    loss_sum = mae_sum = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=DEVICE, dtype=torch.float16 if DEVICE=='cuda' else torch.bfloat16):
            y_hat = model(x)
            loss = criterion(y_hat, y)
            mae = criterion_mae(y_hat, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item()
        mae_sum += mae.item()
    n = len(loader)
    return loss_sum / n, mae_sum / n


def eval_epoch(model, loader, criterion, criterion_mae):
    model.eval()
    loss_sum = mae_sum = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with autocast(device_type=DEVICE, dtype=torch.float16 if DEVICE=='cuda' else torch.bfloat16):
            y_hat = model(x)
            loss = criterion(y_hat, y)
            mae = criterion_mae(y_hat, y)
        loss_sum += loss.item()
        mae_sum += mae.item()
    n = len(loader)
    return loss_sum / n, mae_sum / n
