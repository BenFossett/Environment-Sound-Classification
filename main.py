#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

from dataset import UrbanSound8KDataset
from model import CNN
from trainer import Trainer

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import argparse
from pathlib import Path

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a simple CNN on CIFAR-10",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=1e-3, type=float, help="Learning rate")
parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=20,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for"
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)
parser.add_argument(
    "--checkpoint-path",
    default=Path("checkpointLMC.pkl"),
    type=Path,
    help="Provide a file to store checkpoints of the model parameters during training."
)
parser.add_argument(
    "--checkpoint-frequency",
    type=int,
    default=10,
    help="Save a checkpoint every N epochs"
)
parser.add_argument(
    "--mode",
    default="LMC",
    type=str,
    help="LMC, MC, or MLMC"
)
parser.add_argument(
    "--dropout",
    default=0.5,
    type=float,
)
parser.add_argument(
    "--weight-decay",
    default=1e-3,
    type=float,
)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def main(args):
    transform = transforms.ToTensor()
    args.dataset_root.mkdir(parents=True, exist_ok=True)

    train_loader = torch.utils.data.DataLoader(
        UrbanSound8KDataset('UrbanSound8K_train.pkl', mode=args.mode),
        batch_size=32, shuffle=True,
        num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        UrbanSound8KDataset('UrbanSound8K_test.pkl', mode=args.mode),
        batch_size=32, shuffle=False,
        num_workers=8, pin_memory=True)

    if args.mode == "LMC" or args.mode == "MC":
        model = CNN(height=85, width=41, channels=1, class_count=10, dropout=args.dropout, mode=args.mode)
    elif args.mode == "MLMC":
        model = CNN(height=145, width=41, channels=1, class_count=10, dropout=args.dropout, mode=args.mode)
    elif args.mode == "TSCNN":
        print("Use file late_fusion.py to run TSCNN with trained LMCNet and MCNet")

    criterion = nn.CrossEntropyLoss()

    #optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    trainer = Trainer(
        model, train_loader, test_loader, criterion, optimizer, summary_writer,
        DEVICE, args.checkpoint_path, args.checkpoint_frequency
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    summary_writer.close()

def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = f'CNN_bn_mode={args.mode}_dropout={args.dropout}_bs={args.batch_size}_lr={args.learning_rate}_run_'
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

if __name__ == "__main__":
    main(parser.parse_args())
