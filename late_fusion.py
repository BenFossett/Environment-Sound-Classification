#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

from dataset import UrbanSound8KDataset
from model import CNN

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
parser.add_argument(
    "--checkpoint-LMC",
    default=Path("checkpointLMC.pkl"),
    type=Path,
    help="Provide a file to store checkpoints of the model parameters during training."
)
parser.add_argument(
    "--checkpoint-MC",
    default=Path("checkpointMC.pkl"),
    type=Path,
    help="Provide a file to store checkpoints of the model parameters during training."
)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def main(args):
    loader_LMC = torch.utils.data.DataLoader(
    UrbanSound8KDataset('UrbanSound8K_test.pkl', mode='LMC'),
    batch_size=32, shuffle=False,
    num_workers=8, pin_memory=True)
    model_LMC = CNN(height=85, width=41, channels=1, class_count=10, dropout=0.5, mode='LMC')
    checkpoint_LMC = torch.load(args.checkpoint_LMC, map_location = DEVICE)
    model_LMC.load_state_dict(checkpoint_LMC)
    print("LMC Loaded")

    loader_MC = torch.utils.data.DataLoader(
    UrbanSound8KDataset('UrbanSound8K_test.pkl', mode='MC'),
    batch_size=32, shuffle=False,
    num_workers=8, pin_memory=True)
    model_MC = CNN(height=85, width=41, channels=1, class_count=10, dropout=0.5, mode='MC')
    checkpoint_MC = torch.load(args.checkpoint_MC, map_location = DEVICE)
    model_MC.load_state_dict(checkpoint_MC)
    print("MC Loaded")

    criterion = nn.CrossEntropyLoss()
    validate_TSCNN(model_LMC, model_MC, loader_LMC, loader_MC, criterion, DEVICE)

def validate_TSCNN(model_LMC, model_MC, loader_LMC, loader_MC, criterion, device):
    print("Performing late fusion")
    results = {"preds": [], "labels": []}
    dict = {}
    total_loss = 0
    model_LMC.eval()
    model_MC.eval()
    loader_MC = iter(loader_MC)

    with torch.no_grad():
        for i, (input_LMC, target, filename) in enumerate(loader_LMC):
            print(i)
            (input_MC, _, _) = next(loader_MC)
            print("MC")
            batch_LMC = input_LMC.to(device)
            batch_MC = input_MC.to(device)
            labels = target.to(device)
            logits_LMC = nn.Softmax(model_LMC.forward(batch_LMC))
            logits_MC = nn.Softmax(model_MC.forward(batch_MC))
            print("logits")
            logits = (logits_array_LMC + logits_array_MC) / 2
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            print("loss")

            logits_array = logits.cpu().numpy()
            labels_array = labels.cpu().numpy()
            batch_size = len(filename)
            print("loop")
            for j in range(0, batch_size):
                file = filename[j]
                if file in dict:
                    dict[file]["sum"] += logits_array[j]
                    dict[file]["n_segments"] += 1
                else:
                    dict[file] = {}
                    dict[file]["label"] = labels_array[j]
                    dict[file]["sum"] = logits_array[j]
                    dict[file]["n_segments"] = 1
                    dict[file]["average"] = 0

        for f in dict:
            sum = dict[f]["sum"]
            n_segments = dict[f]["n_segments"]
            dict[f]["average"] = sum / n_segments

        file_labels = np.hstack([dict[k]["label"] for k, l in dict.items()])
        file_logits = np.vstack([dict[k]["average"] for k, a in dict.items()])
        labels = torch.from_numpy(file_labels).to(self.device)
        logits = torch.from_numpy(file_logits).to(self.device)
        preds = np.argmax(file_logits, axis=-1)
        results["preds"].extend(list(preds))
        results["labels"].extend(list(labels.cpu().numpy()))

    accuracy = compute_accuracy(
        np.array(results["labels"]), np.array(results["preds"])
    )
    average_loss = total_loss / len(loader_LMC)

    classes = ["air conditioner", "car horn", "children playing",
    "dog bark", "drilling", "engine idling", "gun shot", "jack hammer",
    "siren", "street music"]
    class_accuracies = compute_per_class_accuracy(np.array(results["labels"]), np.array(results["preds"]))

    print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")

    for i in range(0, 10):
        print(f"class: {classes[i]}, class accuracy: {class_accuracies[i] * 100:2.2f}")

    print(f"average class accuracy: {np.average(class_accuracies) * 100:2.2f}")

def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)

def compute_per_class_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)

    accuracies = []
    for i in range(0, 10):
        class_labels = labels == i
        correct_preds = (preds == labels)
        accuracies.append(float((class_labels & correct_preds).sum() / (class_labels).sum()))
    return accuracies

if __name__ == "__main__":
    main(parser.parse_args())
