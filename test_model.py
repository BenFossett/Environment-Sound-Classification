#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
import pickle
import os.path

from dataset import UrbanSound8KDataset
from utils import compute_accuracy, compute_per_class_accuracy
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
    "--mode",
    default="LMC",
    type=str,
    help="LMC, MC, or MLMC"
)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def main(args):
    criterion = nn.CrossEntropyLoss()

    if args.mode in ["LMC", "MC", "MLMC"]:
        test_loader = torch.utils.data.DataLoader(
            UrbanSound8KDataset('UrbanSound8K_test.pkl', mode=args.mode),
            batch_size=32, shuffle=False,
            num_workers=8, pin_memory=True)
        if args.mode in ["LMC", "MC"]:
            model = CNN(height=85, width=41, channels=1, class_count=10, dropout=0.5, mode=args.mode)
        else:
            model = CNN(height=145, width=41, channels=1, class_count=10, dropout=0.5, mode=args.mode)
        checkpoint = torch.load("checkpoint" + args.mode +".pkl", map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])
        validator = Validator(model, test_loader, criterion, DEVICE)
        validator.validate(args.mode)
    elif args.mode == "TSCNN":
        if os.path.isfile("LMCscores.pkl") and os.path.isfile("MCscores.pkl"):
            LMCscores = pickle.load(open("LMCscores.pkl", 'rb'))
            MCscores = pickle.load(open("MCscores.pkl", 'rb'))
            TSCNN(LMCscores, MCscores, DEVICE)
        else:
            print("Run both LMCNet and MCNet to obtain scores before running TSCNN")
    else:
        print("Input one of LMC, MC, MLMC, or TSCNN as the mode argument")

class Validator:
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device

    def validate(self, mode):
        results = {"preds": [], "labels": []}
        file_results = {}
        total_loss = 0
        self.model.eval()

        with torch.no_grad():
            for i, (input, target, filename) in enumerate(self.test_loader):
                batch = input.to(self.device)
                labels = target.to(self.device)
                logits = self.model.forward(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                preds = logits.cpu().numpy()
                labels_array = labels.cpu().numpy()
                batch_size = len(filename)

                # Collect together the predictions for segments corresponding
                # to their respective files in order to calculate validation
                # accuracy.
                for j in range(0, batch_size):
                    file = filename[j]
                    label = labels_array[j]
                    pred = preds[j]
                    if file not in file_results:
                        file_results[file] = {"preds": [], "labels": []}
                    file_results[file]["preds"].append(pred)
                    file_results[file]["labels"].append(label)

        # Save logits to be used for late fusion.
        f = open(mode + "scores.pkl", "wb")
        pickle.dump(file_results, f)
        f.close()

        # Compute final per-file predictions.
        for f in file_results:
            file_pred = np.argmax(np.mean(file_results[f]["preds"], axis=0))
            file_label = np.round(np.mean(file_results[f]["labels"])).astype(int)
            results["preds"].append(file_pred)
            results["labels"].append(file_label)

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_loss = total_loss / len(self.test_loader)

        classes = ["air conditioner", "car horn", "children playing",
        "dog bark", "drilling", "engine idling", "gun shot", "jack hammer",
        "siren", "street music"]
        class_accuracies = compute_per_class_accuracy(np.array(results["labels"]), np.array(results["preds"]))

        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")

        for i in range(0, 10):
            print(f"class: {classes[i]}, class accuracy: {class_accuracies[i] * 100:2.2f}")

        print(f"average class accuracy: {np.average(class_accuracies) * 100:2.2f}")


def TSCNN(LMCscores, MCscores, device):
    results = {"preds": [], "labels": []}
    softmax = nn.Softmax(dim=-1)

    for file in LMCscores:
        LMC_file = LMCscores[file]
        MC_file = MCscores[file]

        # Obtain LMC logits and label for each file.
        LMC_pred = torch.from_numpy(np.mean(LMC_file["preds"], axis=0)).to(device)
        LMC_logits = softmax(LMC_pred)
        LMC_label = np.round(np.mean(LMC_file["labels"])).astype(int)

        # Obtain MC logits and label for each file.
        MC_pred = torch.from_numpy(np.mean(MC_file["preds"], axis=0)).to(device)
        MC_logits = softmax(MC_pred)
        MC_label = np.round(np.mean(MC_file["labels"])).astype(int)

        # Average logits and calculate final prediction.
        logits = (LMC_logits + MC_logits) / 2
        pred = logits.argmax().cpu().numpy()
        results["preds"].append(pred)
        results["labels"].append(LMC_label)

    accuracy = compute_accuracy(
        np.array(results["labels"]), np.array(results["preds"])
    )

    classes = ["air conditioner", "car horn", "children playing",
    "dog bark", "drilling", "engine idling", "gun shot", "jack hammer",
    "siren", "street music"]
    class_accuracies = compute_per_class_accuracy(np.array(results["labels"]), np.array(results["preds"]))

    for i in range(0, 10):
        print(f"class: {classes[i]}, class accuracy: {class_accuracies[i] * 100:2.2f}")

    print(f"average class accuracy: {np.average(class_accuracies) * 100:2.2f}")

if __name__ == "__main__":
    main(parser.parse_args())
