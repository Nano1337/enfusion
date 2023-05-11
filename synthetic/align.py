import torch
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from unimodals.common_models import Linear, MLP  # noqa
from ensemble import train, test  # noqa
from get_data import get_dataloader  # noqa
from fusions.ensemble_fusions import AdditiveEnsemble  # noqa
from objective_functions.contrast import NCESoftmaxLoss
import numpy as np

import argparse


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cos = torch.nn.CosineSimilarity()

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default="/home/yuncheng/MultiBench/synthetic/SIMPLE_DATA_DIM=3_STD=0.5.pickle", type=str, help="input path of synthetic dataset")
parser.add_argument("--keys", nargs='+', default=['a','b','c','d','e','label'], type=str, help="keys to access data of each modality and label, assuming dataset is structured as a dict")
parser.add_argument("--modalities", nargs='+', default=[0,1], type=int, help="specify the index of modalities in keys")
parser.add_argument("--bs", default=128, type=int)
parser.add_argument("--input-dim", nargs='+', default=30, type=int)
parser.add_argument("--hidden-dim", default=40, type=int)
parser.add_argument("--output-dim", default=100, type=int)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--num-classes", default=2, type=int)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--weight-decay", default=0, type=float)
parser.add_argument("--eval", default=True, type=int)
parser.add_argument("--setting", default='redundancy', type=str)
parser.add_argument("--weight", default=1, type=float)
parser.add_argument("--saved-model", default=None, type=str)
args = parser.parse_args()


def EncourageAlignment(ce_weight=2, align_weight=args.weight, criterion=torch.nn.CrossEntropyLoss()):
    def _actualfunc(pred, truth, args):
        ce_loss = criterion(pred, truth)
        outs = args['outs']
        outs[0] = outs[0].view(-1,).cpu().detach().numpy()
        outs[1] = outs[1].view(-1,).cpu().detach().numpy()
        align_loss = np.dot(outs[0], outs[1])/(np.linalg.norm(outs[0])*np.linalg.norm(outs[1]))
        return align_loss * align_weight + ce_loss * ce_weight
    return _actualfunc

# Load data
traindata, validdata, _, testdata = get_dataloader(path=args.data_path, keys=args.keys, modalities=args.modalities, batch_size=args.bs, num_workers=args.num_workers)

# Specify model
if len(args.input_dim) == 1:
    input_dims = args.input_dim * len(args.modalities)
else:
    input_dims = args.input_dim
encoders = [Linear(input_dim, args.output_dim).to(device) for input_dim in input_dims]
heads = [MLP(args.output_dim, args.hidden_dim, args.num_classes, dropout=False).to(device) for _ in args.modalities]
# encoders = [torch.load(f'/usr0/home/yuncheng/MultiBench/synthetic/model_selection/{args.setting}/{args.setting}_unimodal{i}_encoder.pt') for i in args.modalities]
# heads = [torch.load(f'/usr0/home/yuncheng/MultiBench/synthetic/model_selection/{args.setting}/{args.setting}_unimodal{i}_head.pt') for i in args.modalities]
ensemble = AdditiveEnsemble().to(device)
objective = EncourageAlignment()

# Training
train(encoders, heads, ensemble, traindata, validdata, args.epochs, optimtype=torch.optim.AdamW, lr=args.lr, weight_decay=args.weight_decay, save_model=args.saved_model, modalities=args.modalities, criterion=objective)

# Testing
print("Testing:")
model = torch.load(args.saved_model).to(device)
test(model, testdata, no_robust=True, criterion=torch.nn.CrossEntropyLoss())