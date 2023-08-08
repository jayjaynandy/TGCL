import argparse

from loader import MoleculeDataset_candidate, list_collate_fn
from torch.utils.data import DataLoader
import torch_geometric
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

import os
from itertools import combinations

if torch_geometric.__version__ >= '2.0.0':
    # Ignore UserWarning of torch_geometric (>=2.0.0)
    import warnings
    warnings.filterwarnings("ignore")

div_factor = 4
TAU = 10
ALPHA = 0.95

class GNNWrapper(nn.Module):
    def __init__(self, gnn, percept_gnn, emb_dim, args):
        super(GNNWrapper, self).__init__()
        self.gnn = gnn
        self.percept_gnn = percept_gnn
        self.pool = global_mean_pool
        self.score_func = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(), 
            nn.Linear(emb_dim, emb_dim), nn.ReLU(), 
            nn.Linear(emb_dim, 1)
        )
        
        if args.edit_learn:
            # combination of index
            self.edit_idx1 = []
            self.edit_idx2 = []
            for i in range(args.num_candidates-2):
                self.edit_idx1.extend([i]*(args.num_candidates-i-2))
                for j in range(i+1, args.num_candidates-1):
                    self.edit_idx2.append(j)        

            self.edit_coeff = args.edit_coeff

        if args.margin_learn:
            self.margin = args.margin
            self.margin_coeff = args.margin_coeff


    def forward(self, x, edge_index, edge_attr, batch):
        rep = self.gnn(x, edge_index, edge_attr)
        feats = self.percept_gnn(x, edge_index, edge_attr)
        graph_rep = self.pool(rep, batch)

        scores = self.score_func(graph_rep).squeeze(-1)

        return scores, graph_rep, feats

    def GD_loss(self, scores, feats, device):
        feats = feats.div(div_factor)
        perceptual_dist = (feats[:,1:] - feats[:,:1]).norm(dim=-1)
        normalized_perceptual_dist = perceptual_dist / perceptual_dist.sum(axis=1).unsqueeze(-1)
        perceptual_dist = torch.cat(
            [10 * torch.ones((perceptual_dist.size(0), 1), dtype=torch.float32).to(device), 
             1 / perceptual_dist
            ], axis=1
        )
        normalized_perceptual_dist = torch.cat(
            [torch.ones((normalized_perceptual_dist.size(0), 1), dtype=torch.float32).to(device), 
             normalized_perceptual_dist
            ], axis=1
        )
        
        answer = torch.zeros((scores.size(0),), dtype=torch.long).to(device)
        pred = scores.argmax(dim=-1)
        correct = (pred == answer).sum().cpu().item()
        total = scores.size(0)

        GD_loss = nn.KLDivLoss()(F.log_softmax(scores/TAU, dim=1),
                                 F.softmax(perceptual_dist/TAU, dim=1)) * (ALPHA * TAU * TAU)
        answer = torch.zeros_like(scores).to(device)
        answer[:, 0] = 1.0
        GD_loss += F.binary_cross_entropy_with_logits(
            scores.flatten(), answer.flatten(), weight=normalized_perceptual_dist.flatten()) * (1. - ALPHA)

        GD_loss += F.cross_entropy(scores, answer) * (1. - ALPHA)

        return GD_loss, correct, total

    def edit_loss(self, graph_rep, feats):
        rep_diff = (graph_rep[:,1:] - graph_rep[:,:1]).norm(dim=-1)
        feats = feats.div(div_factor)
        perceptual_dist = (feats[:,1:] - feats[:,:1]).norm(dim=-1)
        dist_norm = rep_diff / perceptual_dist
        
        edit_loss = F.mse_loss(dist_norm[:, self.edit_idx1], dist_norm[:, self.edit_idx2])

        return self.edit_coeff * edit_loss


    def margin_loss(self, graph_rep, feats, device, args):
        num_candidates = args.num_candidates

        # Remove the strongly perturbed candidate.
        if args.add_strong_pert:
            graph_rep = graph_rep[:, :-1]
            feats = feats[:, :-1].div(div_factor)
            num_candidates -= 1

        batch_size = graph_rep.size(0)

        # compute positive sim
        anchor = graph_rep[:,:1]
        pos = graph_rep[:,1:]
        pos_sim = (anchor - pos).norm(dim=-1) # pos_sim : (B, num_candidates-1)
        pos_sim = pos_sim.repeat_interleave(batch_size-1, dim=0).view(batch_size, batch_size-1, -1) # pos_sim : (B, B-1, num_candidates-1)
        
        # compute positive margin
        anchor = feats[:,:1]
        pos = feats[:,1:]
        pos_mar = (anchor - pos).norm(dim=-1) # pos_sim : (B, num_candidates-1)
        pos_mar = pos_mar.repeat_interleave(batch_size-1, dim=0).view(batch_size, batch_size-1, -1)

        # Compute negative sim
        ori_rep = graph_rep[:,0]

        interleave = ori_rep.repeat_interleave(batch_size, dim=0).view(batch_size, batch_size, -1)
        repeat = ori_rep.repeat(batch_size, 1).view(batch_size, batch_size, -1)

        sim_matrix = (interleave - repeat).norm(dim=-1) # sim_matrix : (B, B)

        # Compute negative margin
        ori_rep = feats[:,0]

        interleave = ori_rep.repeat_interleave(batch_size, dim=0).view(batch_size, batch_size, -1)
        repeat = ori_rep.repeat(batch_size, 1).view(batch_size, batch_size, -1)

        mar_matrix = (interleave - repeat).norm(dim=-1) # mar_matrix : (B, B)

        
        # Remove diagonal of ''sim_matrix''
        diag_idx = torch.eye(batch_size, dtype=torch.bool)
        sim_matrix = sim_matrix[~diag_idx].view(batch_size, batch_size-1) # sim_matrix: (B, B-1)
        
        # Remove diagonal of ''mar_matrix''
        diag_idx = torch.eye(batch_size, dtype=torch.bool)
        mar_matrix = mar_matrix[~diag_idx].view(batch_size, batch_size-1) # mar_matrix: (B, B-1)

        # neg_sim: (B, B-1, num_candidates-1)
        neg_sim = sim_matrix.repeat_interleave(num_candidates-1, dim=-1).view(batch_size, batch_size-1, num_candidates-1)
        
        # neg_mar: (B, B-1, num_candidates-1)
        neg_mar = mar_matrix.repeat_interleave(num_candidates-1, dim=-1).view(batch_size, batch_size-1, num_candidates-1)

        # Loss computation
        margin_loss = torch.maximum(
            torch.tensor(0).to(device), 
            torch.maximum(torch.tensor(self.margin).to(device), (neg_mar - pos_mar).abs()) + 
            pos_sim - neg_sim
        ).mean()

        return self.margin_coeff * margin_loss



def train(args, loader, model, optimizer, device):
    model.train()

    train_loss_accum = 0
    GD_loss_accum, edit_loss_accum, edit_loss_v2_accum, margin_loss_accum = 0, 0, 0, 0
    GD_total_correct, GD_total_num = 0, 0

    pbar = range(len(loader))
    pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)

    loader_iterator = iter(loader)

    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for idx in pbar:
        optimizer.zero_grad()
        batch = next(loader_iterator)
        batch = batch.to(device)
        GD_loss, edit_loss, edit_loss_v2, margin_loss = 0, 0, 0, 0

        scores, graph_rep, feats = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        scores = scores.view(-1, args.num_candidates)
        graph_rep = graph_rep.view(-1, args.num_candidates, args.emb_dim)
        feats = global_mean_pool(feats, batch.batch).view(
            -1, 
            args.num_candidates, 
            (args.num_layer + 1 - int(args.percept_JK[-1])) * args.emb_dim
        )
        
        # Graph discrimination
        GD_loss, GD_correct, GD_total = model.GD_loss(scores, feats, device)
        GD_total_correct += GD_correct
        GD_total_num += GD_total

        GD_loss_accum += float(GD_loss.detach().cpu().item())

        # Graph edit distance-based discrepancy learning
        if args.edit_learn:            
            edit_loss = model.edit_loss(graph_rep, feats)
            edit_loss_accum += float(edit_loss.detach().cpu().item())

        # Discrepancy learning with negative graphs
        if args.margin_learn:
            margin_loss = model.margin_loss(graph_rep, feats, device, args)
            margin_loss_accum += float(margin_loss.detach().cpu().item())
        
        loss = GD_loss + edit_loss + margin_loss
        
        loss.backward()
        optimizer.step()
        
        train_loss_accum += float(loss.detach().cpu().item())

        pbar.set_description(
            f"loss: {(train_loss_accum / (idx + 1)):.4f}, "
            f"gd_loss: {(GD_loss_accum / (idx + 1)):.4f}, "
            f"edit_loss: {(edit_loss_accum / (idx + 1)):.4f}, "
            f"margin_clf_loss: {(margin_loss_accum / (idx + 1)):.4f}"
        )

    return GD_total_correct/GD_total_num


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--percept_JK', type=str, default="list2",
                        help='how the node features across layers are combined. list0, list1, list2, list3, list4, list5')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--output_dir', type = str, default = f'ckpts/TGCL_tau{TAU}/', help='directoy name for saving checkpoints')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed")
    parser.add_argument('--num_workers', type=int, default = 16, help='number of workers for dataset loading')

    # D-SLA arguments
    parser.add_argument('--mask_strength', type=float, default = 0.8)
    parser.add_argument('--edge_pert_strength', type=float, default = 0.2)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--num_candidates', type=int, default=4)
    parser.add_argument('--add_strong_pert', action='store_true')

    # Graph edit distance-based discrepancy learning
    parser.add_argument('--edit_learn', action='store_true')
    parser.add_argument('--edit_coeff', type=float, default=0.7)

    # Discrepancy learning with negative graphss
    parser.add_argument('--margin_learn', action='store_true')
    parser.add_argument('--margin', type=float, default=5.0)
    parser.add_argument('--margin_coeff', type=float, default=0.5)


    args = parser.parse_args()


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # adding a candidates for strongly perturbed graph
    if args.add_strong_pert:
        args.num_candidates += 1

    #set up dataset
    dataset = MoleculeDataset_candidate("../../dataset/" + args.dataset, dataset=args.dataset, \
                                        edge_pert_strength=args.edge_pert_strength, \
                                        mask_strength=args.mask_strength, \
                                        k=args.k, num_candidates=args.num_candidates, \
                                        increase_pert=args.edit_learn, \
                                        return_strong_pert=args.add_strong_pert)
    print(dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, \
                        collate_fn=list_collate_fn)
    
    #set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio,
              gnn_type = args.gnn_type)
    
    percept_gnn = GNN(args.num_layer, args.emb_dim, JK=args.percept_JK, 
                      drop_ratio=args.dropout_ratio, gnn_type='gin')
    checkpoint = torch.load('../../D-SLA/transferLearning_MoleculeNet_PPI/chem/ckpts/DSLA/100.pth')
    percept_gnn.load_state_dict(checkpoint)

    model = GNNWrapper(gnn, percept_gnn, args.emb_dim, args)
    model.to(device)

    #set up optimizer
    for param in model.percept_gnn.parameters():
        param.requires_grad = False
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.decay
    )
    print(optimizer)
     
    for epoch in range(1, args.epochs+1):
        GD_acc = train(args, loader, model, optimizer, device)
        print("Epoch:{}, GD Acc:{}\n".format(epoch, GD_acc*100))

        if epoch % 5 == 0:
            torch.save(model.gnn.state_dict(), os.path.join(args.output_dir, '{}.pth'.format(epoch)))
if __name__ == "__main__":
    main()
