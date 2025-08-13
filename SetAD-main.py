import numpy as np
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.utils import column_or_1d
from sklearn.metrics import roc_auc_score, average_precision_score
import math
import json
import random
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime 
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def setup_seed(seed):
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class ODDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return len(self.y)

class SetAD(nn.Module):
    def __init__(self, input_dim, hidden_layer, seq_len):
        super(SetAD, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_layer),
            # nn.Dropout(0.2),
            nn.ReLU()
        )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_layer, num_heads=2, batch_first=True, dropout=0.2)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_layer, 1),
            # nn.ReLU(),
            # nn.Linear(hidden_layer, 1)
        )

    def forward(self, x): # x: B, S, D  
        x = self.fc(x) # B, S, H
        pool_out, _ = self.attention(query=x, key=x, value=x) # B, S, H
        x = pool_out.sum(dim=1)
        out = self.regression_head(x)#.squeeze() #B, 1
        return out

def get_batch(X, normal_indices, anomaly_indices, batch_size, seq_len):
    n = int(batch_size / 3)
    n = batch_size // 3
    feature_dim = X.shape[1]

    # 计算每种序列类型所需的正常和异常样本总数
    # 类型0: 0个异常点, seq_len个正常点
    # 类型1: 1个异常点, seq_len-1个正常点
    # 类型2: 2个异常点, seq_len-2个正常点
    n_normals_type0 = n * seq_len
    n_normals_type1 = n * (seq_len - 1)
    n_normals_type2 = n * (seq_len - 2)
    
    n_anomalies_type1 = n * 1
    n_anomalies_type2 = n * 2

    normal_ids = np.random.choice(normal_indices, n_normals_type0 + n_normals_type1 + n_normals_type2, replace=True)
    anomaly_ids = np.random.choice(anomaly_indices, n_anomalies_type1 + n_anomalies_type2, replace=True)

    all_normal_data = X[normal_ids]
    all_anomaly_data = X[anomaly_ids]

    group0_data = all_normal_data[:n_normals_type0].view(n, seq_len, feature_dim)
    start_idx = n_normals_type0
    end_idx = n_normals_type0 + n_normals_type1
    group1_normal_part = all_normal_data[start_idx:end_idx].view(n, seq_len - 1, feature_dim)
    group1_anomaly_part = all_anomaly_data[:n_anomalies_type1].view(n, 1, feature_dim)
    group1_data = torch.cat((group1_normal_part, group1_anomaly_part), dim=1)
    start_idx = end_idx
    group2_normal_part = all_normal_data[start_idx:].view(n, seq_len - 2, feature_dim)
    group2_anomaly_part = all_anomaly_data[n_anomalies_type1:].view(n, 2, feature_dim)
    group2_data = torch.cat((group2_normal_part, group2_anomaly_part), dim=1)
    X_batch = torch.cat((group0_data, group1_data, group2_data), dim=0)
    batch_size = X_batch.shape[0]
    perms = torch.rand(batch_size, seq_len, device=device).argsort(dim=1)
    X_batch = X_batch[torch.arange(batch_size).unsqueeze(1), perms]
    y_batch = torch.tensor([0, 1, 2], dtype=torch.float32, device=device).repeat_interleave(n)
    y_batch = y_batch.unsqueeze(-1)
    
    return X_batch, y_batch

def predict_batch_ultimate_fast(model, X_batch, X_train, train_normal_indices, seq_len, num_refs, num_ensembles):
    """
    对一个batch的测试数据进行极致高效的、完全向量化的ensemble推理。
    注意：此函数会消耗大量GPU内存。

    Args:
        model: 预训练模型。
        X_batch (torch.Tensor): 一个batch的测试数据，shape: (B, N)。
        X_train (torch.Tensor): 完整的训练集，用于采样正常点。
        train_normal_indices (np.array): 训练集中正常点的索引。
        seq_len (int): 序列长度。
        num_refs (int): 每个测试点对应的参考序列数量。
        num_ensembles (int): ensemble的数量。

    Returns:
        torch.Tensor: 该batch的最终异常得分，shape: (B, 1)。
    """
    model.eval()
    E = num_ensembles
    B, N = X_batch.shape # B: batch_size, N: feature_dim
    S = seq_len
    R = num_refs

    # Data Construction
    num_base_samples = E * (S - 1)
    base_indices = np.random.choice(train_normal_indices, num_base_samples, replace=True)
    # Shape: (E, S-1, N)
    base_sequences = X_train[base_indices].view(E, S - 1, N)
    base_for_test = base_sequences.unsqueeze(1).expand(-1, B, -1, -1)
    x_batch_for_test = X_batch.unsqueeze(0).unsqueeze(2).expand(E, -1, 1, -1)
    test_seqs = torch.cat((base_for_test, x_batch_for_test), dim=2)
    num_ref_samples = E * B * R
    ref_indices = np.random.choice(train_normal_indices, num_ref_samples, replace=True)
    # Shape: (E, B, R, 1, N)
    ref_points = X_train[ref_indices].view(E, B, R, 1, N)
    #(E, B, R, S-1, N)
    base_for_refs = base_sequences.unsqueeze(1).unsqueeze(2).expand(-1, B, R, -1, -1)
    #(E, B, R, S, N)
    ref_seqs = torch.cat((base_for_refs, ref_points), dim=3)

    # (E, B, 1, S, N)
    test_seqs_expanded = test_seqs.unsqueeze(2)
    # 合并 -> (E, B, 1+R, S, N)
    all_seqs = torch.cat((test_seqs_expanded, ref_seqs), dim=2)

    # (E * B * (1+R), S, N)
    all_seqs_reshaped = all_seqs.view(E * B * (1 + R), S, N)
    perms = torch.rand(E * B * (1 + R), S, device=X_batch.device).argsort(dim=1)
    all_seqs_shuffled = all_seqs_reshaped[torch.arange(E * B * (1 + R)).unsqueeze(1), perms]

    # One pass forward
    with torch.no_grad():
        # 将所有序列一次性送入模型
        # 输入 shape: (E * B * (1+R), S, N)
        # 输出 shape: (E * B * (1+R), 1)
        all_outs = model(all_seqs_shuffled)
        all_outs_structured = all_outs.view(E, B, 1 + R, 1)
        test_outs = all_outs_structured[:, :, 0, :] # Shape: (E, B, 1)
        ref_outs = all_outs_structured[:, :, 1:, :] # Shape: (E, B, R, 1)
        ref_outs_mean = ref_outs.mean(dim=2)
        ensemble_scores = test_outs - ref_outs_mean
        final_scores = ensemble_scores.mean(dim=0)

    return final_scores

def predict_batch_fast(model, X_batch, X_train, train_normal_indices, seq_len, num_refs, num_ensembles):
    """
    对一个batch的测试数据X_batch进行高效的、带ensemble的推理。

    Args:
        model: 预训练模型。
        X_batch (torch.Tensor): 一个batch的测试数据，shape: (B, N)。
        X_train (torch.Tensor): 完整的训练集，用于采样正常点。
        train_normal_indices (np.array): 训练集中正常点的索引。
        seq_len (int): 序列长度。
        num_refs (int): 每个测试点对应的参考序列数量。
        num_ensembles (int): ensemble的数量。

    Returns:
        torch.Tensor: 该batch的最终异常得分，shape: (B, 1)。
    """
    model.eval()
    B, N = X_batch.shape # B: batch_size, N: feature_dim
    total_scores = torch.zeros(B, 1, device=X_batch.device)

    # 保留Ensemble循环，以控制内存使用
    for _ in range(num_ensembles):
        # Data Construction
        # shared context
        base_indices = np.random.choice(train_normal_indices, seq_len - 1, replace=False)
        base_sequence = X_train[base_indices] # Shape: (S-1, N)

        base_sequence_expanded = base_sequence.unsqueeze(0).expand(B, -1, -1) # Shape: (B, S-1, N)
        test_seqs = torch.cat((base_sequence_expanded, X_batch.unsqueeze(1)), dim=1) # Shape: (B, S, N)
        perms = torch.rand(B, seq_len, device=X_batch.device).argsort(dim=1)
        test_seqs = test_seqs[torch.arange(B).unsqueeze(1), perms]
        ref_indices = np.random.choice(train_normal_indices, B * num_refs, replace=True)
        ref_points = X_train[ref_indices].view(B, num_refs, 1, N) # Shape: (B, num_refs, 1, N)
        base_for_refs = base_sequence.view(1, 1, seq_len - 1, N).expand(B, num_refs, -1, -1)
        ref_seqs = torch.cat((base_for_refs, ref_points), dim=2) # Shape: (B, num_refs, S, N)
        
        ref_seqs_reshaped = ref_seqs.view(B * num_refs, seq_len, N)
        perms_ref = torch.rand(B * num_refs, seq_len, device=X_batch.device).argsort(dim=1)
        ref_seqs_shuffled = ref_seqs_reshaped[torch.arange(B * num_refs).unsqueeze(1), perms_ref]
        ref_seqs = ref_seqs_shuffled.view(B, num_refs, seq_len, N)

        # Inference
        with torch.no_grad():
            # s(x)
            test_out = model(test_seqs) # Shape: (B, 1)

            # s(r)
            ref_seqs_reshaped = ref_seqs.view(B * num_refs, seq_len, N)
            ref_out = model(ref_seqs_reshaped) # Shape: (B * num_refs, 1)
            ref_out_mean = ref_out.view(B, num_refs, 1).mean(dim=1) # Shape: (B, 1)

            # bar{s}(x)
            score = test_out - ref_out_mean
            total_scores += score

    return total_scores / num_ensembles

def evaluate_print(clf_name, y, scores):
    roc = np.round(roc_auc_score(y, scores), decimals=4)
    prn = np.round(average_precision_score(y_true=y, y_score=scores, pos_label=1), decimals=4)
    print(f'{clf_name}: ROC: {roc}, PR: {prn}')
    return roc, prn


def train(config, test_loader, X_train, train_normal_indices, train_anomaly_indices, X_test):
    d, hidden_layer, n_epochs, seq_len, best_loss, step, early_stop_count = config.dimention, config.hidden_layer, config.n_epochs, config.seq_len, math.inf, 0, 0
    best_loss = 100000
    best_roc = 0
    best_prn = 0

    criterion = nn.L1Loss()
    if config.write_summary:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('./runs/seqlen'+str(seq_len)+config.dataset+f'{timestamp}')

    model = SetAD(d, hidden_layer, seq_len).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate, weight_decay=0.2)

    train_loss = []
    roc1 = []
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        train_bar = tqdm(range(20), position=0, leave=True)
        for _ in train_bar:
            X_batch, y_batch = get_batch(X_train, train_normal_indices, train_anomaly_indices, config.batch_size, seq_len)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch) 
        
            loss.backward()
            optimizer.step()
            step += 1
            loss_record.append(loss.detach().item())
            train_bar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_bar.set_postfix({'loss': loss.detach().item()})
        mean_loss = np.mean(loss_record)
        print(f'Epoch [{epoch + 1}/{n_epochs}], loss: {mean_loss}]')
        train_loss.append(mean_loss)

        model.eval()
        all_scores = []
        all_labels = []
        
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)

            # batch_scores = predict_batch_fast(
            #     model, X_batch, X_train, train_normal_indices, 
            #     seq_len, 30, 60
            # )

            batch_scores = predict_batch_ultimate_fast(
                model, X_batch, X_train, train_normal_indices, 
                seq_len, 30, 60
            )
            
            all_scores.append(batch_scores.cpu())
            all_labels.append(y_batch.cpu())
        Out = torch.cat(all_scores, dim=0)
        Y = torch.cat(all_labels, dim=0)

        Out = Out.cpu().detach().numpy()
        test_roc, test_pr = evaluate_print('distance', Y, Out)
        roc1.append(test_roc)
        if config.write_summary:
            writer.add_scalar('Loss', mean_loss, epoch)
            writer.add_scalar('ROC', test_roc, epoch)
            writer.add_scalar('PR', test_pr, epoch)
        if mean_loss < best_loss:
            print('best model saved...')
            best_loss = mean_loss
            best_roc = test_roc
            best_prn = test_pr
            early_stop_count = 0
        else:
            early_stop_count += 1
        if early_stop_count == config.early_stop:
            print('early stop...')
            break
    if config.write_summary:
        writer.close()
    return best_roc, best_prn

def run(config, rocs, prns, seed):
    setup_seed(seed)
    print(f'seed: {seed}')
    data = json.load(open('../processed_data/'+config.dataset, 'r'))
    X = np.array(data[0])
    y = np.array(data[1])
    print(X.shape, y.shape) 

    config.dimention = X.shape[1]
    config.hidden_layer = 20
    num_total_anomaly = sum(y)
    print(f'num_total_anomaly: {num_total_anomaly}')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    train_size = len(X_train)
    anomaly_indices = np.where(y_train == 1)[0]
    n_anomalies = len(anomaly_indices)
    n_noise = int(len(np.where(y == 0)[0]) * config.contamination_rate / (1. - config.contamination_rate))
    n_labeled_anomalies = int(n_anomalies * config.labeled_ratio)
    if n_noise > n_anomalies - n_labeled_anomalies:
        n_noise = n_anomalies - n_labeled_anomalies
    n_train_anomalies = n_labeled_anomalies + n_noise
    rng = np.random.RandomState(seed) 
    if n_anomalies > n_train_anomalies:
        mn = n_anomalies - n_train_anomalies
        remove_idx = rng.choice(anomaly_indices, mn, replace=False)        
        retain_idx = set(np.arange(X_train.shape[0])) - set(remove_idx)
        retain_idx = list(retain_idx)
        X_train = X_train[retain_idx]
        y_train = y_train[retain_idx]     

    anomaly_indices = np.where(y_train == 1)[0]
    noise_indices = rng.choice(anomaly_indices, n_noise, replace=False)
    y_train[noise_indices] = 0
    train_anomaly_indices = list(set(anomaly_indices) - set(noise_indices))
    train_normal_indices = list(set(np.arange(X_train.shape[0])) - set(train_anomaly_indices))
    print(f'dataset:{config.dataset} shape:{X.shape} train set:{X_train.shape} labeled anomalies:{n_labeled_anomalies} noise:{n_noise}')

    test_set = ODDataset(X_test, y_test)
    if config.dataset == 'census.json':
        test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, pin_memory=True)
    else:
        test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, pin_memory=True)

    config.max_anomaly = n_labeled_anomalies
    config.min_anomaly = int(n_labeled_anomalies * 0.5) 
    if X.shape[0] < 50000:   # small number of learning epochs is enough for large datasets
        config.n_epochs = 20
        config.early_stop = 10
    else:
        config.n_epochs = 20
        config.early_stop = 10

    print(config) 
    if seed == 9:
        config.write_summary = True
    else:
        config.write_summary = False
    # exit()
    roc, prn = train(config, test_loader, X_train, train_normal_indices, train_anomaly_indices, X_test)   
    rocs.append(roc)
    prns.append(prn)


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameters")
    parser.add_argument('--dataset', default='Cardiotocography.json', type=str, help='Dataset') 
    parser.add_argument('--batch_size', default=512, type=int, help='Batch size, you may increase it when dealing with large datasets')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--specify_anomaly_ratio', default=False, type=bool, help='Whether to specify the ratio of anomalies')
    parser.add_argument('--labeled_ratio', default=0.05, type=float, help='Ratio of labeled anomalies')
    parser.add_argument('--contamination_rate', default=0.02, type=float, help='Ratio of unlabeled anomalies in the training set')
    parser.add_argument('--seq_len', default=2, type=int, help='Length of the sequence')
    parser.add_argument('-o', default=False, type=str, help='Write output?')
    return parser.parse_args()


config = parse_args()
seed_list = range(10)
rocs = []
prns = []
cad_rocs = []
cad_prns = []
dev_rocs = []
dev_prns = []
for seed in seed_list:
    run(config, rocs, prns, seed)
    print(rocs)
    print(prns)
    print(np.mean(rocs), np.mean(prns))

if config.o:
    dir = os.path.basename(__file__).split('.')[0] + 'out/'
    os.makedirs(dir, exist_ok=True)
    with open(dir + config.dataset + '.txt', 'a') as f:
        f.write(str(config)+'\n')
        f.write(str(rocs)+'\n')
        f.write(str(prns)+'\n')
        f.write(str(np.mean(rocs))+'  ' +str(np.mean(prns))+'\n\n')
