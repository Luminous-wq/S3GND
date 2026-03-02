import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
import torch
import logging
import torch.nn.functional as F
import numpy as np
import random

from K2G import build_huperGraph
from Utils.utils import generate_G_from_H, evaluate, pruning_power, generate_query, calculate_expected_pruned_count, evaluate_real_distribution
from torch.utils.data import Dataset, DataLoader
from HGNNmodel import KeywordHGNN
import torch.optim as optim


def build_balanced_multi_rel_dataset(
    hyperedges_weight,
    keyword_to_idx,
    max_pos_attempts=500000, # 寻找包含关系的最大尝试次数
    ratios=(1, 1, 1),        # 包含:交集:不相交 的比例 (1:1:1 表示等量)
    seed=42,
    fallback_size=1000
):
    random.seed(seed)
    hyperedges = list(hyperedges_weight.keys())
    n = len(hyperedges)
    edge_sets = [set(keyword_to_idx[k] for k in edge) for edge in hyperedges]
    
    # 类别桶
    buckets = {1: [], 2: [], 0: []} # 1:包含, 2:交集, 0:不相交

    print("Step 1: 寻找包含关系样本 (Baseline)...")
    for _ in range(max_pos_attempts):
        i, j = random.randint(0, n - 1), random.randint(0, n - 1)
        if i == j: continue
        
        set_i, set_j = edge_sets[i], edge_sets[j]
        
        # 严格判断：包含关系
        if set_i.issubset(set_j):
            buckets[1].append((hyperedges[i], hyperedges[j], 1))
        elif set_j.issubset(set_i):
            buckets[1].append((hyperedges[j], hyperedges[i], 1))
            
    # 确定基准数量
    num_inc = len(buckets[1])
    target_int = 0
    target_dis = 0

    if num_inc > 0:
        print(f"找到包含关系样本: {num_inc} 个。将以此为基准采样其他类别...")
        
        base_ratio = ratios[0] if ratios[0] > 0 else 1 
        target_int = int(num_inc * (ratios[1] / base_ratio))
        target_dis = int(num_inc * (ratios[2] / base_ratio))
    else:
        print(f"提示：未找到包含关系 (Count=0)。")
        print(f"将忽略包含关系，仅按比例 {ratios[1]}:{ratios[2]} 生成剩余样本 (总数约 {fallback_size})...")
        
    
        sum_rest_ratios = ratios[1] + ratios[2]
        if sum_rest_ratios == 0:
            print("错误：交集和不相交的比例配置均为 0，无法生成。")
            return []
            
        target_int = int(fallback_size * (ratios[1] / sum_rest_ratios))
        target_dis = int(fallback_size * (ratios[2] / sum_rest_ratios))


    print(f"Step 2: 采样交集样本 (目标: {target_int}) 和不相交样本 (目标: {target_dis})...")
    
    attempts = 0
    while (len(buckets[2]) < target_int or len(buckets[0]) < target_dis) and attempts < max_pos_attempts:
        i, j = random.randint(0, n - 1), random.randint(0, n - 1)
        if i == j: continue
        
        set_i, set_j = edge_sets[i], edge_sets[j]
      
        if set_i.issubset(set_j) or set_j.issubset(set_i):
            continue 
        
        if not set_i.isdisjoint(set_j):
       
            if len(buckets[2]) < target_int:
                buckets[2].append((hyperedges[i], hyperedges[j], 2))
        else:
         
            if len(buckets[0]) < target_dis:
                buckets[0].append((hyperedges[i], hyperedges[j], 0))
        
        attempts += 1

    dataset = buckets[1] + buckets[2] + buckets[0]
    random.shuffle(dataset)
    
    print(f"最终数据集构成: 包含={len(buckets[1])}, 交集={len(buckets[2])}, 不相交={len(buckets[0])}")
    return dataset

def compute_batch_pair_losses_containment(A_emb, B_emb, labels, A_mask, B_mask, margin=1.0):

    inf_mask_A = (~A_mask).unsqueeze(-1)
    min_a = A_emb.masked_fill(inf_mask_A, 1e9).min(dim=1)[0]
    max_a = A_emb.masked_fill(inf_mask_A, -1e9).max(dim=1)[0]
    
    inf_mask_B = (~B_mask).unsqueeze(-1)
    min_b = B_emb.masked_fill(inf_mask_B, 1e9).min(dim=1)[0]
    max_b = B_emb.masked_fill(inf_mask_B, -1e9).max(dim=1)[0]


    width_a = torch.abs(max_a - min_a)
    width_b = torch.abs(max_b - min_b)

    inter_width = torch.abs(torch.min(max_a, max_b) - torch.max(min_a, min_b))
    
    area_inter = torch.sum(torch.log(inter_width + 1e-6), dim=1)
   
    
    area_a = torch.sum(torch.log(width_a + 1e-6), dim=1)
    area_b = torch.sum(torch.log(width_b + 1e-6), dim=1)

    total_loss = 0
    

    mask1 = (labels == 1)
    if mask1.any():

        loss1 = torch.exp(area_a[mask1])
        total_loss += loss1.mean()

    mask2 = (labels == 2)
    if mask2.any():
        eps = 1e-8
        log_area_a = torch.sum(torch.log(F.relu(max_a - min_a) + eps), dim=1)
        log_area_inter = torch.sum(torch.log(F.relu(torch.min(max_a, max_b) - torch.max(min_a, min_b)) + eps), dim=1)
        log_prob = log_area_inter - log_area_a
        prob_redundant = torch.exp(log_prob)
        loss2 = prob_redundant[mask2]
        total_loss += loss2.mean()


    mask0 = (labels == 0)
    if mask0.any():
        loss0 = torch.exp(area_a[mask0] + area_b[mask0]) 
        total_loss += loss0.mean()

    return total_loss

def to_torch_sparse(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

class KeywordsDataset(Dataset):
    def __init__(self, train_list):
        super().__init__()
        self.data = train_list
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        A, B, label = self.data[index]
        return list(A), list(B), label

def collate_fn(batch):
    A_batch = [item[0] for item in batch]
    B_batch = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return A_batch, B_batch, torch.tensor(labels, dtype=torch.float32)

from torch.nn.utils.rnn import pad_sequence

def indexGet_and_Pad(all_embeddings, keyword_to_idx, A_batch, B_batch):
    A_embeddings_list = []
    B_embeddings_list = []

    for kw_list in A_batch:
        idx_list = [keyword_to_idx[k] for k in kw_list]
        idx_tensor = torch.tensor(idx_list, device=all_embeddings.device, dtype=torch.long)
        emb = all_embeddings[idx_tensor] # emb 形状: [num_keywords_i, D]
        A_embeddings_list.append(emb)

    for kw_list in B_batch:
        idx_list = [keyword_to_idx[k] for k in kw_list]
        idx_tensor = torch.tensor(idx_list, device=all_embeddings.device, dtype=torch.long)
        emb = all_embeddings[idx_tensor] # emb 形状: [num_keywords_j, D]
        B_embeddings_list.append(emb)


    A_embeddings_tensor = pad_sequence(A_embeddings_list, batch_first=True, padding_value=0.0)
    B_embeddings_tensor = pad_sequence(B_embeddings_list, batch_first=True, padding_value=0.0)

    A_mask = pad_sequence([torch.ones(e.size(0), dtype=torch.bool, device=all_embeddings.device) for e in A_embeddings_list], 
                          batch_first=True, padding_value=False)
    B_mask = pad_sequence([torch.ones(e.size(0), dtype=torch.bool, device=all_embeddings.device) for e in B_embeddings_list], 
                          batch_first=True, padding_value=False)

    return A_embeddings_tensor, B_embeddings_tensor, A_mask, B_mask

def save_embeddings(embeddings, keyword_to_idx, keywords, output_file="../Results/hgnn_keyword_embeddings_sample.pt"):
    keyword_embeddings = {}
    for keyword in keywords:
        idx = keyword_to_idx[keyword]
        keyword_embeddings[keyword] = embeddings[idx].cpu().numpy()
    
    torch.save(keyword_embeddings, output_file)
    print(f"关键词嵌入已保存到 {output_file}")

    embedding_matrix = embeddings.cpu().numpy()
    np.save(output_file.replace('.pt', '_matrix.npy'), embedding_matrix)
    
    return keyword_embeddings

def main(gml_file, batch_size, epochs, lr, embedding_dim, hidden_dim, query_graph_size, keywordDomain, evalnumber, vision):
    
    print("Now, construct hypergraph...")
    time1 = time.time()
    H, keyword_to_idx, hyperedges_weight, keywords, features, Graph, W = build_huperGraph(gml_file)
    print(type(H), type(keyword_to_idx), type(hyperedges_weight), type(keywords), type(features))
    # print(H[0])
    
    G = generate_G_from_H(H, W)
    print("keyword_to_idx: {}".format(keyword_to_idx))

    time2 = time.time()
    print(f"\nconstruct hypergraph needs time: {time2-time1}\n")
    
    print("\nNow, construct training datasets...")
    detaset = build_balanced_multi_rel_dataset(hyperedges_weight, keyword_to_idx)
    # print("training dataset example: {}".format(detaset[:5]))
    
    time3 = time.time()
    print(f"\nconstruct training dataset needs time: {time3-time2}\n")
    
    D = KeywordsDataset(detaset)
    KDataset = DataLoader(
        D, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # for A_batch, B_batch, labels in KDataset:
    #     # A_batch: List[List[str]]
    #     # B_batch: List[List[str]]
    #     # labels: Tensor [64]
    #     print(len(A_batch), len(B_batch), labels.shape)
    #     # print(A_batch, B_batch, labels)
    #     break
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = features.shape[1]
    # print(in_dim)
    model = KeywordHGNN(input_dim=in_dim, embedding_dim=embedding_dim, hidden_dim=hidden_dim, dropout=0.5)
    # G_tensor = torch.tensor(G, dtype=torch.float32, device=device)
    G_tensor = to_torch_sparse(G).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    model.to(device)
    features = features.to(device)

    
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        nsteps = 0
        time_train = time.time()
        for A_batch, B_batch, labels in KDataset:
            all_embeddings = model(features, G_tensor)
            # 从all_embeddings按照keyword_to_idx获得对应batch的编码表示
            # A_embedding, B_embedding = indexGet(all_embeddings, keyword_to_idx, A_batch, B_batch)
            # loss = compute_batch_pair_losses_dist(A_embedding, B_embedding, labels)
            
            # 1. 调用新的 indexGet_and_Pad
            A_embedding, B_embedding, A_mask, B_mask = indexGet_and_Pad(all_embeddings, keyword_to_idx, A_batch, B_batch)
            
            # 2. 传入 mask 给损失函数
            loss = compute_batch_pair_losses_containment(A_embedding, B_embedding, labels, A_mask, B_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # if nsteps % 100 == 0:
            #     print(nsteps, loss.item())
            nsteps += 1
        # avg_loss = epoch_loss / max(nsteps, 1)
        # logging.info(f"[Epoch {epoch}] avg_loss={avg_loss:.4f}")
        # time_end = time.time()
        # print(f"one epoch needs {time_end-time_train} time.")
        
    #     if epoch % 10 == 0:
    #         model.eval()
    #         with torch.no_grad():
    #             all_embeddings = model(features, G_tensor)
            
    #         keyword_embeddings = {}
    #         for keyword in keywords:
    #             idx = keyword_to_idx[keyword]
    #             keyword_embeddings[keyword] = all_embeddings[idx].cpu().numpy()
    #         evaluate_real_distribution(None, keyword_embeddings, keywords)
    #         # G_q = generate_query(G=Graph, n=query_graph_size, p=0.3, keyword_domain=keywordDomain)
    #         # pp, fp, fn = pruning_power(G_q, Graph, keyword_embeddings)
    #         # expected_pruned, _, _ = calculate_expected_pruned_count(G_q, Graph)
    #         rate_pp, rate_fp, rate_fn = 0, 0, 0
    #         exp = 0
    #         r_number = 0
            
    #         for _ in range(r_number):
    #             G_q = generate_query(G=Graph, n=query_graph_size, p=0.3, keyword_domain=keywordDomain)
    #             pp, fp, fn = pruning_power(G_q, Graph, keyword_embeddings)
    #             expected_pruned, pruned_ratio, _ = calculate_expected_pruned_count(G_q, Graph)
    #             rate_pp += pp
    #             rate_fp += fp
    #             rate_fn += fn
    #             exp += pruned_ratio
            
    #         # print(f"\n=== 5次剪枝统计结果 ===")
    #         # print(f"平均剪枝准确率: {rate_pp/r_number:.4f}")
    #         # print(f"平均假阳性率: {rate_fp/r_number:.4f}")
    #         # print(f"平均假阴性率: {rate_fn/r_number:.4f}")
    #         # print(f"平均ground truth: {exp/r_number:.4f}")
            
    #     if epoch == epochs-1:
    #         checkpoint = {
    #             "epoch": epoch,
    #             "model_state": model.state_dict(),
    #             "optimizer_state": optimizer.state_dict(),
    #         }

    #         torch.save(checkpoint, "../Results/checkpoint_"+vision+".pth")
    # with torch.no_grad():
    #     all_embeddings = model(features, G_tensor)
        
    # E = save_embeddings(all_embeddings, keyword_to_idx, keywords, output_file="../Results/hgnn_keyword_embeddings_"+vision+".pt")
    # evaluate_real_distribution("../Results/hgnn_keyword_embeddings_"+vision+".pt", None, keywords_list=keywords)
    # rate_pp, rate_fp, rate_fn = 0, 0, 0
    # exp = 0
    # r_number = 0
    
    
    # keyword_embeddings = {}
    # for keyword in keywords:
    #     idx = keyword_to_idx[keyword]
    #     keyword_embeddings[keyword] = all_embeddings[idx].cpu().numpy()
    
    # for _ in range(r_number):
    #     G_q = generate_query(G=Graph, n=query_graph_size, p=0.3, keyword_domain=keywordDomain)
    #     pp, fp, fn = pruning_power(G_q, Graph, keyword_embeddings)
    #     expected_pruned, pruned_ratio, _ = calculate_expected_pruned_count(G_q, Graph)
    #     rate_pp += pp
    #     rate_fp += fp
    #     rate_fn += fn
    #     exp += pruned_ratio
    
    # print(f"\n=== 50次剪枝统计结果 ===")
    # print(f"平均剪枝准确率: {rate_pp/r_number:.4f}")
    # print(f"平均假阳性率: {rate_fp/r_number:.4f}")
    # print(f"平均假阴性率: {rate_fn/r_number:.4f}")
    # print(f"平均ground truth: {exp/r_number:.4f}")
        
        
        
    # torch.save(model.state_dict(), "../Results/model"+vision+".pth")
    return model

if __name__ == "__main__":
    
    # python -u train.py 2>&1 | tee ../Results/train.log

    parser = argparse.ArgumentParser()
    parser.add_argument("--gml", type=str, 
                        help="Input gml file for hypergraph builder.",
                        default="../Datasets/precompute/synthetic/10000-24979-50-3/G-gau.gml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=64) 
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--query-size", type=int, default=5)
    parser.add_argument("--keywordDomain", type=int, default=50)
    parser.add_argument("--evalnumber", type=int, default=10)
    parser.add_argument("--out", type=str, default="../Results")
    parser.add_argument("--dataset", type=str, default="1WGau")
    parser.add_argument("--n", type=str, help="the version name for this training to save outfile", default="1WGau")
    args = parser.parse_args()
    print(args)
    logging.getLogger().setLevel(logging.INFO)
    t = time.time()
    model = main(gml_file=args.gml, 
                 batch_size=args.batch_size, 
                 epochs=args.epochs,
                 lr=args.lr,
                 embedding_dim=args.embedding_dim,
                 hidden_dim=args.hidden_dim,
                 query_graph_size=args.query_size,
                 keywordDomain=args.keywordDomain,
                 evalnumber=args.evalnumber,
                 vision=args.n)
    print("Done.")
    print(f"Total time: {time.time() - t} seconds.")