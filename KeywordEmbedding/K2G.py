import igraph as ig
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch


def build_huperGraph(file_gml, method="onehot"):
    '''
    craete the hypergraph using gml graph
    '''
    graph = ig.Graph.Read_GML(file_gml)
    hyperedges = []
    hyperedges_weight = defaultdict(int)

    all_keywords = set()

    print(f"Readding {file_gml} now, wait...")

    for node in graph.vs:
        if "keywords" in node.attributes():
            keywords_str = node["keywords"]
            if keywords_str and keywords_str.strip():
                keywords_list = [k.strip() for k in keywords_str.split(',') if k.strip()]
                keywords_tuple = tuple(sorted(keywords_list))
                hyperedges.append(keywords_tuple)
                hyperedges_weight[keywords_tuple] += 1
                all_keywords.update(keywords_list)
    
    print(f"\n{file_gml} graph informations: ")
    print(f"keywords number: {len(all_keywords)}")
    print(f"hyperedges number: {len(hyperedges)}")
    print(f"hyperedge weights number: {len(hyperedges_weight)}")
    print(f"\nhyperedge example: ")
    print(hyperedges[0])
    print(f"\nhyperedge weights example: ")
    print(hyperedges_weight[hyperedges[0]])

    keywords = sorted(list(all_keywords))
    keyword_to_idx = {keyword: idx for idx, keyword in enumerate(keywords)}

    unique_hyperedges = list(hyperedges_weight.keys())
    hyperedge_to_idx = {hyperedge: idx for idx, hyperedge in enumerate(unique_hyperedges)}

    # the hypergraph matrix H

    n_keywords = len(keywords)
    n_hyperedges = len(unique_hyperedges)

    H = np.zeros((n_keywords, n_hyperedges), dtype=np.float32)
    W = np.zeros(n_hyperedges, dtype=np.float32)
    
    for hyperedge, hyperedge_idx in hyperedge_to_idx.items():
        weight = hyperedges_weight[hyperedge]
        W[hyperedge_idx] = weight
        for keyword in hyperedge:
            keyword_idx = keyword_to_idx[keyword]
            H[keyword_idx, hyperedge_idx] = 1

    # H = H / np.maximum(H.sum(axis=0), 1)
    W = W / H.sum(axis=0)
    print(f"keywords: {keywords}")
    print(f"hypergraph shape: {H.shape}")
    print("non zero number: {}, non zero rate: {:.2f}%".format(np.count_nonzero(H), np.count_nonzero(H)/(H.shape[0]*H.shape[1]) * 100))    

    if method == 'onehot':
        n_keywords = len(keywords)
        features = np.eye(n_keywords)
    elif method == 'random':
        # 使用随机特征（示例用）
        n_keywords = len(keywords)
        features = np.random.randn(n_keywords, 128)
    elif method == 'tfidf':
        # 如果关键词有文本描述，可以使用TF-IDF
        # 这里假设keywords就是文本
        vectorizer = TfidfVectorizer()
        features = vectorizer.fit_transform(keywords).toarray()
    
    fe = torch.from_numpy(features).float()

    return H, keyword_to_idx, hyperedges_weight, keywords, fe, graph, W

def calculate_optimal_space_cost(H, keywords, hyperedges_weight):
    nnz = np.count_nonzero(H)
    n_v = len(keywords)
    n_e = len(hyperedges_weight)
    
    # 1. 关联矩阵 H (CSR 估算)
    # data (float32) + indices (int32) + indptr (int32)
    h_sparse_bytes = (nnz * 4) + (nnz * 4) + ((n_v + 1) * 4)
    h_sparse_mb = h_sparse_bytes / (1024**2)
    
    # 2. 特征 fe (Index Mapping 估算)
    # 仅存储索引 ID 序列
    fe_index_bytes = n_v * 4
    fe_index_mb = fe_index_bytes / (1024**2)
    
    # 3. 权重 W
    w_bytes = n_e * 4
    w_mb = w_bytes / (1024**2)
    
    # 4. 对比：原始稠密存储 (H 和 fe 均为 dense)
    h_dense_mb = (n_v * n_e * 4) / (1024**2)
    fe_dense_mb = (n_v * n_v * 4) / (1024**2)
    
    print("\n" + "★"*15 + " 超图最优存储成本报告 " + "★"*15)
    print(f"{'指标':<20} | {'规模':<15} | {'优化后内存 (MB)':<15}")
    print("-" * 60)
    print(f"{'节点(关键词)数量':<20} | {n_v:<15} | --")
    print(f"{'超边数量':<20} | {n_e:<15} | --")
    print(f"{'非零关联数(NNZ)':<20} | {nnz:<15} | --")
    print("-" * 60)
    print(f"{'关联矩阵 H (CSR)':<20} | {'Sparse':<15} | {h_sparse_mb:.4f} MB")
    print(f"{'节点特征 fe (Index)':<20} | {'Mapping':<15} | {fe_index_mb:.4f} MB")
    print(f"{'超边权重 W (Dense)':<20} | {'Vector':<15} | {w_mb:.4f} MB")
    print("-" * 60)
    print(f"【总计最优开销】: {h_sparse_mb + fe_index_mb + w_mb:.4f} MB")
    print(f"【原始稠密开销】: {h_dense_mb + fe_dense_mb:.4f} MB")
    print(f"【空间缩减率】: {(1 - (h_sparse_mb + fe_index_mb + w_mb)/(h_dense_mb + fe_dense_mb))*100:.6f}%")
    print("★"*52)

def print_hypergraph_space_info(H, fe, hyperedges_weight):
    # 1. 逻辑规模统计
    n_nodes, n_edges = H.shape
    total_incidences = np.count_nonzero(H)
    avg_edge_size = total_incidences / n_edges if n_edges > 0 else 0
    
    # 2. 内存占用统计 (单位: MB)
    h_memory = H.nbytes / (1024**2)
    fe_memory = fe.element_size() * fe.nelement() / (1024**2)
    
    # 3. 稀疏度计算
    sparsity = (1 - total_incidences / (n_nodes * n_edges)) * 100

    print("\n" + "—"*20 + " 空间复杂度统计 " + "—"*20)
    print(f"| 节点数量 (V): {n_nodes}")
    print(f"| 超边数量 (E): {n_edges}")
    print(f"| 总关联数 (Non-zeros): {total_incidences}")
    print(f"| 平均每个超边包含节点数: {avg_edge_size:.2f}")
    print(f"| 矩阵稀疏度: {sparsity:.2f}%")
    print("-" * 50)
    print(f"| 关联矩阵 H 占用内存: {h_memory:.2f} MB")
    print(f"| 特征矩阵 fe 占用内存: {fe_memory:.2f} MB")
    
    # 警告：如果 H 很大且仍在使用 dense 存储
    if h_memory > 1024:
        print(">> [警告] H 矩阵占用内存已超过 1GB，建议考虑使用 scipy.sparse 存储。")
    print("—"*50)

# 在 main 中调用：
# H, keyword_to_idx, hyperedges_weight, keywords, fe, graph, W = build_huperGraph(...)
# print_hypergraph_space_info(H, fe, hyperedges_weight)

def print_hypergraph_info(H, keyword_to_idx, hyperedges_weight, keywords, top_n=10):
    """
    打印超图信息
    """
    print("\n" + "="*50)
    print("超图统计信息:")
    print("="*50)
    print(f"关键词数量: {len(keywords)}")
    print(f"超边数量: {len(hyperedges_weight)}")
    print(f"关联矩阵密度: {np.count_nonzero(H) / (H.shape[0] * H.shape[1]):.4f}")
    
    print(f"\n权重最高的前{top_n}个超边:")
    sorted_hyperedges = sorted(hyperedges_weight.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    for i, (hyperedge, weight) in enumerate(sorted_hyperedges, 1):
        print(f"{i:2d}. {hyperedge} (权重: {weight})")
    
    keyword_degrees = H.sum(axis=1)
    print(f"\n关键词度分布:")
    print(f"  平均度: {keyword_degrees.mean():.2f}")
    print(f"  最大度: {keyword_degrees.max():.0f}")
    print(f"  最小度: {keyword_degrees.min():.0f}")

def estimate_sparse_overhead(H, fe):
    """
    在不转换矩阵的情况下，估算如果使用稀疏矩阵存储时的空间开销
    """
    # 基础参数获取
    nnz = np.count_nonzero(H)
    n_rows, n_cols = H.shape
    
    # 定义字节大小 (float32=4, int32=4)
    f_size = 4 
    i_size = 4
    
    # 1. 当前稠密矩阵开销
    dense_size_mb = H.nbytes / (1024**2)
    
    # 2. 模拟 COO 格式开销 (Row indices, Col indices, Data)
    # 每个非零元存两个 int 和一个 float
    coo_size_mb = (nnz * (i_size + i_size + f_size)) / (1024**2)
    
    # 3. 模拟 CSR 格式开销 (Data, Indices, Indptr)
    # Data: nnz * float, Indices: nnz * int, Indptr: (rows+1) * int
    csr_size_mb = (nnz * f_size + nnz * i_size + (n_rows + 1) * i_size) / (1024**2)
    
    # 4. 特征矩阵 fe 的开销 (通常特征是稠密的)
    fe_size_mb = fe.element_size() * fe.nelement() / (1024**2)

    print("\n" + "="*20 + " 空间开销深度统计 " + "="*20)
    print(f"矩阵规模: {n_rows} (节点) x {n_cols} (超边)")
    print(f"非零元素数量 (NNZ): {nnz}")
    print(f"矩阵稀疏度: {(1 - nnz/(n_rows*n_cols))*100:.2f}%")
    print("-" * 50)
    print(f"【当前】稠密矩阵 H 内存占用:  {dense_size_mb:.2f} MB")
    print(f"【估算】稀疏 COO 格式内存占用: {coo_size_mb:.2f} MB")
    print(f"【估算】稀疏 CSR 格式内存占用: {csr_size_mb:.2f} MB")
    print(f"【实际】节点特征 fe 内存占用: {fe_size_mb:.2f} MB")
    
    if dense_size_mb > 0:
        ratio = (1 - csr_size_mb / dense_size_mb) * 100
        print(f"\n结论：切换到 CSR 格式可节省约 {ratio:.2f}% 的内存空间。")
    print("="*55)

if __name__ == "__main__":
    file_gml = "../Datasets/precompute/synthetic/50000-124812-50-3/G-uni.gml"
    H, keyword_to_idx, hyperedges_weight, keywords, fe, graph, W = build_huperGraph(file_gml, method="onehot")
    print_hypergraph_info(H, keyword_to_idx, hyperedges_weight, keywords)
    
    # print_hypergraph_space_info(H, fe, hyperedges_weight)
    
    # estimate_sparse_overhead(H, fe)
    
    calculate_optimal_space_cost(H, keywords, hyperedges_weight)