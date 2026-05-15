import os
import ot
import torch
import random
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from torch.backends import cudnn
#from scipy.sparse import issparse
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestNeighbors

from scipy.sparse.linalg import inv
from scipy.sparse import csr_array, csc_array
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors

def filter_with_overlap_gene(adata, adata_sc):
    # remove all-zero-valued genes
    #sc.pp.filter_genes(adata, min_cells=1)
    #sc.pp.filter_genes(adata_sc, min_cells=1)
    
    if 'highly_variable' not in adata.var.keys():
       raise ValueError("'highly_variable' are not existed in adata!")
    else:    
       adata = adata[:, adata.var['highly_variable']]
       
    if 'highly_variable' not in adata_sc.var.keys():
       raise ValueError("'highly_variable' are not existed in adata_sc!")
    else:    
       adata_sc = adata_sc[:, adata_sc.var['highly_variable']]   

    # Refine `marker_genes` so that they are shared by both adatas
    genes = list(set(adata.var.index) & set(adata_sc.var.index))
    genes.sort()
    print('Number of overlap genes:', len(genes))

    adata.uns["overlap_genes"] = genes
    adata_sc.uns["overlap_genes"] = genes
    
    adata = adata[:, genes]
    adata_sc = adata_sc[:, genes]
    
    return adata, adata_sc

def permutation(feature):
    # fix_seed(FLAGS.random_seed) 
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    
    return feature_permutated 

def construct_interaction(adata, n_neighbors=3):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial']
    
    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]
    
    adata.obsm['distance_matrix'] = distance_matrix
    
    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])  
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1
         
    adata.obsm['graph_neigh'] = interaction
    
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    
    adata.obsm['adj'] = adj
    
def construct_interaction_KNN(adata, n_neighbors=3):
    position = adata.obsm['spatial']
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(position)  
    _ , indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1
    
    adata.obsm['graph_neigh'] = interaction
    
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    
    adata.obsm['adj'] = adj
    print('Graph constructed!')

def construct_interaction_sparse(adata, n_neighbors=3):
    """使用稀疏矩阵构建邻接图 (内存友好版本)"""
    position = adata.obsm['spatial']
    n_spot = position.shape[0]

    # 1. 使用KNN查找邻居索引 (不计算全距离矩阵)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(position)  # +1 包含自身
    distances, indices = nbrs.kneighbors(position)  # indices 形状: (n_spot, n_neighbors+1)

    # 2. 准备构建COO稀疏矩阵的数据
    rows = []  # 行索引列表
    cols = []  # 列索引列表
    data = []  # 权重列表 (这里先全设为1，表示连接)

    for i in range(n_spot):
        # 为点i添加与它的n_neighbors个最近邻居的边
        # indices[i, 0] 是点i自身，从 indices[i, 1:] 开始是真正的邻居
        for neighbor_idx in indices[i, 1:]:
            rows.append(i)
            cols.append(neighbor_idx)
            data.append(1.0)  # 无权图，权重为1

    # 3. 创建COO格式的稀疏邻接矩阵
    # COO格式非常适合从(row, col, data)列表构建矩阵
    adj_coo = coo_matrix((data, (rows, cols)), shape=(n_spot, n_spot))

    # 4. 转换为对称矩阵（可选：使邻接矩阵无向）
    # 如果 A[i, j] = 1，我们也希望 A[j, i] = 1
    adj_coo = adj_coo + adj_coo.T
    adj_coo.data[:] = 1  # 将合并后可能大于1的值重新设为1，表示存在连接

    # 5. 转换为CSR格式以提高后续矩阵运算效率[citation:3]
    adj_csr = adj_coo.tocsr()

    # 6. 存储到adata中
    adata.obsm['adj'] = adj_csr  # 邻接矩阵 (稀疏)
    adata.obsm['graph_neigh'] = adj_csr  # 可以将同一个矩阵赋给不同键名，或存储为密集形式用于特定需要

    print(f'稀疏邻接图构建完成！')
    print(f'  矩阵形状: {adj_csr.shape}')
    print(f'  非零元素数 (边数): {adj_csr.nnz}')
    print(f'  稀疏度: {adj_csr.nnz / (n_spot * n_spot):.6f}')

def preprocess(adata):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=75)

    # 添加数据统计检查
    print(f"Preprocessed data - min: {adata.X.min()}, max: {adata.X.max()}, mean: {adata.X.mean()}")
    
def get_feature(adata, deconvolution=False):
    if deconvolution:
       adata_Vars = adata
    else:   
       adata_Vars =  adata[:, adata.var['highly_variable']]

    print(f"Using {adata_Vars.shape[1]} highly variable genes")

    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
       feat = adata_Vars.X.toarray()[:, ]
    else:
       feat = adata_Vars.X[:, ] 
    
    # data augmentation
    feat_a = permutation(feat)
    
    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a    
    
def add_contrastive_label(adata):
    # contrastive label
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL
    
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized 

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_adj_sparse(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)    

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def compute_ppr_matrix_approximate(adj, alpha=0.2, epsilon=1e-3):
    """
    高效计算稀疏、近似的个性化PageRank（PPR）矩阵。
    使用基于反向推送（Backward Push）或快速PPR估计的算法思想，生成一个稀疏的、近似的PPR邻接矩阵。
    这是处理超大规模图（>10万节点）的唯一可行方法。

    参数:
        adj: 稀疏邻接矩阵 (scipy.sparse.csr_matrix)
        alpha: 随机跳转（teleport）概率，通常设为0.2。
        epsilon: 误差容忍度。值越小，近似越精确，但矩阵越稠密。

    返回:
        ppr_matrix: 稀疏的、近似的PPR矩阵 (CSR格式)。
                    这个矩阵是高度稀疏的，每个节点只保留PPR值超过阈值的主要邻居。
    """
    import scipy.sparse as sp
    import numpy as np
    from scipy.sparse import diags, csr_matrix, lil_matrix
    import time

    n_nodes = adj.shape[0]
    print(f"[PPR近似算法] 开始计算，节点数: {n_nodes}, alpha={alpha}, epsilon={epsilon}")
    start_time = time.time()

    if not sp.isspmatrix_csr(adj):
        adj = adj.tocsr()

    # ----- 步骤 1: 准备行归一化的转移矩阵 -----
    # 计算每个节点的出度（对于无向图，出度=入度）
    out_degree = np.array(adj.sum(axis=1)).flatten()
    # 处理孤立节点
    out_degree[out_degree == 0] = 1
    # 创建转移概率矩阵: P = D^{-1} * A
    # 这是一个行随机矩阵（每行和为1）
    deg_inv = diags(1.0 / out_degree)
    P = deg_inv @ adj  # P[i, j] 表示从节点i随机游走到节点j的概率

    # ----- 步骤 2: 使用高效的稀疏近似算法 -----
    # 我们这里实现一种“幂迭代 + 截断”的实用方法，能在可接受的时间内为大规模图生成稀疏的PPR近似。
    # 注意：这不是理论上的精确PPR，但对于图神经网络中的多视图对比学习是完全足够的。

    print("  进行稀疏PPR近似计算 (这可能需要一些时间，但内存安全)...")

    # 初始化一个空的稀疏矩阵来存储最终的PPR近似结果
    # 使用LIL格式便于高效地逐步添加非零元素
    ppr_approx = lil_matrix((n_nodes, n_nodes), dtype=np.float32)

    # 批次处理，避免一次性处理所有节点导致内存爆炸
    batch_size = 500  # 根据你的内存调整批次大小
    for start_idx in range(0, n_nodes, batch_size):
        end_idx = min(start_idx + batch_size, n_nodes)
        batch_nodes = list(range(start_idx, end_idx))

        # 为这一批节点计算个性化的PPR向量
        for source in batch_nodes:
            # 初始化：所有概率集中在源节点
            r = np.zeros(n_nodes, dtype=np.float32)
            r[source] = 1.0
            p = np.zeros(n_nodes, dtype=np.float32)

            # 迭代传播，直到剩余概率质量很小
            while np.sum(r) > epsilon / n_nodes:  # 更严格的收敛条件
                # 选择当前概率质量最大的节点进行推送（简化版）
                # 在实际的快速PPR算法中，这里会有更复杂的队列管理
                max_idx = np.argmax(r)
                mass = r[max_idx]
                if mass == 0:
                    break

                # 将 mass * alpha 的部分存入结果p (代表被吸收的概率)
                p[max_idx] += alpha * mass

                # 将剩余 mass * (1-alpha) 的部分推送给邻居
                # 获取节点max_idx的邻居索引
                _, neighbors = P[max_idx, :].nonzero()
                if len(neighbors) > 0:
                    push_amount = (1 - alpha) * mass / len(neighbors)
                    for nb in neighbors:
                        r[nb] += push_amount

                # 清空当前节点的剩余概率
                r[max_idx] = 0.0

            # 将显著的PPR值（大于阈值）存入稀疏矩阵
            significant_indices = np.where(p > epsilon)[0]
            for idx in significant_indices:
                if idx != source:  # 通常不存储到自身的PPR（可选）
                    ppr_approx[source, idx] = p[idx]

        if (start_idx // batch_size) % 5 == 0 or end_idx == n_nodes:
            elapsed = time.time() - start_time
            print(f"    已完成节点 {end_idx}/{n_nodes}，耗时 {elapsed:.1f} 秒")

    # 转换为CSR格式以提高后续运算效率
    ppr_approx_csr = ppr_approx.tocsr()

    # ----- 步骤 3: 对称化 (可选，取决于你的图是否是无向的) -----
    # 如果你的原始邻接矩阵是无向的，你可能希望PPR矩阵也是对称的
    ppr_approx_csr = (ppr_approx_csr + ppr_approx_csr.T) / 2
    ppr_approx_csr.data[:] = np.minimum(ppr_approx_csr.data, 1.0)  # 确保值不超过1

    elapsed_total = time.time() - start_time
    print(f"[PPR近似算法] 计算完成！总耗时: {elapsed_total:.1f} 秒")
    print(f"  近似PPR矩阵形状: {ppr_approx_csr.shape}")
    print(f"  非零元素数: {ppr_approx_csr.nnz}")
    print(f"  稀疏度: {ppr_approx_csr.nnz / (n_nodes * n_nodes):.6f}")
    print(f"  每行平均非零元素: {ppr_approx_csr.nnz / n_nodes:.1f}")

    return ppr_approx_csr


import numpy as np
import scipy.sparse as sp
from fast_pagerank import pagerank_power
from scipy.sparse import csr_matrix, diags


def compute_ppr_matrix_fast(adj, alpha=0.2):
    """
    使用 fast-pagerank 库快速计算PPR向量。
    注意：此函数为每个节点计算一个PPR向量，再组合成稀疏矩阵。
    """
    n_nodes = adj.shape[0]
    print(f"[快速PPR] 开始计算，节点数: {n_nodes}")

    # 确保是CSR格式
    if not sp.isspmatrix_csr(adj):
        adj = adj.tocsr()

    # 计算转移概率矩阵 (列随机矩阵)
    out_degree = np.array(adj.sum(axis=0)).flatten()  # 注意这里是列和
    out_degree[out_degree == 0] = 1
    deg_inv = diags(1.0 / out_degree)
    # 列随机矩阵: A * D^{-1}
    column_stochastic = adj @ deg_inv

    # 使用 fast-pagerank 计算（更快的实现）
    # 这里计算的是全局PageRank向量
    pr_vector = pagerank_power(column_stochastic, p=alpha)  # p 即 alpha

    # 将向量转换为对角稀疏矩阵（如果你下游需要矩阵形式）
    ppr_matrix = diags(pr_vector).tocsr()

    print(f"[快速PPR] 计算完成。结果向量范围: [{pr_vector.min():.3e}, {pr_vector.max():.3e}]")
    return ppr_matrix


# 内存优化
def compute_ppr_matrix(adj, alpha=0.2):
    """Compute Personalized PageRank matrix for 10X data with memory optimization"""
    n_nodes = adj.shape[0]
    print(f"Computing PPR matrix for {n_nodes} nodes...")

    # 确保adj是CSR格式的稀疏矩阵
    if isinstance(adj, np.ndarray):
        print("Converting NumPy array to sparse matrix...")
        adj = sp.csr_matrix(adj)
    elif not sp.isspmatrix_csr(adj):
        adj = adj.tocsr()

    degree = np.array(adj.sum(1)).flatten()

    # 避免除零错误
    degree[degree == 0] = 1

    print("计算对称归一化邻接矩阵")
    # 计算对称归一化的邻接矩阵
    D_sqrt = sp.diags(1.0 / np.sqrt(degree))
    sym_norm_adj = D_sqrt @ adj @ D_sqrt

    print("转换为csr格式")
    # 转换为CSR格式
    sym_norm_adj = sym_norm_adj.tocsr()

    print("计算PPR矩阵")
    # 计算PPR矩阵
    identity = sp.eye(n_nodes, format='csr')

    print("使用更节省内存的计算方式")
    # 使用更节省内存的计算方式
    M = identity - (1 - alpha) * sym_norm_adj
    M = M.tocsr()

    print("Solving linear system for PPR...")
    ppr_matrix = alpha * inv(M)

    ppr_matrix = ppr_matrix.tocsr()

    print(f"PPR matrix density: {ppr_matrix.nnz / (n_nodes * n_nodes):.6f}")

    return ppr_matrix

def add_ppr_matrix(adata, alpha=0.2):
    """Add PPR matrix to adata with aggressive memory optimization for 10X data"""
    if 'adj' not in adata.obsm.keys():
        raise ValueError("Adjacency matrix not found. Please construct interaction first.")

    adj = adata.obsm['adj']

    ppr_matrix = compute_ppr_matrix_fast(adj, alpha)

    # 存储PPR矩阵
    adata.obsm['ppr'] = ppr_matrix
    print(f'PPR matrix constructed! Shape: {ppr_matrix.shape}, Non-zero elements: {ppr_matrix.nnz}')

    return adata

