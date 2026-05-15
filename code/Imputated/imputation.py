import os
import csv
import re
import time
import cv2
import pandas as pd
import numpy as np
import torch
from torch import nn
import json
import timm
from .util import *
from .contour_util import *
from .calculate_dis import *
from anndata import AnnData # 用于创建 AnnData 对象
from scipy.spatial import ConvexHull # 用于计算凸包
from shapely.geometry import Polygon, Point # 用于多边形判断
from scipy.spatial import cKDTree

# 加载模型
def load_model(model_path, config_path):
    """
    加载模型权重并构建模型。
    Args:
        model_path (str): 模型权重文件路径
        config_path (str): 模型配置文件路径
    Returns:
        nn.Module: 加载好的模型
        dict: 预训练配置
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found.")

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    architecture = config_dict.get("architecture")
    if not architecture:
        raise ValueError("config.json must contain 'architecture' key to specify the model architecture.")

    if architecture != "vit_huge_patch14_224":
        raise ValueError(f"Expected architecture 'vit_huge_patch14_224', but got '{architecture}'.")

    model_args = config_dict.get("model_args", {})
    pretrained_cfg = config_dict.get("pretrained_cfg", {})

    model = timm.create_model(
        architecture,
        pretrained=False,
        num_classes=model_args.get("num_classes", 0),
        img_size=model_args.get("img_size", 224),
        dynamic_img_size=model_args.get("dynamic_img_size", True),
        mlp_ratio=model_args.get("mlp_ratio", 4),
        init_values=model_args.get("init_values", 1e-5),
        global_pool=model_args.get("global_pool", ""),
        reg_tokens=model_args.get("reg_tokens", 0),
    )

    state_dict = torch.load(model_path, map_location="cpu")
    model_state_dict = model.state_dict()

    if "pos_embed" in state_dict:
        checkpoint_pos_embed = state_dict["pos_embed"]
        model_pos_embed = model_state_dict["pos_embed"]
        if checkpoint_pos_embed.shape[1] == 257 and model_pos_embed.shape[1] == 261:
            pos_embed_new = torch.zeros_like(model_pos_embed)
            pos_embed_new[:, :257, :] = checkpoint_pos_embed
            pos_embed_new[:, 257:, :] = checkpoint_pos_embed[:, -4:, :]
            state_dict["pos_embed"] = pos_embed_new

    for i in range(32):
        checkpoint_weight = state_dict.get(f"blocks.{i}.mlp.fc2.weight", None)
        if checkpoint_weight is not None and checkpoint_weight.shape == (1280, 3416):
            model_weight = model_state_dict[f"blocks.{i}.mlp.fc2.weight"]
            weight_new = torch.zeros_like(model_weight)
            weight_new[:, :3416] = checkpoint_weight
            weight_new[:, 3416:] = checkpoint_weight[:, -1:]
            state_dict[f"blocks.{i}.mlp.fc2.weight"] = weight_new

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, pretrained_cfg

# 提取图像特征的函数
def extract_embedding(x_pixel, y_pixel, image, model, beta, pretrained_cfg, device="cuda"):
    """
    使用深度学习模型提取图像区域的特征 embedding。
    """
    embeddings = []
    model = model.to(device)

    mean = np.array(pretrained_cfg.get("mean", [0.485, 0.456, 0.406]))
    std = np.array(pretrained_cfg.get("std", [0.229, 0.224, 0.225]))
    img_size = pretrained_cfg.get("input_size", [3, 224, 224])[1]
    print(f"Debug: img_size = {img_size}")  # 添加调试
    print(len(x_pixel), len(y_pixel), image.shape)
    index = 1

    for x, y in zip(x_pixel, y_pixel):
        if index % 1000 == 0:
            print(f"已经进行了{index}次提取")

        x_start = max(0, x - beta // 2)
        x_end = min(image.shape[0], x + beta // 2)
        y_start = max(0, y - beta // 2)
        y_end = min(image.shape[1], y + beta // 2)

        region = image[x_start:x_end, y_start:y_end]
        # if len(x_pixel) < 10000:
        #     print(f"Debug: region shape = {region.shape},index={index}")  # 添加调试

        region = cv2.resize(region, (img_size, img_size))
        region = region / 255.0
        region = region.transpose(2, 0, 1)
        region = (region - mean[:, None, None]) / std[:, None, None]
        region = torch.tensor(region, dtype=torch.float32).unsqueeze(0)

        region = region.to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(region)

        class_token = output[:, 0]
        patch_tokens = output[:, 1:]
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        embedding = embedding.to(torch.float16).cpu().numpy()
        embeddings.append(embedding.flatten())

        index += 1

    return np.array(embeddings)

def imputation(img, raw, cnt, genes, shape="None", res=50, s=1, k=2, num_nbs=10):
    # 加载深度学习模型
    model_path = "Virchow2/pytorch_model.bin"
    config_path = "Virchow2/config.json"
    model, pretrained_cfg = load_model(model_path, config_path)

    # ------------------------------------Known points Initialization and Sequenced Area Definition---------------------------------#
    known_adata = raw[:, raw.var.index.isin(genes)]
    known_adata.obs["x"] = known_adata.obs["pixel_x"]
    known_adata.obs["y"] = known_adata.obs["pixel_y"]
    known_coords = known_adata.obs[["x", "y"]].values
    # 1. 定义测序区域边界 (Sequenced Area Polygon) - 使用 Convex Hull
    try:
        if known_coords.shape[0] < 3:
            # 如果点太少，用异常来触发 Bounding Box 模式
            raise ValueError("Not enough points to compute Convex Hull.")

        hull = ConvexHull(known_coords)
        # 将凸包顶点坐标转换为 Shapely Polygon 对象
        sequenced_area_polygon = Polygon(known_coords[hull.vertices])
        print(f"Sequenced area defined by Convex Hull of {known_coords.shape[0]} known points.")
    except Exception as e:
        # 如果 Convex Hull 失败（点共线，点数太少等），则使用简单的 Bounding Box
        print(f"Warning: Could not calculate convex hull ({e}). Falling back to Bounding Box for sequenced area.")
        x_min_k, x_max_k = known_coords[:, 0].min(), known_coords[:, 0].max()
        y_min_k, y_max_k = known_coords[:, 1].min(), known_coords[:, 1].max()
        # 扩展 Bounding Box 1个 res，确保边界上的伪点被包含
        x_min_k -= res
        x_max_k += res
        y_min_k -= res
        y_max_k += res
        sequenced_area_polygon = Polygon([
            (x_min_k, y_min_k),
            (x_max_k, y_min_k),
            (x_max_k, y_max_k),
            (x_min_k, y_max_k)
        ])

    # 创建二值掩码
    binary = np.zeros((img.shape[0:2]), dtype=np.uint8)
    cv2.drawContours(binary, [cnt], -1, (1), thickness=-1)

    # 放大轮廓并创建二值掩码
    cnt_enlarged = scale_contour(cnt, 1.05)
    binary_enlarged = np.zeros(img.shape[0:2])
    cv2.drawContours(binary_enlarged, [cnt_enlarged], -1, (1), thickness=-1)

    # 生成伪点坐标
    x_max, y_max = img.shape[0], img.shape[1]
    x_list = list(range(int(res), x_max, int(res)))
    y_list = list(range(int(res), y_max, int(res)))
    x = np.repeat(x_list, len(y_list)).tolist()
    y = y_list * len(x_list)
    sudo = pd.DataFrame({"x": x, "y": y})
    # 原始过滤：基于放大后的组织轮廓 (确保点在组织内)
    initial_count = sudo.shape[0]
    tissue_filtered_indices = [i for i in sudo.index if (binary_enlarged[sudo.x[i], sudo.y[i]] != 0)]
    sudo_tissue_filtered = sudo.loc[tissue_filtered_indices].reset_index(drop=True)
    print(f"Initial sudo spots generated: {initial_count}. Filtered by tissue contour: {sudo_tissue_filtered.shape[0]}")

    # 【新增】二次过滤：只保留在测序区域 (sequenced_area_polygon) 内的伪点

    # 使用 Shapely Point 和 Polygon.contains() 进行判断
    def check_in_polygon(row):
        return sequenced_area_polygon.contains(Point(row['x'], row['y']))

    in_sequenced_area = sudo_tissue_filtered.apply(check_in_polygon, axis=1)
    sudo = sudo_tissue_filtered[in_sequenced_area].reset_index(drop=True)

    print(f"Final number of sudo points after filtering to sequenced area: {sudo.shape[0]}")
    # 提取伪点的特征嵌入
    b = res
    embeddings = extract_embedding(
        x_pixel=sudo.x.tolist(),
        y_pixel=sudo.y.tolist(),
        image=img,
        model=model,
        beta=b,
        pretrained_cfg=pretrained_cfg
    )

    # 创建 sudo_adata 并存储嵌入
    sudo_adata = AnnData(np.zeros((sudo.shape[0], len(genes))))
    sudo_adata.obs = sudo
    sudo_adata.var = pd.DataFrame(index=genes)
    sudo_adata.obsm["embedding"] = embeddings
    z_scale = np.max([np.std(sudo.x), np.std(sudo.y)]) * s
    sudo_adata.obs["z_raw"] = np.mean(embeddings, axis=1) # 使用嵌入均值作为 z

    # ------------------------------------Known points processing continued---------------------------------#
    print("   [Debug]   Known points的pixel坐标")
    print(known_adata.obs["x"])
    print(known_adata.obs["y"])
    # 检查坐标是否在图像范围内
    x_min = known_adata.obs["x"].min()
    x_max = known_adata.obs["x"].max()
    y_min = known_adata.obs["y"].min()
    y_max = known_adata.obs["y"].max()
    print(f"   [Debug]   已知点X坐标范围: {x_min} ~ {x_max}")
    print(f"   [Debug]   已知点Y坐标范围: {y_min} ~ {y_max}")

    # 如果坐标有超出图像边界的，打印警告
    if x_min < 0 or x_max >= img.shape[0] or y_min < 0 or y_max >= img.shape[1]:
        print("   [Warning] 部分已知点坐标超出图像边界，这可能导致特征提取失败。")
    embeddings_known = extract_embedding(
        x_pixel=known_adata.obs["pixel_x"].astype(int).tolist(),
        y_pixel=known_adata.obs["pixel_y"].astype(int).tolist(),
        image=img,
        model=model,
        beta=b,
        pretrained_cfg=pretrained_cfg
    )
    known_adata.obsm["embedding"] = embeddings_known
    known_adata.obs["z_raw"] = np.mean(embeddings_known, axis=1)

    # ------------------------------------Normalization and Scaling (Critical Fix)---------------------------------#

    # 1. 组合所有数据点的 X, Y, Z_raw
    all_x = np.concatenate([known_adata.obs["x"].values, sudo_adata.obs["x"].values])
    all_y = np.concatenate([known_adata.obs["y"].values, sudo_adata.obs["y"].values])
    all_z_raw = np.concatenate([known_adata.obs["z_raw"].values, sudo_adata.obs["z_raw"].values])

    # 2. 计算 Min/Max
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    z_min, z_max = all_z_raw.min(), all_z_raw.max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    # 3. 归一化函数
    def normalize_coord(data, min_val, range_val):
        if range_val == 0:
            return np.zeros_like(data)
        return (data - min_val) / range_val

    # 4. 对 Known points 和 Sudo points 进行归一化
    # X, Y 归一化 (空间坐标)
    known_adata.obs["x_norm"] = normalize_coord(known_adata.obs["x"].values, x_min, x_range)
    known_adata.obs["y_norm"] = normalize_coord(known_adata.obs["y"].values, y_min, y_range)
    sudo_adata.obs["x_norm"] = normalize_coord(sudo_adata.obs["x"].values, x_min, x_range)
    sudo_adata.obs["y_norm"] = normalize_coord(sudo_adata.obs["y"].values, y_min, y_range)

    # Z 归一化 (形态学特征)
    z_norm_known = normalize_coord(known_adata.obs["z_raw"].values, z_min, z_range)
    z_norm_sudo = normalize_coord(sudo_adata.obs["z_raw"].values, z_min, z_range)

    # 5. 最终的 Z 坐标 = 归一化 Z 乘以 形态学权重 s
    known_adata.obs["z"] = z_norm_known * s
    sudo_adata.obs["z"] = z_norm_sudo * s

    # ----------------------- 使用KD树进行高效最近邻搜索，避免全距离矩阵 -----------------------#
    start_time = time.time()

    # 1. 准备已知点的3D坐标数组 (用于构建KD树)
    known_coords_3d = np.column_stack([
        known_adata.obs["x_norm"].values,
        known_adata.obs["y_norm"].values,
        known_adata.obs["z"].values
    ])

    # 2. 准备伪点的3D坐标数组 (用于查询)
    sudo_coords_3d = np.column_stack([
        sudo_adata.obs["x_norm"].values,
        sudo_adata.obs["y_norm"].values,
        sudo_adata.obs["z"].values
    ])

    print(f"构建KD树 (基于 {known_coords_3d.shape[0]} 个已知点)...")
    # 为已知点构建KD树索引
    kdtree = cKDTree(known_coords_3d)

    print(f"为 {sudo_coords_3d.shape[0]} 个伪点查询最近的 {num_nbs} 个邻居...")
    # 关键：一次性查询所有伪点的最近邻居
    # distances: 形状 (sudo_adata.shape[0], num_nbs)，每个伪点到其k个最近邻居的距离
    # indices:   形状 (sudo_adata.shape[0], num_nbs)，每个伪点的k个最近邻居在 known_adata 中的索引
    distances, indices = kdtree.query(sudo_coords_3d, k=num_nbs, workers=-1)  # workers=-1 使用所有CPU核心

    print(f"KD树查询完成！总耗时: {time.time() - start_time:.2f} 秒")
    print(f"  距离数组形状: {distances.shape} (占用内存约 {distances.nbytes / 1024 ** 2:.1f} MB)")
    print(f"  索引数组形状: {indices.shape} (占用内存约 {indices.nbytes / 1024 ** 2:.1f} MB)")

    # ------------------------- 准备基因表达矩阵用于插值 -------------------------#
    print("检查已知点基因表达矩阵格式...")
    import scipy.sparse as sp

    # 检查known_adata.X的格式
    is_sparse = sp.issparse(known_adata.X)
    print(f"  known_adata.X 格式: {'稀疏矩阵' if is_sparse else '密集矩阵'}")

    # 如果是稀疏矩阵，考虑转换为密集格式以提高插值速度
    n_spots, n_genes = known_adata.shape
    dense_memory_mb = n_spots * n_genes * 4 / 1024 ** 2  # float32的内存估算

    if is_sparse:
        print(f"  稀疏矩阵转换为密集将占用约 {dense_memory_mb:.1f} MB 内存")
        if dense_memory_mb < 2000:  # 如果小于2GB，转换是可行的
            print("  正在转换为密集格式以提高插值速度...")
            known_adata_dense = known_adata.X.toarray()
            use_dense = True
        else:
            print("  内存占用较大，保持稀疏格式但插值时需转换子矩阵")
            use_dense = False
    else:
        print(f"  已经是密集格式，内存占用约 {dense_memory_mb:.1f} MB")
        use_dense = False  # 不需要转换

    # ------------------------- 使用查询结果进行插值 -------------------------#
    print(f"开始基于最近邻进行基因表达插值...")
    for i in range(sudo_adata.shape[0]):
        if i % 10000 == 0:  # 每处理10000个点打印一次进度
            print(f"  插值进度: {i}/{sudo_adata.shape[0]} ({(i / sudo_adata.shape[0]) * 100:.1f}%)")

        # 从KD树查询结果中直接获取当前伪点的邻居信息
        neighbor_dists = distances[i]  # 当前伪点到k个邻居的距离，形状 (num_nbs,)
        neighbor_indices = indices[i]  # 这k个邻居在 known_adata 中的行索引，形状 (num_nbs,)

        # 处理可能出现的零距离（完全相同的位置）
        # 为了防止分母过小且不稳定，增大 epsilon$ 可以为分母设置一个更高的基准，从而限制最近邻点权重的绝对上限。
        min_dist = np.min(neighbor_dists)
        epsilon = 0.01
        if min_dist < epsilon:  # 如果距离几乎为零
            # 直接使用最近点的表达值
            closest_idx = neighbor_indices[np.argmin(neighbor_dists)]
            if use_dense:
                sudo_adata.X[i, :] = known_adata_dense[closest_idx, :]
            else:
                sudo_adata.X[i, :] = known_adata.X[closest_idx, :].toarray() if is_sparse else known_adata.X[
                                                                                               closest_idx, :]
            continue

        # --- 权重计算 (与原逻辑一致) ---
        # 归一化距离：相对于最近点的距离
        dis_tmp = neighbor_dists / min_dist

        if isinstance(k, int):
            # 逆距离加权 (IDW)
            weights = (1.0 / (dis_tmp ** k))
        else:
            # 径向基函数 (RBF)
            weights = np.exp(-dis_tmp)

        # 归一化权重，使和为1
        weights = weights / np.sum(weights)

        # --- 加权平均进行插值 ---
        # 根据矩阵格式获取邻居的表达矩阵
        if use_dense:
            # 使用预先转换的密集矩阵
            neighbor_expressions = known_adata_dense[neighbor_indices, :]
        elif is_sparse:
            # 稀疏矩阵：索引后转换为密集子矩阵
            neighbor_expressions = known_adata.X[neighbor_indices, :].toarray()
        else:
            # 已经是密集矩阵，直接索引
            neighbor_expressions = known_adata.X[neighbor_indices, :]

        # 计算加权平均：weights (1D) 与 neighbor_expressions (2D) 的点积
        sudo_adata.X[i, :] = np.dot(weights, neighbor_expressions)

    print(f"基因表达插值完成！")
    return sudo_adata


    # # -----------------------Distance matrix between sudo and known points-------------#
    # start_time = time.time()
    # dis = np.zeros((sudo_adata.shape[0], known_adata.shape[0]))
    # # 距离矩阵现在只需计算 (过滤后的 sudo_adata.shape[0]) x (known_adata.shape[0]) 这么大的矩阵
    # # 使用归一化后的 X_norm, Y_norm 和 缩放后的 Z 坐标
    # x_sudo, y_sudo, z_sudo = sudo_adata.obs["x_norm"].values, sudo_adata.obs["y_norm"].values, sudo_adata.obs[
    #     "z"].values
    # x_known, y_known, z_known = known_adata.obs["x_norm"].values, known_adata.obs["y_norm"].values, known_adata.obs[
    #     "z"].values
    #
    #
    # print("Total number of sudo points for imputation: ", sudo_adata.shape[0])
    # for i in range(sudo_adata.shape[0]):
    #     if i % 100 == 0:
    #         print("Calculating spot", i)
    #     # cord1 现在是一个 3D 坐标 [X_norm, Y_norm, Z_scaled]
    #     cord1 = np.array([x_sudo[i], y_sudo[i], z_sudo[i]])
    #     for j in range(known_adata.shape[0]):
    #         cord2 = np.array([x_known[j], y_known[j], z_known[j]])
    #         # 距离计算：D = sqrt(ΔX_norm² + ΔY_norm² + ΔZ_scaled²)
    #         dis[i][j] = distance(cord1, cord2)
    # print("--- %s seconds ---" % (time.time() - start_time))
    # dis = pd.DataFrame(dis, index=sudo_adata.obs.index, columns=known_adata.obs.index)
    #
    # # -------------------------Fill gene expression using nbs---------------------------#
    # for i in range(sudo_adata.shape[0]):
    #     if i % 100 == 0:
    #         print("Imputing spot", i)
    #     index = sudo_adata.obs.index[i]
    #     dis_tmp = dis.loc[index, :].sort_values()
    #
    #     # 使用 num_nbs 个最近的邻居
    #     nbs = dis_tmp[0:num_nbs]
    #
    #     # 归一化距离，避免除以零
    #     dis_tmp = (nbs.to_numpy() + 1e-6) / np.min(nbs.to_numpy() + 1e-6)  # 避免 0 距离，使用更小的 epsilon
    #
    #     # 使用 IDW 幂次 k=5 (或用户设置的 k)
    #     if isinstance(k, int):
    #         # IDW 权重计算
    #         weights = ((1 / (dis_tmp ** k)) / ((1 / (dis_tmp ** k)).sum()))
    #     else:
    #         # 径向基函数（RBF）权重计算 (如果 k 不是整数)
    #         weights = np.exp(-dis_tmp) / np.sum(np.exp(-dis_tmp))
    #
    #     row_index = [known_adata.obs.index.get_loc(i) for i in nbs.index]
    #     # 加权平均插值
    #     sudo_adata.X[i, :] = np.dot(weights, known_adata.X[row_index, :])
    #
    # return sudo_adata