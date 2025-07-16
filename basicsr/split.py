import os
import cv2
import numpy as np
from tqdm import tqdm


def find_focus_from_semantic(depth_img, semantic_mask):
    """根据语义分割的主体部分确定聚焦位置和深度"""
    if semantic_mask is None:
        # 如果没有语义mask，使用深度图分析选择焦点区域
        return find_focus_from_depth(depth_img)

    # 确保语义mask是二值化的
    if len(semantic_mask.shape) > 2:
        semantic_mask = cv2.cvtColor(semantic_mask, cv2.COLOR_BGR2GRAY)
    _, semantic_mask_bin = cv2.threshold(semantic_mask, 127, 255, cv2.THRESH_BINARY)

    # 查找主体区域
    main_subject_mask = semantic_mask_bin > 0

    # 如果没有主体区域，使用深度图分析选择焦点区域
    if np.sum(main_subject_mask) == 0:
        return find_focus_from_depth(depth_img)

    # 计算主体区域的深度均值作为焦点深度
    if len(depth_img.shape) > 2:
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)

    # 计算主体区域重心和平均深度
    y, x = np.where(main_subject_mask)
    center_y = int(np.mean(y))
    center_x = int(np.mean(x))

    # 获取主体区域的平均深度
    subject_depths = depth_img[main_subject_mask]
    focal_depth = np.mean(subject_depths)

    return (center_y, center_x), focal_depth


def find_focus_from_depth(depth_img):
    """当语义掩码缺失时，通过深度分析选择焦点区域"""
    # 确保深度图是单通道
    if len(depth_img.shape) > 2:
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)

    # 预处理深度图（反转深度值，使深度越深像素值越小）
    depth_inverted = cv2.normalize(255 - depth_img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # 应用边缘保留平滑
    depth_smoothed = cv2.bilateralFilter(depth_inverted, d=9, sigmaColor=0.3, sigmaSpace=5)

    # 1. 选择深度值居中的区域（避免过近或过远）
    depth_values = depth_smoothed.flatten()
    lower_thresh = np.percentile(depth_values, 30)  # 排除最远的30%
    upper_thresh = np.percentile(depth_values, 70)  # 排除最近的30%

    # 创建候选区域掩码
    candidate_mask = (depth_smoothed >= lower_thresh) & (depth_smoothed <= upper_thresh)

    # 2. 连通分量分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidate_mask.astype(np.uint8), connectivity=8)

    if num_labels < 2:  # 如果没有连通区域，使用整个图像中心
        h, w = depth_img.shape[:2]
        return (h // 2, w // 2), np.mean(depth_smoothed)

    # 3. 选择面积最大的连通域（跳过背景标签0）
    max_area_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    main_region_mask = (labels == max_area_idx)

    # 4. 计算该区域的质心和平均深度
    y, x = np.where(main_region_mask)
    center_y = int(np.mean(y))
    center_x = int(np.mean(x))
    focal_depth = np.mean(depth_smoothed[main_region_mask])

    return (center_y, center_x), focal_depth


def calculate_balanced_thresholds(depth_values):
    """计算平衡三个区域面积的阈值，确保过渡区域占比不小于15%"""
    # 计算直方图
    hist, bins = np.histogram(depth_values, bins=256, range=(0, 1))

    # 计算累积分布
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]

    # 设置合理的区域比例，增加过渡区比例
    near_ratio = 0.35  # 近景比例
    far_ratio = 0.35   # 远景比例
    transition_ratio = 0.3  # 过渡区比例，确保不小于15%

    # 确保比例总和为1
    total = near_ratio + far_ratio + transition_ratio
    near_ratio /= total
    far_ratio /= total
    transition_ratio /= total

    # 计算阈值
    near_thresh = bins[np.searchsorted(cdf_normalized, near_ratio)]
    far_thresh = bins[np.searchsorted(cdf_normalized, 1 - far_ratio)]

    # 确保阈值之间有合适距离
    min_transition_size = 0.2  # 确保过渡区足够大
    if far_thresh - near_thresh < min_transition_size:
        mid_point = (near_thresh + far_thresh) / 2
        near_thresh = mid_point - min_transition_size/2
        far_thresh = mid_point + min_transition_size/2

    return near_thresh, far_thresh


def create_depth_mask(depth_img, semantic_mask=None):
    """创建景深mask，确保过渡区域占比不小于15%"""
    # 确保深度图是单通道
    if len(depth_img.shape) > 2:
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)

    # 预处理深度图（反转深度值，使深度越深像素值越小）
    depth_inverted = cv2.normalize(255 - depth_img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # 应用边缘保留平滑
    depth_smoothed = cv2.bilateralFilter(depth_inverted, d=9, sigmaColor=0.3, sigmaSpace=5)

    # 使用语义分割主体部分确定焦点位置和深度
    _, focal_depth_normalized = find_focus_from_semantic(depth_smoothed, semantic_mask)

    # 获取所有深度值并计算平衡阈值
    depth_values = depth_smoothed.flatten()
    near_thresh, far_thresh = calculate_balanced_thresholds(depth_values)

    # 调整阈值使焦点在过渡区中心附近
    mid_point = (near_thresh + far_thresh) / 2
    shift = mid_point - focal_depth_normalized

    # 移动阈值窗口，但保持窗口大小不变
    if abs(shift) > 0.05:  # 只有当偏移明显时才调整
        near_thresh -= shift
        far_thresh -= shift

    # 确保阈值在有效范围内
    if near_thresh < 0:
        far_thresh -= near_thresh  # 保持窗口大小
        near_thresh = 0
    if far_thresh > 1:
        near_thresh -= (far_thresh - 1)  # 保持窗口大小
        far_thresh = 1

    # 确保不会因为上述调整导致near_thresh为负
    near_thresh = max(0, near_thresh)

    # 验证三个区域都不为空
    near_pixels = np.sum(depth_smoothed <= near_thresh)
    far_pixels = np.sum(depth_smoothed >= far_thresh)
    transition_pixels = np.sum((depth_smoothed > near_thresh) & (depth_smoothed < far_thresh))

    total_pixels = depth_smoothed.size
    min_area_ratio = 0.15  # 每个区域至少占总面积的15%

    # 如果某区域过小，重新调整阈值
    if (near_pixels / total_pixels < min_area_ratio or
            far_pixels / total_pixels < min_area_ratio or
            transition_pixels / total_pixels < min_area_ratio):
        # 使用百分位数直接划分
        sorted_depths = np.sort(depth_values)

        # 调整比例确保过渡区至少占15%
        trans_ratio = max(1 - 2 * min_area_ratio, min_area_ratio)  # 确保过渡区占比
        near_thresh = sorted_depths[int(total_pixels * min_area_ratio)]
        far_thresh = sorted_depths[int(total_pixels * (1 - min_area_ratio))]

        # 迭代调整阈值以确保各区域占比合理
        max_iterations = 20
        for _ in range(max_iterations):
            # 重新计算区域占比
            near_pixels = np.sum(depth_smoothed <= near_thresh)
            far_pixels = np.sum(depth_smoothed >= far_thresh)
            transition_pixels = np.sum((depth_smoothed > near_thresh) & (depth_smoothed < far_thresh))

            near_ratio = near_pixels / total_pixels
            far_ratio = far_pixels / total_pixels
            transition_ratio = transition_pixels / total_pixels

            # 如果所有区域都满足最小比例要求，退出循环
            if (near_ratio >= min_area_ratio and
                    far_ratio >= min_area_ratio and
                    transition_ratio >= min_area_ratio):
                break

            # 重点关注过渡区比例，确保不小于15%
            if transition_ratio < min_area_ratio:
                # 调整near和far阈值，增大过渡区
                near_thresh = sorted_depths[int(total_pixels * (min_area_ratio - 0.05))]
                far_thresh = sorted_depths[int(total_pixels * (1 - min_area_ratio + 0.05))]
            else:
                # 正常调整各区域
                if near_ratio < min_area_ratio:
                    near_thresh = sorted_depths[int(total_pixels * min_area_ratio)]
                if far_ratio < min_area_ratio:
                    far_thresh = sorted_depths[int(total_pixels * (1 - min_area_ratio))]

    # 生成单通道mask
    mask = np.zeros(depth_smoothed.shape, dtype=np.float32)
    mask[depth_smoothed <= near_thresh] = 1.0  # 近景区
    mask[depth_smoothed >= far_thresh] = 0.0  # 远景区

    # 处理过渡区域，使用平滑过渡
    transition = (depth_smoothed > near_thresh) & (depth_smoothed < far_thresh)
    if np.any(transition):
        # 使用线性过渡
        transition_depth = depth_smoothed[transition]
        transition_range = max(far_thresh - near_thresh, 0.01)  # 避免除零错误
        normalized_transition = (far_thresh - transition_depth) / transition_range
        mask[transition] = normalized_transition

    # 优化区域连续性，减少噪点
    mask_u8 = (mask * 255).astype(np.uint8)

    # 使用较小的结构元素，避免过度扩张主区域
    kernel_small = np.ones((3, 3), np.uint8)

    # 对不同区域分别进行形态学操作
    near_region = np.zeros_like(mask_u8)
    near_region[mask_u8 > 230] = 255  # 提高阈值，减小近景区
    near_region = cv2.morphologyEx(near_region, cv2.MORPH_CLOSE, kernel_small)

    far_region = np.zeros_like(mask_u8)
    far_region[mask_u8 < 25] = 255  # 降低阈值，减小远景区
    far_region = cv2.morphologyEx(far_region, cv2.MORPH_CLOSE, kernel_small)

    # 重新组合mask
    mask = np.zeros_like(depth_smoothed, dtype=np.float32)
    mask[near_region > 0] = 1.0
    mask[far_region > 0] = 0.0

    # 重新计算过渡区，增大过渡区域
    transition = (near_region == 0) & (far_region == 0)


    mask[transition] = 0.5

    # 如果有语义分割mask，确保其区域完全包含在主体中
    if semantic_mask is not None:
        # 确保语义mask是二值化的
        if len(semantic_mask.shape) > 2:
            semantic_mask = cv2.cvtColor(semantic_mask, cv2.COLOR_BGR2GRAY)
        _, semantic_mask = cv2.threshold(semantic_mask, 127, 255, cv2.THRESH_BINARY)

        # 将语义mask区域强制设为1.0（主体区域）
        mask[semantic_mask > 0] = 1.0

        # 在语义区域边缘创建适度的过渡区
        kernel_edge = np.ones((7, 7), np.uint8)  # 增大边缘过渡区
        sem_dilated = cv2.dilate(semantic_mask, kernel_edge)
        sem_border = cv2.subtract(sem_dilated, semantic_mask)

        # 仅当非主体区域时才设置边缘过渡
        border_area = (sem_border > 0) & (semantic_mask == 0) & (mask < 1.0)

        mask[border_area] = 0.5

    # 最后验证各区域占比
    near_percent = np.sum(mask > 0.9) / mask.size * 100
    far_percent = np.sum(mask < 0.1) / mask.size * 100
    transition_percent = np.sum((mask >= 0.1) & (mask <= 0.9)) / mask.size * 100

    print(f"区域占比 - 近景: {near_percent:.1f}%, 过渡: {transition_percent:.1f}%, 远景: {far_percent:.1f}%")

    # # 确保每个区域至少占15%
    # min_area_percent = 15.0
    # assert near_percent > min_area_percent and transition_percent > min_area_percent and far_percent > min_area_percent, \
    #     f"区域占比不满足要求 - 近景: {near_percent:.1f}%, 过渡: {transition_percent:.1f}%, 远景: {far_percent:.1f}%"

    return mask


def match_rgb_depth_pairs(depth_folder, semantic_folder=None):
    """匹配RGB、深度图和语义分割mask文件"""
    depth_files = [f for f in os.listdir(depth_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    semantic_files = []
    if semantic_folder is not None:
        semantic_files = [f for f in os.listdir(semantic_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    pairs = []
    for depth_file in depth_files:
        base_name = os.path.splitext(depth_file)[0]

        possible_semantic_names = [
            f"{base_name}.png", f"{base_name}.jpg",
            f"{base_name}_mask.png", f"{base_name}_seg.png",
            f"mask_{base_name}.png", f"seg_{base_name}.png"
        ]

        # 查找匹配的语义mask
        semantic_match = None
        if semantic_folder is not None:
            for semantic_name in possible_semantic_names:
                if semantic_name in semantic_files:
                    semantic_match = semantic_name
                    break

        if semantic_match:
            pairs.append((depth_file, semantic_match))
        else:
            pairs.append((depth_file, None))

    return pairs


def process_separate_folders(depth_folder, output_folder, semantic_folder=None):
    os.makedirs(output_folder, exist_ok=True)
    file_pairs = match_rgb_depth_pairs(depth_folder, semantic_folder)

    for depth_file, semantic_file in tqdm(file_pairs, desc="处理进度"):
        depth_path = os.path.join(depth_folder, depth_file)

        semantic_mask = None
        if semantic_file is not None and semantic_folder is not None:
            semantic_path = os.path.join(semantic_folder, semantic_file)
            semantic_mask = cv2.imread(semantic_path, cv2.IMREAD_GRAYSCALE)

        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

        if depth_img is None:
            print(f"警告: 无法读取 {depth_file}")
            continue

        mask = create_depth_mask(depth_img, semantic_mask)

        base_name = os.path.splitext(depth_file)[0]
        output_path = os.path.join(output_folder, f"{base_name}.png")
        cv2.imwrite(output_path, (mask * 255).astype(np.uint8))


if __name__ == "__main__":
    depth_dir = "/Users/ck/Desktop/code/basicsr/VABM/train/depth"
    output_dir = "/Users/ck/Desktop/code/basicsr/VABM/train/tmp"
    semantic_dir = "/Users/ck/Desktop/code/basicsr/VABM/train/masks"  # 语义分割mask文件夹路径

    process_separate_folders(depth_dir, output_dir, semantic_folder=semantic_dir)
    # process_separate_folders(depth_dir, output_dir)
    print("批处理完成！结果保存在:", os.path.abspath(output_dir))