import os
import torch
import json
import collections

def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU (Intersection over Union)。
    框格式: [x_center, y_center, width, height] (归一化坐标)
    """
    # 将中心点宽高格式转换为 (x1, y1, x2, y2)
    b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    # 获取交集区域的坐标
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    # 计算交集区域的面积
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # 计算两个框的面积
    b1_area = box1[2] * box1[3]
    b2_area = box2[2] * box2[3]

    # 计算并集区域的面积
    union_area = b1_area + b2_area - inter_area

    # 避免除以零
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def load_labels_from_yolo_txt(filepath):
    """
    从YOLO格式的txt文件中加载标签。
    格式: [[class_id, x_center, y_center, width, height], ...]
    不假设有置信度，直接加载前5个元素。
    """
    labels = []
    if not os.path.exists(filepath):
        return labels
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) >= 5: # 只取前5个元素 (class_id, x, y, w, h)
                    labels.append(parts[:5])
    except Exception as e:
        print(f"警告: 读取文件 {filepath} 时出错: {e}")
    return labels

def load_class_names(classes_path):
    """从.txt或.json文件加载类别名称。"""
    class_names = []
    if not classes_path or not os.path.exists(classes_path):
        print(f"警告: 未找到类别文件或路径无效: {classes_path}。将使用 'Class ID' 显示类别。")
        return class_names

    try:
        if classes_path.lower().endswith('.json'):
            with open(classes_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    class_names = data
                elif isinstance(data, dict):
                    class_names = [data[str(i)] for i in sorted([int(k) for k in data.keys()])]
                else:
                    print(f"警告：JSON文件 {classes_path} 格式不支持。请确保是列表或键为数字的字典。")
        elif classes_path.lower().endswith('.txt'):
            with open(classes_path, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f.readlines()]
        else:
            print(f"警告：不支持的类别文件格式 {os.path.basename(classes_path)}。只支持 .json 或 .txt。")

    except json.JSONDecodeError as e:
        print(f"错误：解析JSON类别文件 {classes_path} 时发生错误：{e}")
    except Exception as e:
        print(f"加载类别文件 {classes_path} 时发生错误：{e}")
    return class_names

def calculate_recall(
    predictions_root_dir,
    labels_root_dir,
    iou_threshold=0.3, # 保持默认 IoU 阈值
    classes_path=None
):
    """
    计算给定目录下所有视频帧的Recall率。
    以predictions目录为基准遍历，预测框无需再进行置信度过滤。
    """
    print(f"\n--- 开始计算 Recall (IoU >= {iou_threshold}) ---")

    total_true_positives = 0
    total_false_negatives = 0
    class_wise_tp = collections.defaultdict(int)
    class_wise_fn = collections.defaultdict(int)

    class_names = load_class_names(classes_path)

    # 遍历 predictions 目录下的所有 videoX 子文件夹
    video_folders = [f for f in os.listdir(predictions_root_dir) if os.path.isdir(os.path.join(predictions_root_dir, f)) and f.startswith('video')]
    video_folders.sort()

    if not video_folders:
        print(f"在 {predictions_root_dir} 中没有找到任何'video'开头的子文件夹。请检查路径和目录结构。")
        return 0, {}

    for video_folder in video_folders:
        current_pred_dir = os.path.join(predictions_root_dir, video_folder)
        current_label_dir = os.path.join(labels_root_dir, video_folder)

        # 遍历当前视频文件夹下的所有预测标签文件
        pred_files = [f for f in os.listdir(current_pred_dir) if f.lower().endswith('.txt')]
        pred_files.sort()

        for pred_file in pred_files:
            base_name = os.path.splitext(pred_file)[0]

            pred_filepath = os.path.join(current_pred_dir, pred_file)
            gt_filepath = os.path.join(current_label_dir, f"{base_name}.txt") # 对应的真实标签文件

            preds = load_labels_from_yolo_txt(pred_filepath)
            gts = load_labels_from_yolo_txt(gt_filepath)

            # True Positives (TP) 和 False Negatives (FN) 计数
            gt_matched = [False] * len(gts) # 标记每个真实框是否已被匹配
            pred_used = [False] * len(preds) # 标记每个预测框是否已被使用

            # 遍历真实框，尝试寻找匹配的预测框
            for i, gt_box_data in enumerate(gts):
                gt_class_id = int(gt_box_data[0])
                best_iou = 0
                best_pred_idx = -1

                for j, pred_box_data in enumerate(preds):
                    if pred_used[j]: # 如果这个预测框已经被匹配过了，跳过
                        continue

                    pred_class_id = int(pred_box_data[0])

                    # 类别必须一致
                    if gt_class_id == pred_class_id:
                        # 框坐标是第1到第4个元素 [class_id, x, y, w, h]
                        iou = calculate_iou(gt_box_data[1:5], pred_box_data[1:5])
                        if iou > best_iou:
                            best_iou = iou
                            best_pred_idx = j

                # 如果找到了满足IoU阈值的最佳匹配预测框
                if best_iou >= iou_threshold:
                    total_true_positives += 1
                    class_wise_tp[gt_class_id] += 1
                    gt_matched[i] = True # 标记此真实框已匹配
                    pred_used[best_pred_idx] = True # 标记此预测框已使用
                else:
                    # 如果真实框没有找到匹配的预测框（或者IoU不够），则认为是 False Negative
                    total_false_negatives += 1
                    class_wise_fn[gt_class_id] += 1


    overall_recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0

    print(f"\n--- 计算结果 ---")
    print(f"总真实正例 (TP): {total_true_positives}")
    print(f"总假反例 (FN): {total_false_negatives}")
    print(f"总体召回率 (Overall Recall @ IoU={iou_threshold:.1f}): {overall_recall:.4f}")

    print("\n按类别召回率:")
    class_recall_results = {}
    for class_id in sorted(set(class_wise_tp.keys()) | set(class_wise_fn.keys())):
        tp_count = class_wise_tp[class_id]
        fn_count = class_wise_fn[class_id]
        class_total = tp_count + fn_count
        class_recall = tp_count / class_total if class_total > 0 else 0

        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        class_recall_results[class_name] = class_recall
        print(f"  - {class_name} (ID: {class_id}): TP={tp_count}, FN={fn_count}, Recall={class_recall:.4f}")

    print("\n--- 计算完毕 ---")
    return overall_recall, class_recall_results


if __name__ == "__main__":
    # --- 配置您的路径和参数 ---
    # **请修改为您的实际数据集根目录**
    YOUR_DATASET_ROOT = "data/tzb/Data/comp"

    # 预测标签的根目录
    predictions_root = os.path.join(YOUR_DATASET_ROOT, "predictions")
    # 真实标签的根目录
    labels_root = os.path.join(YOUR_DATASET_ROOT, "labels")

    # 类别名称文件路径 (例如：coco.names, classes.txt 或 classes.json)
    # **请修改为您的类别文件路径或设置为 None**
    classes_file_path = "data/tzb/class.json"

    # 设置 IoU 阈值
    # **IoU 阈值设置为 0.3，如你所要求**
    IOU_THRESHOLD = 0.3

    # 运行召回率计算
    overall_recall, class_recall_results = calculate_recall(
        predictions_root,
        labels_root,
        iou_threshold=IOU_THRESHOLD,
        classes_path=classes_file_path
    )