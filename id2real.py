import os
import json
import shutil
from pathlib import Path

def reorganize_predictions(json_file_path, input_pred_dir, new_output_root_dir):
    """
    根据JSON文件中的图片路径结构，重新组织预测结果文件。

    Args:
        json_file_path (str): 包含图片元数据（id, file_name）的JSON文件路径。
        input_pred_dir (str): 存放原始预测文件（output_ID.txt）的目录。
        new_output_root_dir (str): 重新组织后的预测文件存放的根目录。
                                   例如，如果 file_name 是 'comp/images/video1/frame0000.jpg'，
                                   且 new_output_root_dir 是 'reorganized_preds'，
                                   则文件会存放到 'reorganized_preds/comp/images/video1/frame0000.txt'。
    """
    json_file_path = Path(json_file_path)
    input_pred_dir = Path(input_pred_dir)
    new_output_root_dir = Path(new_output_root_dir)

    if not json_file_path.exists():
        print(f"错误: JSON文件不存在于 {json_file_path}")
        return

    if not input_pred_dir.is_dir():
        print(f"错误: 输入预测目录不存在或不是目录 {input_pred_dir}")
        return

    # 确保新的输出根目录存在
    new_output_root_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取JSON文件并建立 image_id 到 file_name 的映射
    print(f"正在读取JSON文件: {json_file_path}...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 假设 'images' 键存在
    if "images" not in data:
        print("错误: JSON文件中未找到 'images' 键。")
        return

    # 构建 ID 到文件路径的映射
    id_to_file_name = {img_info["id"]: img_info["file_name"] for img_info in data["images"]}
    print(f"从JSON文件中加载了 {len(id_to_file_name)} 条图片信息。")

    # 2. 遍历预测文件目录，进行重命名和移动
    print(f"正在处理预测文件目录: {input_pred_dir}...")
    processed_count = 0
    for pred_file in input_pred_dir.iterdir():
        if pred_file.is_file() and pred_file.name.startswith("output_") and pred_file.suffix == ".txt":
            try:
                # 提取 image_id
                image_id_str = pred_file.stem.replace("output_", "")
                image_id = int(image_id_str)

                if image_id in id_to_file_name:
                    original_file_name = id_to_file_name[image_id]
                    
                    # 构建新的目标路径
                    # 例如 'comp/images/video1/frame0000.jpg' -> 'comp/images/video1/frame0000.txt'
                    # 替换掉图片后缀，变成txt后缀
                    p = Path(original_file_name)
                    if len(p.parts) >= 2:
                        new_relative_path = Path(p.parts[-2]) / Path(p.parts[-1]).with_suffix(".txt")
                    else:
                        # 如果只有一层，直接改后缀
                        new_relative_path = p.with_suffix(".txt")
                    new_full_path = new_output_root_dir / new_relative_path

                    # 确保目标目录存在
                    new_full_path.parent.mkdir(parents=True, exist_ok=True)

                    # 移动/复制文件
                    shutil.copy2(pred_file, new_full_path) # 使用 copy2 保留元数据
                    processed_count += 1
                else:
                    print(f"警告: 预测文件 {pred_file.name} 对应的 image_id {image_id} 未在JSON中找到。跳过。")
            except ValueError:
                print(f"警告: 文件名 {pred_file.name} 无法解析出有效的 image_id。跳过。")
            except Exception as e:
                print(f"处理文件 {pred_file.name} 时发生错误: {e}")

    print(f"处理完成。成功组织了 {processed_count} 个预测文件到 {new_output_root_dir}")
    print(f"原始预测文件仍保留在 {input_pred_dir} 中。")

# --- 使用示例 ---
if __name__ == "__main__":
    # 请根据您的实际路径修改以下变量
    JSON_FILE = "data/tzb/annotations/tzb_test_exclude.json" # 或者 tzb_test_exclude.json
    INPUT_PRED_DIR = "output/tzb_multi"
    NEW_OUTPUT_ROOT_DIR = "output/reorganized_predictions" # 新的预测文件存放的根目录

    reorganize_predictions(JSON_FILE, INPUT_PRED_DIR, NEW_OUTPUT_ROOT_DIR)

    # 示例2：如果您想把新组织的预测文件放到某个特定位置，例如像图像那样放
    # 假设你的原始图像根目录是 /data/tzb_dataset/
    # 那么 reorganize_predictions 的 new_output_root_dir 可以是 /data/tzb_dataset/predictions/
    # 这样，最终预测文件会是 /data/tzb_dataset/predictions/comp/images/video1/frame0000.txt
    # JSON_FILE = "/path/to/your/dataset_root/annotations/tzb_test_exclude.json"
    # INPUT_PRED_DIR = "output/tzb_multi"
    # NEW_OUTPUT_ROOT_DIR = "/path/to/your/dataset_root/predictions" 
    # reorganize_predictions(JSON_FILE, INPUT_PRED_DIR, NEW_OUTPUT_ROOT_DIR)