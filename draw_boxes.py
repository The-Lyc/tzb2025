import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import json

def draw_yolo_boxes(image_path, true_labels_path, predicted_labels_path, output_dir, classes_path=None):
    """
    在图像上绘制YOLO格式的真实标签和预测标签。

    Args:
        image_path (str): 原始图像文件的路径。
        true_labels_path (str): 对应真实标签文件的路径（YOLO格式）。
        predicted_labels_path (str): 对应预测标签文件的路径（YOLO格式）。
        output_dir (str): 保存可视化结果图像的目录。
        classes_path (str, optional): 包含类别名称的文件的路径，每行一个类别。
                                      如果提供，则会在边界框旁边显示类别名称。
    """

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 加载图像
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"错误：找不到图像文件 {image_path}")
        return
    except Exception as e:
        print(f"加载图像 {image_path} 时发生错误：{e}")
        return

    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size

    # 颜色定义
    true_color = (0, 255, 0)  # 绿色 (R, G, B)
    pred_color = (0, 0, 255)  # 蓝色

    # 尝试加载字体，如果加载失败则使用默认字体
    try:
        # 尝试加载更常见的字体或指定完整路径
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        try:
            # 对于Linux系统，可能是'DejaVuSansMono.ttf'或其他
            font = ImageFont.truetype("DejaVuSansMono.ttf", 15)
        except IOError:
            print("警告：找不到arial.ttf或DejaVuSansMono.ttf字体，使用默认字体。")
            font = ImageFont.load_default()

    # 加载类别名称
    if classes_path and os.path.exists(classes_path):
        try:
            if classes_path.lower().endswith('.json'):
                with open(classes_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list): # 如果是简单的列表
                        class_names = data
                    elif isinstance(data, dict): # 如果是字典 (键为索引)
                        # 确保按数字顺序排序键并取值
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

    def draw_box(label_path, color, label_type):
        """辅助函数：读取YOLO标签并绘制边界框"""
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue # 跳过格式不正确的行

                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])

                    # 将YOLO归一化坐标转换为像素坐标
                    x1 = int((x_center - width / 2) * img_width)
                    y1 = int((y_center - height / 2) * img_height)
                    x2 = int((x_center + width / 2) * img_width)
                    y2 = int((y_center + height / 2) * img_height)

                    # 绘制矩形框
                    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=1)

                    # 绘制类别名称（如果可用）
                    if class_names and class_id < len(class_names):
                        label_text = f"{label_type}: {class_names[class_id]}"
                    else:
                        label_text = f"{label_type}: Class {class_id}"

                    # 确保文本在图像范围内，避免出界
                    text_x = x1
                    # 尝试在框上方显示文本，如果空间不足则在下方
                    text_y = y1 - 18 if y1 - 18 > 0 else y1 + 2
                    
                    # 获取文本的实际宽度和高度
                    bbox = draw.textbbox((0,0), label_text, font=font) # 使用 (0,0) 获取文本宽度和高度
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    if text_x + text_width > img_width:
                        text_x = img_width - text_width - 5 # 靠右对齐
                    if text_y + text_height > img_height:
                        text_y = img_height - text_height - 5 # 靠下对齐

                    draw.text((text_x, text_y), label_text, fill=color, font=font)
        except FileNotFoundError:
            print(f"警告：找不到 {label_type} 标签文件 {label_path}")
        except Exception as e:
            print(f"处理 {label_type} 标签文件 {label_path} 时发生错误：{e}")


    # 绘制真实标签
    draw_box(true_labels_path, true_color, "True")

    # 绘制预测标签
    draw_box(predicted_labels_path, pred_color, "Pred")

    # 保存结果
    output_filename = os.path.basename(image_path).replace('.', '_visualized.')
    output_path = os.path.join(output_dir, output_filename)
    img.save(output_path)
    # print(f"可视化结果已保存到：{output_path}") # 避免过多输出，只在最后汇总

def process_single_video_frames(video_frames_dir, true_labels_dir, predicted_labels_dir, output_visualizations_dir, classes_path=None):
    """
    处理单个视频目录下的所有视频帧及其对应的标签文件。

    Args:
        video_frames_dir (str): 存放原始视频帧图像的目录。
        true_labels_dir (str): 存放真实标签文件的目录。
        predicted_labels_dir (str): 存放预测标签文件的目录。
        output_visualizations_dir (str): 保存所有可视化结果的目录。
        classes_path (str, optional): 包含类别名称的文件的路径。
    """
    print(f"正在处理视频帧目录：{video_frames_dir}")
    frame_files = [f for f in os.listdir(video_frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    frame_files.sort() # 确保按顺序处理

    for frame_file in frame_files:
        base_name = os.path.splitext(frame_file)[0] # 获取不带扩展名的文件名
        image_path = os.path.join(video_frames_dir, frame_file)
        true_labels_path = os.path.join(true_labels_dir, f"{base_name}.txt")
        predicted_labels_path = os.path.join(predicted_labels_dir, f"{base_name}.txt")

        draw_yolo_boxes(image_path, true_labels_path, predicted_labels_path, output_visualizations_dir, classes_path)
    print(f"视频目录 {os.path.basename(video_frames_dir)} 处理完毕。")

def process_all_videos_in_structure(images_root_dir, labels_root_dir, predictions_root_dir, output_base_dir, classes_path=None):
    """
    遍历给定结构下的所有视频文件夹，并处理其中的帧和标签。

    Args:
        images_root_dir (str): 'images' 目录的根路径。
        labels_root_dir (str): 'labels' 目录的根路径。
        predictions_root_dir (str): 'predictions' 目录的根路径。
        output_base_dir (str): 保存所有可视化结果的根目录。
        classes_path (str, optional): 包含类别名称的文件的路径。
    """
    print(f"开始处理数据集，图像根目录：{images_root_dir}")

    # 获取images根目录下所有的视频子文件夹（例如：video1, video2等）
    video_folders = [f for f in os.listdir(images_root_dir) if os.path.isdir(os.path.join(images_root_dir, f)) and f.startswith('video')]
    video_folders.sort() # 确保按顺序处理 video1, video2...

    if not video_folders:
        print(f"在 {images_root_dir} 中没有找到任何'video'开头的子文件夹。请检查路径和目录结构。")
        return

    for video_folder in video_folders:
        current_video_frames_dir = os.path.join(images_root_dir, video_folder)
        current_true_labels_dir = os.path.join(labels_root_dir, video_folder)
        current_predicted_labels_dir = os.path.join(predictions_root_dir, video_folder)
        
        # 为当前视频创建独立的输出目录
        current_output_vis_dir = os.path.join(output_base_dir, video_folder)
        os.makedirs(current_output_vis_dir, exist_ok=True) # 确保输出目录存在

        # 检查当前视频的标签目录是否存在
        if not os.path.exists(current_true_labels_dir):
            print(f"警告：真实标签目录不存在：{current_true_labels_dir}，跳过此视频的真实标签可视化。")
            # 可以选择跳过整个视频，或者只跳过真实标签绘制
            # continue 
        if not os.path.exists(current_predicted_labels_dir):
            print(f"警告：预测标签目录不存在：{current_predicted_labels_dir}，跳过此视频的预测标签可视化。")
            # 可以选择跳过整个视频，或者只跳过预测标签绘制
            # continue 

        process_single_video_frames(
            current_video_frames_dir,
            current_true_labels_dir,
            current_predicted_labels_dir,
            current_output_vis_dir,
            classes_path
        )
    print("所有视频处理完毕。可视化结果已保存到指定的输出目录。")

if __name__ == "__main__":
    # --- 配置您的路径 ---
    # 请根据您的实际目录结构修改以下变量
    # 假设您的数据集根目录是 'your_dataset_root'
    # 例如：/home/user/my_project/your_dataset_root
    
    # 请确保这些路径是绝对路径或相对于您运行脚本的路径
    YOUR_DATASET_ROOT = "data/tzb/Data/comp" # <-- **请修改为您的实际数据集根目录**
    
    # 构造images, labels, predictions的根目录
    images_root = os.path.join(YOUR_DATASET_ROOT, "images")
    labels_root = os.path.join(YOUR_DATASET_ROOT, "labels")
    predictions_root = os.path.join(YOUR_DATASET_ROOT, "predictions")
    
    # 存放所有可视化结果的根目录，会在此目录下创建 video1, video2... 子文件夹
    output_visualizations_base_dir = os.path.join(YOUR_DATASET_ROOT, "visualized_output")
    
    # 您的类别名称文件路径 (例如：coco.names, classes.txt 等)
    # 如果没有，可以设置为 None，脚本将显示 "Class N"
    classes_file_path = "data/tzb/class.json" # <-- **请修改为您的类别文件路径或设置为 None**

    # 检查主目录是否存在
    if not os.path.exists(images_root):
        print(f"错误：图像根目录不存在：{images_root}")
    elif not os.path.exists(labels_root):
        print(f"错误：真实标签根目录不存在：{labels_root}")
    elif not os.path.exists(predictions_root):
        print(f"错误：预测标签根目录不存在：{predictions_root}")
    else:
        process_all_videos_in_structure(
            images_root,
            labels_root,
            predictions_root,
            output_visualizations_base_dir,
            classes_file_path
        )