import json
import os
import random
from collections import defaultdict, Counter

VAL_RATIO = 0.2
INPUT_JSON = "data/vid/annotations/tzb_train.json"
TRAIN_JSON = "data/vid/annotations/converted_tzb_train.json"
VAL_JSON = "data/vid/annotations/converted_tzb_val.json"
SEED = 42

random.seed(SEED)

with open(INPUT_JSON, "r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

# 统计每张图像出现的类别
image_id_to_categories = defaultdict(set)
for ann in annotations:
    image_id_to_categories[ann["image_id"]].add(ann["category_id"])

# 统计每张图像属于哪个视频
image_id_to_video = {}
video_to_image_ids = defaultdict(list)
for img in images:
    image_id = img["id"]
    video_name = img["file_name"].split("/")[0]
    image_id_to_video[image_id] = video_name
    video_to_image_ids[video_name].append(image_id)

# 统计每个视频中出现的类别
video_to_categories = defaultdict(set)
for vid, img_ids in video_to_image_ids.items():
    cats = set()
    for img_id in img_ids:
        cats |= image_id_to_categories[img_id]
    video_to_categories[vid] = cats

# 统计全局类别分布
total_category_count = Counter()
for ann in annotations:
    total_category_count[ann["category_id"]] += 1

# 按视频划分验证集，使得验证集中各类目标尽可能接近 VAL_RATIO
videos = list(video_to_image_ids.keys())
random.shuffle(videos)

val_videos = set()
val_category_count = Counter()
target_count_per_class = {
    k: int(v * VAL_RATIO) for k, v in total_category_count.items()
}

def is_helpful(video):
    cats = video_to_categories[video]
    return any(val_category_count[c] < target_count_per_class[c] for c in cats)

for vid in videos:
    if not is_helpful(vid):
        continue
    val_videos.add(vid)
    for img_id in video_to_image_ids[vid]:
        for cat in image_id_to_categories[img_id]:
            val_category_count[cat] += 1
    # 停止条件：验证集图像数量达标
    val_img_count = sum(len(video_to_image_ids[v]) for v in val_videos)
    if val_img_count >= len(images) * VAL_RATIO:
        break

train_videos = set(videos) - val_videos

# 构建最终的图像/标注列表
train_image_ids = [img_id for vid in train_videos for img_id in video_to_image_ids[vid]]
val_image_ids = [img_id for vid in val_videos for img_id in video_to_image_ids[vid]]

id_to_image = {img["id"]: img for img in images}
train_images = [id_to_image[iid] for iid in train_image_ids]
val_images = [id_to_image[iid] for iid in val_image_ids]
train_anns = [ann for ann in annotations if ann["image_id"] in train_image_ids]
val_anns = [ann for ann in annotations if ann["image_id"] in val_image_ids]

# 保存 JSON
def save_json(path, images, annotations):
    with open(path, "w") as f:
        json.dump({
            "images": images,
            "annotations": annotations,
            "categories": categories
        }, f, indent=2)

os.makedirs(os.path.dirname(TRAIN_JSON), exist_ok=True)
save_json(TRAIN_JSON, train_images, train_anns)
save_json(VAL_JSON, val_images, val_anns)

# 打印信息
print(f"✅ 训练视频数: {len(train_videos)}, 图像数: {len(train_images)}")
print(f"✅ 验证视频数: {len(val_videos)}, 图像数: {len(val_images)}\n")

print("📊 验证集类别分布:")
for cat in categories:
    cid = cat["id"]
    val_count = sum(1 for ann in val_anns if ann["category_id"] == cid)
    total = total_category_count[cid]
    print(f"- {cat['name']:<12}: {val_count:4d} / {total:4d} ({val_count / total:.1%})")
