import os
import json
from PIL import Image

YOLO_ROOT = "data/tzb"  # 根目录，下面有 images/ 和 labels/
OUTPUT_JSON = "data/annotations/tzb_annotations.json"

CATEGORY_MAP = {
    0: "drone",
    1: "car",
    2: "ship",
    3: "bus",
    4: "pedestrian",
    5: "cyclist"
}

def list_videos(base_path):
    return sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

def convert_yolo_to_coco(images_root, labels_root):
    image_id = 1
    annotation_id = 1
    coco_images = []
    coco_annotations = []

    for video_dir in list_videos(images_root):
        image_dir = os.path.join(images_root, video_dir)
        label_dir = os.path.join(labels_root, video_dir)
        if not os.path.exists(label_dir):
            print(f"标签目录不存在，跳过: {label_dir}")
            continue

        img_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])

        for img_file in img_files:
            img_path = os.path.join(image_dir, img_file)
            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(label_dir, label_file)

            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except:
                print(f"无法读取图像: {img_path}")
                continue

            coco_images.append({
                "id": image_id,
                "file_name": f"{video_dir}/{img_file}",
                "width": width,
                "height": height
            })

            if os.path.exists(label_path):
                with open(label_path, "r") as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cls_id, x_center, y_center, w, h = map(float, parts)
                        x_center *= width
                        y_center *= height
                        w *= width
                        h *= height
                        x = x_center - w / 2
                        y = y_center - h / 2

                        coco_annotations.append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(cls_id),
                            "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                            "area": round(w * h, 2),
                            "iscrowd": 0
                        })
                        annotation_id += 1

            image_id += 1

    return {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": cid, "name": name} for cid, name in CATEGORY_MAP.items()]
    }

if __name__ == "__main__":
    images_dir = os.path.join(YOLO_ROOT, "images")
    labels_dir = os.path.join(YOLO_ROOT, "labels")
    output_dir = os.path.dirname(OUTPUT_JSON)
    os.makedirs(output_dir, exist_ok=True)

    print("转换中，请稍等...")
    coco_dict = convert_yolo_to_coco(images_dir, labels_dir)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(coco_dict, f, indent=2)

    print(f"转换完成，输出文件位于：{OUTPUT_JSON}")
