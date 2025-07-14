import os
import re

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def rename_subdirs(root_dir):
    for i in range(1, 27):
        old_name = f"data{str(i).zfill(2)}"
        new_name = f"video{i}"
        old_path = os.path.join(root_dir, old_name)
        new_path = os.path.join(root_dir, new_name)
        if os.path.exists(old_path):
            print(f"重命名目录: {old_name} -> {new_name}")
            os.rename(old_path, new_path)
        else:
            print(f"目录不存在: {old_name}, 跳过")

def rename_labels_in_dir(dir_path):
    files = sorted(os.listdir(dir_path), key=natural_key)
    txt_files = [f for f in files if f.lower().endswith('.txt')]

    digits = len(str(len(txt_files) - 1))  # 编号位数，从0开始

    for idx, old_name in enumerate(txt_files):
        new_name = f"frame{str(idx).zfill(digits)}.txt"
        old_path = os.path.join(dir_path, old_name)
        new_path = os.path.join(dir_path, new_name)

        if old_path != new_path:
            print(f"重命名标签: {old_name} -> {new_name}")
            os.rename(old_path, new_path)

def main(root_dir):
    rename_subdirs(root_dir)
    for i in range(1, 27):
        labels_dir = os.path.join(root_dir, f"video{i}")
        if not os.path.exists(labels_dir):
            print(f"目录不存在: {labels_dir}, 跳过")
            continue
        print(f"处理标签目录: {labels_dir}")
        rename_labels_in_dir(labels_dir)

if __name__ == '__main__':
    root_dir = 'data/tzb/labels'  # 替换成你的 labels 根目录路径
    main(root_dir)
