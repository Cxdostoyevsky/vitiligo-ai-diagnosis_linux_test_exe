import json
import os
import torch
from PIL import Image
from config import IMAGE_ROOT_DIR, INPUT_CONFIG

def extract_id_from_path(image_path):
    """从图像路径中提取ID"""
    parts = image_path.split('/')
    file_id = parts[-1].split('.')[0]
    #这里得到的id是001_N, 001_W, 001_N_overlay, 001_W_overlay，这里只要前面的编号
    file_id = file_id.split('_')[0]
    return f"{file_id}"

def load_single_image(image_path, processor):
    """加载并处理单张图像"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像不存在: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    return inputs['pixel_values'].squeeze(0)

def load_test_data(processor=None, input_config=None, image_root_dir=None):
    """加载测试数据 - 使用外部传入的处理器"""
    
    # 使用传入的配置，如果没有则使用默认配置
    if input_config is None:
        input_config = INPUT_CONFIG
    # if image_root_dir is None:
    #     image_root_dir = IMAGE_ROOT_DIR
    
    # 检查input_config是否为空
    if not input_config:
        raise ValueError("input_config不能为空，至少需要一个输入类型配置")
    
    # 第一步：确定主顺序（使用第一个输入类型作为参考）
    primary_type = next(iter(input_config))
    primary_json = input_config[primary_type]["json_path"]
    # 读取主JSON并记录ID顺序
    with open(primary_json, 'r', encoding='utf-8') as f:
        primary_data = json.load(f)   
    # 提取主顺序的样本ID列表
    ordered_ids = []
    for item in primary_data:
        sample_id = extract_id_from_path(item["image"][0])
        ordered_ids.append(sample_id)   
    
    # 第二步：创建有序样本字典
    test_samples = {}
    for sample_id in ordered_ids:
        test_samples[sample_id] = {
            "id": sample_id,
            "images": {}
        }
    
    # 第三步：按输入类型加载数据，保持主顺序
    for input_type, config in input_config.items():
        with open(config["json_path"], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 创建当前JSON的ID到数据的映射
        id_to_data = {}
        for item in data:
            item_id = extract_id_from_path(item["image"][0])
            id_to_data[item_id] = item
        
        # 按照主顺序处理样本
        for sample_id in ordered_ids:
            if sample_id in id_to_data:
                item = id_to_data[sample_id]
            else:
                print(f"警告: 样本 {sample_id} 在 {input_type} JSON中缺失")
                item = {"image": []}  # 创建空条目
            
            images = []
            for img_path in item["image"]:
                full_path = img_path
                try:
                    img_tensor = load_single_image(full_path, processor)
                    images.append(img_tensor)
                except FileNotFoundError:
                    placeholder = torch.zeros((3, 512, 512))
                    images.append(placeholder)
                    print(f"警告: {input_type} 图像未找到: {img_path}")
            
            test_samples[sample_id]["images"][input_type] = images
    
            
    return ordered_ids, test_samples


def load_single_test_sample(case_id, img_path_dict, processor):
    """
    根据给定的case_id和图像路径字典，为单个样本加载所有需要的图像组合。
    先加载所有需要的图像，然后根据组合进行组装，避免重复加载。
    直接返回一个可供InferenceSystem使用的test_samples字典。
    """
    test_sample = {
        "id": case_id,
        "images": {}
    }

    # 1. 收集所有不重复的有效图像路径并一次性加载
    loaded_images = {}
    unique_paths = set(p for p in img_path_dict.values() if p)
    for img_path in unique_paths:
        try:
            img_tensor = load_single_image(img_path, processor)
            loaded_images[img_path] = img_tensor
        except FileNotFoundError:
            # 如果文件找不到，用一个全黑的占位符代替
            placeholder = torch.zeros((3, 512, 512))
            loaded_images[img_path] = placeholder
            print(f"警告: 图像文件未找到: {img_path}，已使用占位符代替。")

    # 定义所有可能的图像组合
    image_combinations = {
        "oc": [img_path_dict.get("oc")],
        "ow": [img_path_dict.get("ow")],
        "oc_ec": [img_path_dict.get("oc"), img_path_dict.get("ec")],
        "ow_ew": [img_path_dict.get("ow"), img_path_dict.get("ew")],
        "oc_ow": [img_path_dict.get("oc"), img_path_dict.get("ow")],
        "oc_ec_ow_ew": [
            img_path_dict.get("oc"), img_path_dict.get("ec"),
            img_path_dict.get("ow"), img_path_dict.get("ew")
        ]
    }
    
    # 3. 从已加载的图像中组装组合
    for combo_name, img_paths in image_combinations.items():
        images = []
        # 过滤掉None的路径
        valid_paths = [p for p in img_paths if p]
        
        if not valid_paths:
            # print(f"警告: 样本 {case_id} 的组合 {combo_name} 缺少所有图像路径。")
            continue

        for img_path in valid_paths:
            images.append(loaded_images[img_path])
        
        if images:
            test_sample["images"][combo_name] = images
            
    # 返回一个包含单个样本的字典，键是case_id
    return {case_id: test_sample}