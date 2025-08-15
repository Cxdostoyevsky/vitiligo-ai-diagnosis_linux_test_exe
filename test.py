import os
import json
import torch
import pandas as pd
import time
from tqdm import tqdm
from data_loader import load_single_test_sample
from transformers import AutoProcessor, AutoModel

# --- 终极解决方案: 显式导入，让PyInstaller能够识别 ---
try:
    from transformers import SiglipProcessor, SiglipImageProcessor, SiglipVisionModel
except ImportError:
    pass # 在正常环境中，这些导入可能不是必需的，但在打包时至关重要

from data_generate.generate_datasets_stage_2_binary_cls import two_binary_cls
from data_generate.generate_datasets_stage_2_choice import two_choice
from models import SingleStreamModel, DualStreamModel, QuadStreamModel
from config import (
    DEVICE, MODEL_PATHS, SIGLIP_MODEL_PATH, SIGLIP_PROCESSOR_PATH,
    PROBABILITIES_CSV, PREDICTIONS_CSV, INPUT_CONFIG, VOTE_CSV, LVLM_MODEL_PATH,
    LVLM_PROBABILITIES_CSV, LVLM_PREDICTIONS_CSV
)
import cv2
import argparse
from config import *
from Feature_extran_vis_tools.feature_extractor import extract_woodlamp_edge_features
from Feature_extran_vis_tools.create_highlight_overlays import create_overlay_image
from Feature_extran_vis_tools.batch_processor_V2 import PARAMETER_CONFIGS
import multiprocessing
from lvlm.utils.arguments import set_seed
from lvlm.dataset.dataset import MultiModalDataset, DataCollatorForMultiModalDataset
from lvlm.model.modeling_lvlm import LVLMForConditionalGeneration
from test_generate import create_prompts
from torch.utils.data import DataLoader
import shutil
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.dirname(os.path.join(BASE_DIR, "results/predictions_lvlm.csv")), exist_ok=True)
os.makedirs(os.path.dirname(LVLM_PREDICTIONS_CSV), exist_ok=True)

# --- 从 create_highlight_overlaysss.py 中获取的关键参数 ---
HIGHLIGHT_THRESHOLD = 5
ORIGINAL_IMAGE_ALPHA = 0.4
HEATMAP_BETA = 0.6
COLOR_MAPPING = {
    "clinical": cv2.COLORMAP_HOT,
    "wood_lamp": cv2.COLORMAP_COOL
}

def extract_features_worker(img_path, temp_feature_map_dir, primary_params, fallback_params):
    """
    第一阶段的Worker函数：为单张图片提取并保存特征图。
    """
    try:
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            return (img_path, False, f"路径无效或不存在")

        base_name = os.path.basename(img_path)
        name, _ = os.path.splitext(base_name)
        
        original_img = cv2.imread(img_path)
        if original_img is None:
            return (img_path, False, "无法读取图片")

        # 优先使用主配置提取特征
        success = extract_woodlamp_edge_features(original_img, temp_feature_map_dir, name, **primary_params)
        
        # 如果主配置失败，尝试回退配置
        if not success:
            success = extract_woodlamp_edge_features(original_img, temp_feature_map_dir, name, **fallback_params)
        
        if success:
            return (img_path, True, "成功")
        else:
            return (img_path, False, "特征提取失败")
            
    except Exception as e:
        return (img_path, False, f"处理时发生异常: {e}")

def create_overlay_worker(img_path, temp_feature_map_dir, output_dir):
    """
    第二阶段的Worker函数：为单张图片创建并保存叠加图。
    """
    try:
        base_name = os.path.basename(img_path)
        name, _ = os.path.splitext(base_name)
        temp_feature_map_path = os.path.join(temp_feature_map_dir, f"{name}_temp_feature.png")

        if not os.path.exists(temp_feature_map_path):
            return (img_path, False, "特征图文件不存在")

        original_img = cv2.imread(img_path)
        gradient_map_gray = cv2.imread(temp_feature_map_path, cv2.IMREAD_GRAYSCALE)

        if original_img is None or gradient_map_gray is None:
            return (img_path, False, "无法读取原图或特征图")

        # 1. 创建二值蒙版
        _, mask = cv2.threshold(gradient_map_gray, HIGHLIGHT_THRESHOLD, 255, cv2.THRESH_BINARY)
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 2. 确定颜色并生成热力图
        image_type = 'wood_lamp' if '_W' in base_name else 'clinical'
        colormap = COLOR_MAPPING.get(image_type, cv2.COLORMAP_JET)
        heatmap_color = cv2.applyColorMap(gradient_map_gray, colormap)

        # 确保尺寸一致
        if original_img.shape[:2] != heatmap_color.shape[:2]:
            heatmap_color = cv2.resize(heatmap_color, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR)

        # 3. 半透明融合
        blended_img = cv2.addWeighted(original_img, ORIGINAL_IMAGE_ALPHA, heatmap_color, HEATMAP_BETA, 0)
        
        # 4. 使用蒙版合成最终图像
        overlay_img = np.where(mask_3channel > 0, blended_img, original_img)

        # 5. 保存最终叠加图到目标输出目录
        output_filename = f"{name}_overlay.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, overlay_img)
        
        return (img_path, True, "成功")

    except Exception as e:
        return (img_path, False, f"创建叠加图时发生异常: {e}")


def preprocess_and_generate_feature_maps(csv_path, output_dir, num_workers=6):
    """
    读取 data.csv，并行地为每张图片提取边缘特征图，然后并行地创建叠加图。
    Args:
        csv_path (str): 输入的CSV文件路径.
        output_dir (str): 最终叠加图的输出目录.
        num_workers (int): 用于并行处理的进程数.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 创建一个临时目录来存放中间生成的特征图
    temp_feature_map_dir = os.path.join(os.path.dirname(output_dir), "temp_edge_features")
    os.makedirs(temp_feature_map_dir, exist_ok=True)

    df = pd.read_csv(csv_path, sep='[,\t]', engine='python')
    configs_by_name = {c['name']: c['params'] for c in PARAMETER_CONFIGS}
    
    # 定义主配置和回退配置
    primary_config_name = 'config_09_sensitive_and_strong_smoothing'
    fallback_config_name = 'config_01_baseline_c10_k5'
    primary_params = configs_by_name.get(primary_config_name)
    fallback_params = configs_by_name.get(fallback_config_name)

    if not primary_params or not fallback_params:
        raise ValueError("指定的配置名称在 PARAMETER_CONFIGS 中找不到，请检查 batch_processor_V2.py")

    image_paths = pd.concat([df['Wood_path'], df['Normal_path']]).dropna().unique()

    # --- 阶段 1: 并行提取特征图 ---
    print(f"--- 阶段 1: 使用 {num_workers} 个进程并行提取特征图 ---")
    successful_images = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 使用 functools.partial 预先绑定不变的参数
        worker_func = functools.partial(extract_features_worker, 
                                        temp_feature_map_dir=temp_feature_map_dir,
                                        primary_params=primary_params,
                                        fallback_params=fallback_params)
        
        futures = {executor.submit(worker_func, img_path): img_path for img_path in image_paths}
        
        progress_bar = tqdm(total=len(futures), desc="提取特征图")
        for future in as_completed(futures):
            img_path, success, message = future.result()
            if success:
                successful_images.append(img_path)
            else:
                print(f"警告: 处理 '{img_path}' 失败 - {message}")
            progress_bar.update(1)
        progress_bar.close()

    # --- 阶段 2: 并行创建叠加图 ---
    print(f"\n--- 阶段 2: 使用 {num_workers} 个进程并行创建叠加图 ---")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 预绑定参数
        worker_func = functools.partial(create_overlay_worker,
                                        temp_feature_map_dir=temp_feature_map_dir,
                                        output_dir=output_dir)

        futures = {executor.submit(worker_func, img_path): img_path for img_path in successful_images}

        progress_bar = tqdm(total=len(futures), desc="创建叠加图")
        for future in as_completed(futures):
            img_path, success, message = future.result()
            if not success:
                print(f"警告: 创建叠加图失败 '{img_path}' - {message}")
            progress_bar.update(1)
        progress_bar.close()

    # 清理临时文件夹
    try:
        shutil.rmtree(temp_feature_map_dir)
        print(f"临时文件夹 {temp_feature_map_dir} 已成功清理。")
    except OSError as e:
        print(f"清理临时文件夹时出错: {e.strerror}")


class InferenceSystem:
    """推理系统 - 提前加载所有资源"""
    def __init__(self, device):
        self.device = device
        self.processor = None
        self.backbone = None
        self.models = {}
        
    def initialize(self):
        """初始化系统 - 加载处理器、主干和所有模型"""
        # 加载处理器
        print("加载SigLIP处理器...")
        start = time.time()
        self.processor = AutoProcessor.from_pretrained(SIGLIP_PROCESSOR_PATH)
        print(f"处理器加载完成，耗时: {time.time() - start:.2f}s")
        
        # 加载主干模型
        print("加载SigLIP主干模型...")
        start = time.time()
        self.backbone = AutoModel.from_pretrained(SIGLIP_MODEL_PATH).to(self.device)
        self.backbone.eval()
        print(f"主干模型加载完成，耗时: {time.time() - start:.2f}s")
        
        # 加载所有分类头
        print("加载分类头模型...")
        start = time.time()
        self.models = {
            "oc": SingleStreamModel(self.backbone).to(self.device),
            "ow": SingleStreamModel(self.backbone).to(self.device),
            "oc_ec": DualStreamModel(self.backbone).to(self.device),
            "ow_ew": DualStreamModel(self.backbone).to(self.device),
            "oc_ow": DualStreamModel(self.backbone).to(self.device),
            "oc_ec_ow_ew": QuadStreamModel(self.backbone).to(self.device),
        }
        
        # 加载训练好的参数
        for input_type, model in self.models.items():
            model_path = MODEL_PATHS[input_type]
            model.load_trainable_params(model_path)
            # for name, param in model.named_parameters():
            #     if name.startswith('classifier'):
            #         print(f"{name}: {param}")
            model.eval()
        
        print(f"所有分类头加载完成，耗时: {time.time() - start:.2f}s")
 
    
    def load_data(self, case_id, img_path_dict):
        """为单个样本加载测试数据"""
        # print("加载单个测试样本...")
        # start = time.time()
        test_samples = load_single_test_sample(case_id, img_path_dict, self.processor)
        # print(f"加载样本 {case_id} 完成，耗时: {time.time() - start:.2f}s")
        return [case_id], test_samples

    
    def run_inference(self, ordered_ids, test_samples):
        """对单个样本运行推理，返回其概率"""
        # print("开始单样本推理...")
        # start_time = time.time()
        
        # 由于我们一次只处理一个样本，所以直接获取
        sample_id = ordered_ids[0]
        sample_data = test_samples[sample_id]
        
        prob_row = {"id": sample_id}
        
        for input_type, model in self.models.items():
            if input_type not in sample_data["images"]:
                # print(f"警告: 样本 {sample_id} 缺少 {input_type} 图像数据，跳过该模型")
                continue
                
            images = sample_data["images"][input_type]
            # 确保即使只有一个图像，它也被包装在列表中
            if not isinstance(images, list):
                images = [images]
            input_tensors = [img.unsqueeze(0).to(self.device) for img in images]
            
            with torch.no_grad():
                outputs = model(*input_tensors)
            
            probs = torch.nn.functional.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
            
            prob_row[f"{input_type}_stable"] = probs[0]
            prob_row[f"{input_type}_active"] = probs[1]
        
        # total_time = time.time() - start_time
        # print(f"单样本推理完成，耗时: {total_time:.2f}s")
        
        # 返回包含单个样本概率的列表
        return [prob_row]

def save_prob_progress(case_id, siglip_probs_row, lvlm_probs, lvlm_preds, output_csv_path):
    """
    将单个样本的SigLIP和LVLM的详细概率追加保存到CSV文件。
    """
    PROMPT_TYPES = ["oc", "ow", "oc_ec", "ow_ew", "oc_ow", "oc_ec_ow_ew"]
    
    header = ["name"]
    for p_type in PROMPT_TYPES:
        header.extend([f"siglip_{p_type}_stable", f"siglip_{p_type}_active"])
    for p_type in PROMPT_TYPES:
        header.extend([f"lvlm_{p_type}_binary_cls_active", f"lvlm_{p_type}_binary_cls_stable", 
                       f"lvlm_{p_type}_choice_active", f"lvlm_{p_type}_choice_stable"])

    row_data = {"name": f"{int(case_id):03d}"}

    # SigLIP
    for p_type in PROMPT_TYPES:
        row_data[f"siglip_{p_type}_stable"] = siglip_probs_row.get(f"{p_type}_stable")
        row_data[f"siglip_{p_type}_active"] = siglip_probs_row.get(f"{p_type}_active")

    # LVLM
    for p_type in PROMPT_TYPES:
        # Binary
        binary_pred = lvlm_preds.get(p_type, [-1,-1])[0]
        active_prob_b, stable_prob_b = None, None
        if binary_pred != -1:
            prob_b = lvlm_probs.get(f"{p_type}_binary_cls", 0)
            active_prob_b = prob_b if binary_pred == 0 else (1 - prob_b)
            stable_prob_b = 1 - active_prob_b
        row_data[f"lvlm_{p_type}_binary_cls_active"] = active_prob_b
        row_data[f"lvlm_{p_type}_binary_cls_stable"] = stable_prob_b
        
        # Choice
        choice_pred = lvlm_preds.get(p_type, [-1,-1])[1]
        active_prob_c, stable_prob_c = None, None
        if choice_pred != -1:
            prob_c = lvlm_probs.get(f"{p_type}_choice", 0)
            active_prob_c = prob_c if choice_pred == 0 else (1 - prob_c)
            stable_prob_c = 1 - active_prob_c
        row_data[f"lvlm_{p_type}_choice_active"] = active_prob_c
        row_data[f"lvlm_{p_type}_choice_stable"] = stable_prob_c

    file_exists = os.path.isfile(output_csv_path)
    dir_name = os.path.dirname(output_csv_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    df = pd.DataFrame([row_data], columns=header)
    df.to_csv(output_csv_path, mode='a', header=not file_exists, index=False)


def fuse_single_item_results(case_id, siglip_probs, lvlm_preds, lvlm_probs, prob_csv_path):
    """
    融合单个样本的SigLIP和LVLM结果，并保存详细概率。
    """
    # --- 保存概率进展 ---
    if siglip_probs: # siglip_probs is a list with one element
        save_prob_progress(case_id, siglip_probs[0], lvlm_probs, lvlm_preds, prob_csv_path)

    PROMPT_TYPES = ["oc", "ow", "oc_ec", "ow_ew", "oc_ow", "oc_ec_ow_ew"]
    
    # --- 1. 提取LVLM的有效预测和概率 ---
    valid_lvlm_active_probs = []
    valid_lvlm_stable_probs = []
    
    for p_type in PROMPT_TYPES:
        if p_type in lvlm_preds:
            binary_cls_pred, choice_pred = lvlm_preds[p_type]
            # 检查预测的一致性
            if binary_cls_pred != -1 and binary_cls_pred == choice_pred:
                # 提取二分类任务的概率
                prob_b = lvlm_probs.get(f"{p_type}_binary_cls", 0)
                active_prob_b = prob_b if binary_cls_pred == 0 else (1 - prob_b)
                stable_prob_b = 1 - active_prob_b
                valid_lvlm_active_probs.append(active_prob_b)
                valid_lvlm_stable_probs.append(stable_prob_b)
                
                # 提取选择题任务的概率
                prob_c = lvlm_probs.get(f"{p_type}_choice", 0)
                active_prob_c = prob_c if choice_pred == 0 else (1 - prob_c)
                stable_prob_c = 1 - active_prob_c
                valid_lvlm_active_probs.append(active_prob_c)
                valid_lvlm_stable_probs.append(stable_prob_c)

    # --- 2. 提取SigLIP的概率 ---
    siglip_active_probs = []
    siglip_stable_probs = []
    # siglip_probs 是一个包含单行字典的列表
    if siglip_probs:
        probs_row = siglip_probs[0]
        for p_type in PROMPT_TYPES:
            active_col = f"{p_type}_active"
            stable_col = f"{p_type}_stable"
            if active_col in probs_row and stable_col in probs_row:
                siglip_active_probs.append(probs_row[active_col])
                siglip_stable_probs.append(probs_row[stable_col])

    # --- 3. 权重计算和融合 ---
    n_siglip = len(siglip_active_probs)
    n_lvlm = len(valid_lvlm_active_probs)

    if n_siglip + n_lvlm == 0:
        return "稳定期" # 默认预测

    # 交叉权重
    weight_siglip = n_lvlm / (n_siglip + n_lvlm) if (n_siglip + n_lvlm) > 0 else 0
    weight_lvlm = n_siglip / (n_siglip + n_lvlm) if (n_siglip + n_lvlm) > 0 else 0

    # 加权求和
    weighted_sum_active = sum(siglip_active_probs) * weight_siglip + sum(valid_lvlm_active_probs) * weight_lvlm
    weighted_sum_stable = sum(siglip_stable_probs) * weight_siglip + sum(valid_lvlm_stable_probs) * weight_lvlm
    
    # 归一化并预测
    total_weighted_sum = weighted_sum_active + weighted_sum_stable
    if total_weighted_sum == 0:
        final_active_prob = 0.5
    else:
        final_active_prob = weighted_sum_active / total_weighted_sum

    return "进展期" if final_active_prob > 0.5 else "稳定期"


def save_final_results(result_item, result_csv_path):
    """将单个融合后的结果追加保存到最终的CSV文件。"""
    file_exists = os.path.isfile(result_csv_path)
    
    # 确保目录存在
    dir_name = os.path.dirname(result_csv_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    df = pd.DataFrame([result_item])
    df.to_csv(result_csv_path, mode='a', header=not file_exists, index=False)


def main():
    parser = argparse.ArgumentParser(description="模型推理脚本")
    parser.add_argument('--data_path', type=str, default='data.csv', help='包含图像路径的输入CSV文件')
    parser.add_argument('--result_path', type=str, default='result.csv', help='保存最终结果的CSV文件路径')
    parser.add_argument('--prob_path', type=str, default='prob_progress.csv', help='保存概率演进的CSV文件路径')
    parser.add_argument("--model_dtype", type=str, default="bfloat16")
    parser.add_argument("--data_path_lvlm", type=str, default="docs/example_2d_inference.json")
    parser.add_argument("--conv_version", type=str, default="qwen2")
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--image3d_path", type=str, default="")
    parser.add_argument("--resume_from_checkpoint", type=str,
                        default=LVLM_MODEL_PATH)
    parser.add_argument("--output_dir", type=str, default="/hdd/common/datasets/medical-image-analysis/tzb/test_code_sym/results/")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()


    device = torch.device(f"cuda:{DEVICE}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    


    #处理数据
    # print("处理数据,生成特征图...")
    original_csv_path = args.data_path
    feature_map_dir = "output_edge_features"
    preprocess_and_generate_feature_maps(original_csv_path, feature_map_dir, num_workers=6)




    # --- 生成 模型所需的json文件 ---
    # print("模型所需要的json文件...")
    # 使用相对路径，在项目根目录下创建 'json_path' 文件夹
    json_path = 'json_path' 
    master_json_path = os.path.join(json_path, 'master.json')
    
    os.makedirs(json_path, exist_ok=True)

    # 读取原始csv以获取所有图片路径
    df = pd.read_csv(original_csv_path, sep='\t', engine='python')

    master_data_list = []
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        wood_path = getattr(row, "Wood_path")
        normal_path = getattr(row, "Normal_path")
        patient_id = f"{idx:03d}"
        master_case_data = {
            "idx": idx,
            "status": "active_or_stable", # 假定状态，可根据需要修改
            "patient_id": patient_id,
            "images": {
                "clinical": [normal_path] if normal_path else [],
                "wood_lamp": [wood_path] if wood_path else []
            }
        }
        master_data_list.append(master_case_data)

    # 保存 master.json
    with open(master_json_path, 'w', encoding='utf-8') as f:
        json.dump(master_data_list, f, indent=2, ensure_ascii=False)
    # print(f"Master JSON 文件已生成: {master_json_path}")

    # 调用函数生成二分类和选择题的数据集
    # 注意：我们将 'output_edge_features' 作为 doot_dir 传递
    two_binary_cls(master_json_path, json_path, doot_dir=feature_map_dir)
   




    


    # =================================================================
    # 主推理流程：按样本逐一处理
    # =================================================================
    print("\n--- 开始按样本进行流式推理与融合 ---")

    # 1. 初始化所有模型
    # SigLIP 推理系统
    inference_system_siglip = InferenceSystem(device)
    inference_system_siglip.initialize()

    # LVLM 推理系统
    model_dtype = torch.float16 if args.model_dtype == "float16" else (torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float32)
    model_lvlm = LVLMForConditionalGeneration.from_pretrained(args.resume_from_checkpoint)
    model_lvlm.to(dtype=model_dtype, device=device)
    model_lvlm.eval()

    # 2. 加载主数据列表
    # 我们使用一个包含所有图像路径的JSON文件作为循环的基础
    json_data_path = os.path.join(json_path, 'train_binary_cls_4img_OC_EC_OW_EW.json')
    print(f"从以下路径加载主数据列表: {json_data_path}")
    with open(json_data_path, 'r', encoding='utf-8') as f:
        main_data_list = json.load(f)

    # 在开始新一轮推理前，检查并删除旧的结果文件
    if os.path.exists(args.result_path):
        os.remove(args.result_path)
        print(f"已删除旧的结果文件: {args.result_path}")
    if os.path.exists(args.prob_path):
        os.remove(args.prob_path)
        print(f"已删除旧的概率文件: {args.prob_path}")

    # 3. 循环处理每个样本
    with torch.no_grad():
        for item in tqdm(main_data_list, desc="模型推理与融合"):
            case_id = str(item['id'])
            # LVLM的图像路径字典
            lvlm_img_path_dict = {
                "oc": item['image'][0], "ec": item['image'][1],
                "ow": item['image'][2], "ew": item['image'][3],
            }
            
            # 使用与LVLM模型兼容的processor来预加载图像
            processor = model_lvlm.encoder_image.processor
            data_list_prompts, loaded_images = create_prompts(case_id, lvlm_img_path_dict, processor)

            for data_prompt in data_list_prompts:
                train_dataset = MultiModalDataset(
                    model=model_lvlm, 
                    data=[data_prompt], 
                    data_arguments=args, 
                    mode="eval",
                    loaded_images=loaded_images
                )
                data_collator = DataCollatorForMultiModalDataset(tokenizer=model_lvlm.tokenizer, mode="eval")
                data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

                for batch in data_loader:
                    # (此处省略了将batch数据移动到device的代码，实际应保留)
                    for k, v in batch.items():
                        if v is not None:
                            if isinstance(v, torch.Tensor):
                                batch[k] = v.to(device)
                                if k == "image" or k == "image3d":
                                    batch[k] = v.to(dtype=model_dtype, device=device)

                    output_ids, output_scores = model_lvlm.generate(
                        **batch, max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                        pad_token_id=model_lvlm.tokenizer.pad_token_id, eos_token_id=model_lvlm.tokenizer.eos_token_id,
                    )
                    outputs = model_lvlm.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                    
                    input_token_len = batch['input_ids'].shape[1]
                    generated_token_ids = output_ids[0, input_token_len:]
                    prob = torch.softmax(output_scores[0][0], dim=0)[generated_token_ids[0]].item() if output_scores else 0.0

                    sample_id_key = data_prompt["id"].replace(f"{case_id}_", "")
                    if "binary_cls" in sample_id_key:
                        sample_id_key = sample_id_key.replace("binary_cls_", "")
                        if sample_id_key not in lvlm_output_preds: lvlm_output_preds[sample_id_key] = [-1, -1]
                        lvlm_output_probs[f"{sample_id_key}_binary_cls"] = prob
                        if "stable" in outputs[0].lower(): lvlm_output_preds[sample_id_key][0] = 1
                        elif "active" in outputs[0].lower(): lvlm_output_preds[sample_id_key][0] = 0
                    else: # choice
                        sample_id_key = sample_id_key.replace("choice_", "")
                        if sample_id_key not in lvlm_output_preds: lvlm_output_preds[sample_id_key] = [-1, -1]
                        lvlm_output_probs[f"{sample_id_key}_choice"] = prob
                        if "a" in outputs[0].lower(): lvlm_output_preds[sample_id_key][1] = 1
                        elif "b" in outputs[0].lower(): lvlm_output_preds[sample_id_key][1] = 0

            # --- Stage 3: 单样本融合 ---
            final_prediction = fuse_single_item_results(
                case_id, probabilities_data_siglip, lvlm_output_preds, lvlm_output_probs, args.prob_path
            )
            
            # 实时保存该样本的最终结果
            formatted_name = f"{int(case_id):03d}"
            result_item = {"name": formatted_name, "predicted": final_prediction,
                           "N_path": lvlm_img_path_dict["oc"], 
                           "W_path": lvlm_img_path_dict["ow"]}
            save_final_results(result_item, args.result_path)

    print("全部样本处理完成!")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()