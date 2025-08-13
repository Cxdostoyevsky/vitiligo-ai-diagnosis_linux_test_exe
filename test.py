import os
import json
import torch
import pandas as pd
import time
from tqdm import tqdm
from data_loader import load_test_data
from transformers import AutoProcessor, AutoModel

# --- 终极解决方案: 显式导入，让PyInstaller能够识别 ---
try:
    from transformers import SiglipProcessor, SiglipImageProcessor, SiglipVisionModel
except ImportError:
    pass # 在正常环境中，这些导入可能不是必需的，但在打包时至关重要

from data_generate.generate_datasets_stage_2_binary_cls import two_binary_cls
from data_generate.generate_datasets_stage_2_choice import two_choice
from models import SingleStreamModel, DualStreamModel, QuadStreamModel
from config import (DEVICE, MODEL_PATHS, SIGLIP_MODEL_PATH, SIGLIP_PROCESSOR_PATH, PROBABILITIES_CSV, PREDICTIONS_CSV, INPUT_CONFIG, VOTE_CSV)
import cv2
import argparse
from config import *
from Feature_extran_vis_tools.feature_extractor import extract_woodlamp_edge_features
from Feature_extran_vis_tools.create_highlight_overlays import create_overlay_image
from Feature_extran_vis_tools.batch_processor_V2 import PARAMETER_CONFIGS
import multiprocessing

def format_time(seconds):
    """格式化时间为 分钟:秒 的形式"""
    if seconds < 1:
        return f"{seconds*1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        mins = seconds // 60
        secs = seconds % 60
        return f"{int(mins)} min {secs:.2f} s"

def preprocess_and_generate_feature_maps(csv_path, output_dir):
    """
    读取 data.csv，为每张图片提取边缘特征图，并保存到新位置。
    会生成一个新的 a new csv file pointing to the feature maps.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(csv_path)
    new_rows = []
    configs_by_name = {c['name']: c['params'] for c in PARAMETER_CONFIGS}
    
    # # 定义参数配置
    # sensitive_config = PARAMETER_CONFIGS['config_09_sensitive_and_strong_smoothing']
    # conservative_config = PARAMETER_CONFIGS['config_01_baseline_c10_k5']

    image_paths = pd.concat([df['Wood_path'], df['Normal_path']]).unique()
    
    progress_bar = tqdm(total=len(image_paths), desc="正在预处理图像")

    for img_path in image_paths:
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            print(f"警告: 路径 '{img_path}' 无效或不存在，已跳过。")
            progress_bar.update(1)
            continue
            
        # 尝试使用敏感参数
        result = extract_woodlamp_edge_features(img_path)
        config_name = 'sensitive'
        
        # 如果失败，则使用保守参数
        if result is None:
            result = extract_woodlamp_edge_features(img_path)
            config_name = 'conservative'

        if result and result['gradient_map'] is not None:
            gradient_map = result['gradient_map']
            
            # --- 生成并保存高亮叠加图 ---
            
            # 1. 归一化梯度图
            gradient_map_normalized = cv2.normalize(gradient_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # 2. 判断图片类型来决定颜色
            image_type = 'wood_lamp' if '_W' in os.path.basename(img_path) else 'clinical'
            
            # 3. 创建叠加图
            overlay_img = create_overlay_image(img_path, gradient_map_normalized, image_type)
            
            if overlay_img is not None:
                # 创建保存路径
                base_name = os.path.basename(img_path)
                name, ext = os.path.splitext(base_name)
                # 我们现在保存的是叠加图，文件名可以更直观
                output_filename = f"{name}_overlay.jpg" # 保存为jpg
                output_path = os.path.join(output_dir, output_filename)
                
                # 保存叠加图
                cv2.imwrite(output_path, overlay_img)
            else:
                print(f"警告: 无法为 '{img_path}' 创建叠加图。")

        else:
            print(f"警告: 无法为 '{img_path}' 提取特征。")

        progress_bar.update(1)
        
    progress_bar.close()

    # 现在，我们基于已生成的特征图创建新的DataFrame
    new_df_rows = []
    for index, row in df.iterrows():
        wood_path = row['Wood_path']
        normal_path = row['Normal_path']
        
        wood_base_name = os.path.basename(wood_path)
        wood_name, wood_ext = os.path.splitext(wood_base_name)
        new_wood_filename = f"{wood_name}_overlay.jpg"
        new_wood_path = os.path.join(output_dir, new_wood_filename)

        normal_base_name = os.path.basename(normal_path)
        normal_name, normal_ext = os.path.splitext(normal_base_name)
        new_normal_filename = f"{normal_name}_overlay.jpg"
        new_normal_path = os.path.join(output_dir, new_normal_filename)
        
        # 检查特征图是否真的被创建了
        if os.path.exists(new_wood_path) and os.path.exists(new_normal_path):
             new_df_rows.append({'Wood_path': new_wood_path, 'Normal_path': new_normal_path})
        else:
            print(f"警告: 无法找到 '{new_wood_path}' 或 '{new_normal_path}' 的特征图，将从新CSV中排除该行。")

    new_df = pd.DataFrame(new_df_rows)
    processed_csv_path = os.path.join(os.path.dirname(csv_path), "processed_data.csv")
    new_df.to_csv(processed_csv_path, index=False)
    
    # print(f"预处理完成。新的CSV文件保存在: {processed_csv_path}")

    return processed_csv_path


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
        print(f"处理器加载完成，耗时: {format_time(time.time() - start)}")
        
        # 加载主干模型
        print("加载SigLIP主干模型...")
        start = time.time()
        self.backbone = AutoModel.from_pretrained(SIGLIP_MODEL_PATH).to(self.device)
        self.backbone.eval()
        print(f"主干模型加载完成，耗时: {format_time(time.time() - start)}")
        
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
        
        print(f"所有分类头加载完成，耗时: {format_time(time.time() - start)}")
 
    
    def load_data(self, input_config=None, image_root_dir=None):
        """加载测试数据"""
        print("加载测试数据...")
        start = time.time()
        # 如果没有传入配置，使用默认配置
        if input_config is None:
            from config import INPUT_CONFIG
            input_config = INPUT_CONFIG
        ordered_ids, test_samples = load_test_data(processor=self.processor, input_config=input_config, image_root_dir=image_root_dir)
        print(f"加载 {len(test_samples)} 个测试样本，耗时: {format_time(time.time() - start)}")
        return ordered_ids, test_samples
    
    def run_inference(self, ordered_ids, test_samples):
        """运行推理"""
        print("开始推理...")
        start_time = time.time()
        
        probabilities_data = []
        predictions_data = []
        
        progress_bar = tqdm(ordered_ids, desc="推理进度", unit="样本")
        
        for sample_id in progress_bar:
            sample_data = test_samples[sample_id]
            prob_row = {"id": sample_id}
            pred_row = {"id": sample_id}
            
            for input_type, model in self.models.items():
                if input_type not in sample_data["images"]:
                    print(f"错误: 样本 {sample_id} 缺少 {input_type} 图像数据")
                    continue
                    
                images = sample_data["images"][input_type]
                input_tensors = [img.unsqueeze(0).to(self.device) for img in images]
                
                with torch.no_grad():
                    if input_type in ["oc", "ow"]:
                        outputs = model(*input_tensors)
                    elif input_type in ["oc_ec", "ow_ew", "oc_ow"]:
                        outputs = model(*input_tensors[:2])
                    else:  # oc_ec_ow_ew
                        outputs = model(*input_tensors)
                
                probs = torch.nn.functional.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
                
                prob_row[f"{input_type}_stable"] = probs[0]
                prob_row[f"{input_type}_active"] = probs[1]
                pred_row[input_type] = torch.argmax(outputs).item()
            
            probabilities_data.append(prob_row)
            predictions_data.append(pred_row)
            progress_bar.update(1)
        
        progress_bar.close()
        
        total_time = time.time() - start_time
        print(f"推理完成，共处理 {len(test_samples)} 个样本，耗时: {format_time(total_time)}")
        # print(f"样本处理速度: {len(test_samples)/total_time:.2f} 样本/秒")
        
        return probabilities_data, predictions_data

def save_results(probabilities_data, predictions_data, result_csv_path):
    """保存结果到CSV文件并返回增强的概率数据"""
    print("保存结果...")
    prob_df = pd.DataFrame(probabilities_data)
    pred_df = pd.DataFrame(predictions_data)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(PROBABILITIES_CSV), exist_ok=True)

    
    # 保存CSV
    prob_df.to_csv(PROBABILITIES_CSV, index=False)
    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    
    # print(f"概率结果已保存到: {PROBABILITIES_CSV}")
    # print(f"预测结果已保存到: {PREDICTIONS_CSV}")
    
    stable_cols = [c for c in prob_df.columns if c.endswith('_stable')]
    active_cols = [c for c in prob_df.columns if c.endswith('_active')]

    prob_df['sum_stable'] = prob_df[stable_cols].sum(axis=1)
    prob_df['sum_active'] = prob_df[active_cols].sum(axis=1)

    prob_df['result'] = (prob_df['sum_active'] > prob_df['sum_stable']).astype(int)

    sums = prob_df[['sum_stable', 'sum_active']].values
    prob_df['confidence'] = sums.max(axis=1) / sums.sum(axis=1)

    # df.drop(columns=['sum_stable', 'sum_active'], inplace=True)

    prob_df.to_csv(VOTE_CSV, index=False)
    # --- 以下是新增的逻辑，用于生成比赛要求的格式 ---
    print(f"正在生成比赛要求的格式文件到: {result_csv_path}...")
    result_df = pd.DataFrame()
    result_df['name'] = prob_df['id']
    result_df['predicted'] = prob_df['result'].map({0: '稳定期', 1: '进展期'})

    # 确保结果文件目录存在 (如果路径包含目录)
    if os.path.dirname(result_csv_path):
        os.makedirs(os.path.dirname(result_csv_path), exist_ok=True)

    result_df.to_csv(result_csv_path, index=False)
    print(f"最终比赛结果已保存到: {result_csv_path}")

    # 将增强后的数据转回字典列表格式，供前端使用
    enhanced_probabilities_data = prob_df.to_dict('records')
    return enhanced_probabilities_data
    
    # 将增强后的数据转回字典列表格式，供前端使用
    enhanced_probabilities_data = prob_df.to_dict('records')
    return enhanced_probabilities_data

def main():
    parser = argparse.ArgumentParser(description="模型推理脚本")
    parser.add_argument('--data_path', type=str, default='data.csv', help='包含图像路径的输入CSV文件')
    parser.add_argument('--result_path', type=str, default='result.csv', help='保存最终结果的CSV文件路径')
    args = parser.parse_args()


    device = torch.device(f"cuda:{DEVICE}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    


    #处理数据
    # print("处理数据,生成特征图...")
    original_csv_path = args.data_path
    feature_map_dir = "output_edge_features"
    processed_csv_path = preprocess_and_generate_feature_maps(original_csv_path, feature_map_dir)




    # --- 生成 模型所需的json文件 ---
    # print("模型所需要的json文件...")
    # 使用相对路径，在项目根目录下创建 'json_path' 文件夹
    json_path = 'json_path' 
    master_json_path = os.path.join(json_path, 'master.json')
    
    os.makedirs(json_path, exist_ok=True)

    # 读取原始csv以获取所有图片路径
    df = pd.read_csv(original_csv_path)

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
    two_choice(master_json_path, json_path, doot_dir=feature_map_dir)




    # print("筛选出相关的配置文件...")
    dynamic_input_config = {
                "oc": {
                    "json_path": os.path.join(json_path, "train_binary_cls_1img_OC.json"),
                    "image_types": ["clinical"]
                },
                "ow": {
                    "json_path": os.path.join(json_path, "train_binary_cls_1img_OW.json"),
                    "image_types": ["wood"]
                },
                "oc_ec": {
                    "json_path": os.path.join(json_path, "train_binary_cls_2img_OC_EC.json"),
                    "image_types": ["clinical", "edge_enhanced_clinical"]
                },
                "ow_ew": {
                    "json_path": os.path.join(json_path, "train_binary_cls_2img_OW_EW.json"),
                    "image_types": ["wood", "edge_enhanced_wood"]
                },
                "oc_ow": {
                    "json_path": os.path.join(json_path, "train_binary_cls_2img_OC_OW.json"),
                    "image_types": ["clinical", "wood"]
                },
                "oc_ec_ow_ew": {
                    "json_path": os.path.join(json_path, "train_binary_cls_4img_OC_EC_OW_EW.json"),
                    "image_types": ["clinical", "edge_enhanced_clinical", "wood", "edge_enhanced_wood"]
                }
            }
            
            # 过滤掉不存在的JSON文件
    available_config = {}
    for input_type, config in dynamic_input_config.items():
        if os.path.exists(config["json_path"]):
            available_config[input_type] = config
        else:
            print(f"警告: JSON文件不存在: {config['json_path']}")




    # 初始化推理系统
    inference_system = InferenceSystem(device)
    inference_system.initialize()

    ordered_ids, test_samples = inference_system.load_data(
                    input_config=available_config, 
                    image_root_dir=None
                )
    
    print(f"加载了 {len(ordered_ids)} 个样本ID，样本数量: {len(test_samples)}")
    # print(ordered_ids)
    # print(test_samples.keys())
    
    # 运行推理
    probabilities_data, predictions_data = inference_system.run_inference(ordered_ids, test_samples)

    # 保存结果
    save_results(probabilities_data, predictions_data, args.result_path)
    print("测试完成!")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()