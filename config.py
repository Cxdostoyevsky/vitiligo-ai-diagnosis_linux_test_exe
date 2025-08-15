# 配置参数
import os
import sys

# --- 新增: 资源路径处理 ---
def get_resource_path(relative_path):
    """
    获取资源的绝对路径，兼容PyInstaller打包后的环境。
    在开发环境中，它返回相对于当前文件的路径；
    在PyInstaller打包的exe中，它返回相对于可执行文件所在目录的路径。
    """
    if getattr(sys, 'frozen', False):
        # 如果是打包后的环境
        base_path = sys._MEIPASS
    else:
        # 如果是正常的开发环境
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(base_path, relative_path)

# CUDA_DEVICE
DEVICE = 0

# 基础路径 (现在使用新的函数来获取)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 这个保持不变，主要用于非模型文件的相对路径

MODEL_PATHS = {
    "oc": get_resource_path(os.path.join("model_ckpt", "siglip2", "final_trainable_params_OC.pth")),
    "ow": get_resource_path(os.path.join("model_ckpt", "siglip2", "final_trainable_params_OW.pth")),
    "oc_ec": get_resource_path(os.path.join("model_ckpt", "siglip2", "final_trainable_params_OC_EC.pth")),
    "ow_ew": get_resource_path(os.path.join("model_ckpt", "siglip2", "final_trainable_params_OW_EW.pth")),
    "oc_ow": get_resource_path(os.path.join("model_ckpt", "siglip2", "final_trainable_params_OC_OW.pth")),
    "oc_ec_ow_ew": get_resource_path(os.path.join("model_ckpt", "siglip2", "final_trainable_params_OC_OW_EC_EW.pth")),
}

# SigLIP 主干模型路径
SIGLIP_MODEL_PATH = get_resource_path(os.path.join("model_ckpt", "siglip2-base-patch16-512"))

# --- 新增: LVLM 模型路径 ---
LVLM_MODEL_PATH = get_resource_path(os.path.join("model_ckpt", "med_vl"))

# 处理器路径（与模型路径相同）
SIGLIP_PROCESSOR_PATH = SIGLIP_MODEL_PATH

# 图像根目录 (这个路径由外部提供，所以保持不变)
IMAGE_ROOT_DIR = "/hdd/chenxi/bzt/uploads_images"

# 输出文件路径 (这些路径是相对于当前工作目录，所以不需要改变)
PROBABILITIES_CSV = os.path.join("results", "probabilities.csv")
PREDICTIONS_CSV = os.path.join("results", "predictions.csv")
VOTE_CSV = os.path.join("results", "vote_results.csv")

# --- 新增: LVLM 结果文件路径 ---
LVLM_PROBABILITIES_CSV = os.path.join("results", "probabilities_lvlm.csv")
LVLM_PREDICTIONS_CSV = os.path.join("results", "predictions_lvlm.csv")

# 输入类型映射和对应的JSON路径
INPUT_CONFIG = {
    "oc": {
        "json_path": "/hdd/chenxi/bzt/uploads_images/20250809_104721_546043/train_binary_cls_1img_OC.json",
        "image_types": ["clinical"]
    },
    "ow": {
        "json_path": "/hdd/chenxi/bzt/uploads_images/20250809_104721_546043/train_binary_cls_1img_OW.json",
        "image_types": ["wood"]
    },
    "oc_ec": {
        "json_path": "/hdd/chenxi/bzt/uploads_images/20250809_104721_546043/train_binary_cls_2img_OC_EC.json",
        "image_types": ["clinical", "edge_enhanced_clinical"]
    },
    "ow_ew": {
        "json_path": "/hdd/chenxi/bzt/uploads_images/20250809_104721_546043/train_binary_cls_2img_OW_EW.json",
        "image_types": ["wood", "edge_enhanced_wood"]
    },
    "oc_ow": {
        "json_path": "/hdd/chenxi/bzt/uploads_images/20250809_104721_546043/train_binary_cls_2img_OC_OW.json",
        "image_types": ["clinical", "wood"]
    },
    "oc_ec_ow_ew": {
        "json_path": "/hdd/chenxi/bzt/uploads_images/20250809_104721_546043/train_binary_cls_4img_OC_EC_OW_EW.json",
        "image_types": ["clinical", "edge_enhanced_clinical", "wood", "edge_enhanced_wood"]
    },
}