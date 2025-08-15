import json
from PIL import Image
import torch

# --- 1. 配置与常量定义 ---

# 定义四种图片类型及其在提问文本中对应的名称
IMAGE_TYPE_DETAILS = {
    'oc': "original clinical image",
    'ow': "original Wood's lamp image",
    'ec': "edge-enhanced clinical image",
    'ew': "edge-enhanced Wood's lamp image",
}

# 直接定义需要生成的6种图片组合
# 组合使用 'oc', 'ow', 'ec', 'ew' 作为键
TARGET_COMBINATIONS = [
    ['oc'],
    ['ow'],
    ['oc', 'ec'],
    ['ow', 'ew'],
    ['oc', 'ow'],
    ['oc', 'ec', 'ow', 'ew']
]

# --- 2. 辅助函数，用于生成提问文本中的图片描述 ---

def _generate_image_description(image_keys):
    """根据图片类型key的列表，动态生成符合语法的图片描述部分。
    
    例如:
    - ['oc'] -> "the original clinical image <image>"
    - ['oc', 'ow'] -> "the original clinical image <image>, and the original Wood's lamp image <image>"
    """
    if not image_keys:
        return ""
        
    # 获取每个key对应的完整名称
    full_names = [IMAGE_TYPE_DETAILS[key] for key in image_keys]
    
    # 为每个名称添加<image>占位符
    name_list_with_placeholder = [f"the {name} <image>" for name in full_names]

    if len(name_list_with_placeholder) == 1:
        return name_list_with_placeholder[0]
    else:
        # 使用逗号连接除最后一项外的所有项，并与最后一项用 "and" 连接
        return ", ".join(name_list_with_placeholder[:-1]) + f", and {name_list_with_placeholder[-1]}"


# --- 3. 核心函数实现 ---

def create_prompts(case_id: str, img_path_dict: dict, processor):
    """
    为单个病例生成12种格式的prompt数据。
    这个版本会预加载所有图像，以避免重复I/O。

    Args:
        case_id (str): 病例的唯一标识符。
        img_path_dict (dict): 包含四种图片路径的字典。
        processor: 用于处理图像的图像预处理器。

    Returns:
        tuple: 包含两项内容：
            - list: 包含12个prompt组合字典的列表。
            - dict: 一个字典，键是图像路径，值是已加载和处理的图像张量。
    """
    data_list = []
    loaded_images = {}

    # 1. 收集所有不重复的有效图像路径并一次性加载
    unique_paths = set(p for p in img_path_dict.values() if p)
    for img_path in unique_paths:
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = processor(image, mode="eval")
            loaded_images[img_path] = image_tensor
        except FileNotFoundError:
            print(f"警告: 图像文件未找到: {img_path}，跳过加载。")
            # 可以在这里存入一个占位符，如果需要的话
            # loaded_images[img_path] = torch.zeros((3, 224, 224)) # 假设尺寸
            continue
    
    # 2. 遍历预设的6种图片组合
    for combo_keys in TARGET_COMBINATIONS:
        # --- a. 准备当前组合所需的数据 ---
        
        # 根据组合键列表，从输入字典中获取对应的图片路径列表
        try:
            image_paths = [img_path_dict[key] for key in combo_keys]
            # 确保所有路径都成功加载了
            if not all(p in loaded_images for p in image_paths):
                print(f"警告: 跳过组合 {combo_keys}，因为其中部分图像未能成功加载。")
                continue
        except KeyError as e:
            print(f"警告: 跳过组合 {combo_keys}，因为在 img_path_dict 中缺少键: {e}")
            continue
            
        # 生成用于文件名和ID的组合字符串, e.g., "oc_ec"
        combo_str_for_id = "_".join(combo_keys)
        
        # 生成用在提问文本中的图片描述, e.g., "the original clinical image <image>, and the edge-enhanced clinical image <image>"
        img_desc_for_prompt = _generate_image_description(combo_keys)

        # --- b. 生成“问答题” (Binary Classification) ---
        
        qa_question = (f"Based on the input of {img_desc_for_prompt}, analyze and determine the vitiligo "
                       f"progression phase: stable stage vitiligo or active stage vitiligo?")
        
        qa_entry = {
            "case_id": case_id,
            "id": f"{case_id}_binary_cls_{combo_str_for_id}",
            "conversations": [
                {"from": "human", "value": qa_question}
                # 根据要求，测试数据不包含 "gpt" 的答案部分
            ],
            "image": image_paths
        }
        data_list.append(qa_entry)

        # --- c. 生成“选择题” (Multiple Choice) ---
        
        mcq_question = (
            f"Based on the input of {img_desc_for_prompt}, analyze and determine the vitiligo "
            f"progression phase:\n"
            f"A. stable stage vitiligo\n"
            f"B. active stage vitiligo\n"
            f"Please answer option A or B directly."
        )

        mcq_entry = {
            "case_id": case_id,
            "id": f"{case_id}_choice_{combo_str_for_id}",
            "conversations": [
                {"from": "human", "value": mcq_question}
                 # 根据要求，测试数据不包含 "gpt" 的答案部分
            ],
            "image": image_paths
        }
        data_list.append(mcq_entry)

    return data_list, loaded_images

# --- 4. 使用示例 ---
if __name__ == '__main__':
    # 模拟输入
    example_case_id = "..."
    example_img_paths = {
        'oc': '...',
        'ow': '...',
        'ec': '...',
        'ew': '...'
    }

    # 调用函数生成数据
    medm_test_data = create_prompts(example_case_id, example_img_paths)

    # 打印结果进行验证
    print(f"成功为病例 '{example_case_id}' 生成了 {len(medm_test_data)} 条prompt数据。")
    
    # 打印其中一个问答题的例子
    print("\n--- 问答题示例 (oc_ec_ow_ew) ---")
    print(json.dumps(medm_test_data[10], indent=2, ensure_ascii=False))

    # 打印其中一个选择题的例子
    print("\n--- 选择题示例 (oc_ow) ---")
    print(json.dumps(medm_test_data[9], indent=2, ensure_ascii=False))