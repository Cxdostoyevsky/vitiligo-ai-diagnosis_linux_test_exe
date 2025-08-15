import argparse
import json
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lvlm.utils.arguments import set_seed
from lvlm.dataset.dataset import MultiModalDataset, DataCollatorForMultiModalDataset
from lvlm.model.modeling_lvlm import LVLMForConditionalGeneration
from test_generate import create_prompts


DEVICE = 0
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROBABILITIES_CSV = os.path.join(BASE_DIR, "results/probabilities.csv")
PREDICTIONS_CSV = os.path.join(BASE_DIR, "results/predictions.csv")
os.makedirs(os.path.dirname(PREDICTIONS_CSV), exist_ok=True)


def main(args):
    # 设置设备
    device = torch.device(f"cuda:{DEVICE}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化推理系统
    model_dtype = torch.float16 if args.model_dtype == "float16" else (torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float32)
    model = LVLMForConditionalGeneration.from_pretrained(args.resume_from_checkpoint)
    model.to(dtype=model_dtype, device=device)
    model.eval()

    # 加载数据
    case_id = "test_01"
    img_path_dict = {
        "oc": "/hdd/common/datasets/medical-image-analysis/tzb/dataset/images/original/active_stage/235/clinical/0.jpg",
        "ow": "/hdd/common/datasets/medical-image-analysis/tzb/dataset/images/original/active_stage/235/wood_lamp/0.jpg",
        "ec": "/hdd/common/datasets/medical-image-analysis/tzb/dataset/images/edge_enhanced/active_stage/235/clinical/0.jpg",
        "ew": "/hdd/common/datasets/medical-image-analysis/tzb/dataset/images/edge_enhanced/active_stage/235/wood_lamp/0.jpg",
    }
    data_list = create_prompts(case_id, img_path_dict)

    # 运行推理
    with torch.no_grad():
        output_dict = {}
        prob_dict = {}
        for data in tqdm(data_list):
            train_dataset = MultiModalDataset(
                model=model,
                data=[data],
                data_arguments=args,
                mode="eval",
            )
            data_collator = DataCollatorForMultiModalDataset(tokenizer=model.tokenizer, mode="eval")
            data_loader = DataLoader(
                train_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=data_collator,
            )

            for batch in data_loader:
                for k, v in batch.items():
                    if v is not None:
                        batch[k] = v.to(device)
                        if k == "image" or k == "image3d":
                            batch[k] = v.to(dtype=model_dtype, device=device)

                output_ids, output_scores = model.generate(
                    **batch,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True if args.temperature > 0 else False,
                    num_beams=args.num_beams,
                    temperature=args.temperature,
                    pad_token_id=model.tokenizer.pad_token_id,
                    eos_token_id=model.tokenizer.eos_token_id,
                )
                outputs = model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                prob = float(output_scores[0][0].softmax(dim=-1)[output_ids[0][0]])
                # print(f"{data['id']}: {data['conversations'][0]['value']}")
                # print(f"输出: {outputs[0]}, {prob}")

                if data["case_id"] not in output_dict:
                    output_dict[data["case_id"]] = {}
                    prob_dict[data["case_id"]] = {}
                sample_id = data["id"].replace(f"{data['case_id']}_", "")
                if "binary_cls" in sample_id:
                    sample_id = sample_id.replace("binary_cls_", "")
                    if sample_id not in output_dict[data["case_id"]]:
                        output_dict[data["case_id"]][sample_id] = [-1, -1]
                        prob_dict[data["case_id"]][f"{sample_id}_binary_cls"] = prob
                    if outputs[0].strip().lower() == "stable stage vitiligo":
                        output_dict[data["case_id"]][sample_id][0] = 1
                    elif outputs[0].strip().lower() == "active stage vitiligo":
                        output_dict[data["case_id"]][sample_id][0] = 0
                else:
                    sample_id = sample_id.replace("choice_", "")
                    if sample_id not in output_dict[data["case_id"]]:
                        output_dict[data["case_id"]][sample_id] = [-1, -1]
                        prob_dict[data["case_id"]][f"{sample_id}_choice"] = prob
                    if outputs[0].strip().lower() == "a":
                        output_dict[data["case_id"]][sample_id][1] = 1
                    elif outputs[0].strip().lower() == "b":
                        output_dict[data["case_id"]][sample_id][1] = 0

    # with open(os.path.join(args.output_dir, "output.json"), "w") as f:
    #     json.dump(output_dict, f, indent=4)
    # with open(os.path.join(args.output_dir, "probabilities.json"), "w") as f:
    #     json.dump(prob_dict, f, indent=4)

    # 判断一致性
    for case_id in output_dict:
        for sample_id in output_dict[case_id]:
            if output_dict[case_id][sample_id][0] != output_dict[case_id][sample_id][1]:
                output_dict[case_id][sample_id] = [-1, -1]

    # 格式化输出
    output_dict_format = {}
    for case_id in output_dict:
        output_dict_format[case_id] = {}
        for sample_id in output_dict[case_id]:
            output_dict_format[case_id][f"{sample_id}_binary_cls"] = output_dict[case_id][sample_id][0]
            output_dict_format[case_id][f"{sample_id}_choice"] = output_dict[case_id][sample_id][1]

    # 保存结果
    pred_df = pd.DataFrame.from_dict(output_dict_format, orient="index")
    pred_df.index.name = "id"
    pred_df.reset_index(inplace=True)
    pred_df.to_csv(PREDICTIONS_CSV, index=False)

    prob_df = pd.DataFrame.from_dict(prob_dict, orient="index")
    prob_df.index.name = "id"
    prob_df.reset_index(inplace=True)
    prob_df.to_csv(PROBABILITIES_CSV, index=False)

    print(f"预测结果已保存到: {PREDICTIONS_CSV}")
    print(f"概率结果已保存到: {PROBABILITIES_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dtype", type=str, default="bfloat16")
    parser.add_argument("--data_path", type=str, default="docs/example_2d_inference.json")
    parser.add_argument("--conv_version", type=str, default="qwen2")
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--image3d_path", type=str, default="")
    parser.add_argument("--resume_from_checkpoint", type=str, default="/hdd/shiym/work_dirs/tzb/train_129_medm_384_4096-finetune")
    parser.add_argument("--output_dir", type=str, default="/hdd/common/datasets/medical-image-analysis/tzb/test_code_sym/results/")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    set_seed(42)

    main(args)
