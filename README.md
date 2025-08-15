# Vitiligo Progression Phase Prediction Program

## 1. Program Description

This program is designed for a medical image analysis competition. It predicts the progression phase of vitiligo (stable or active) based on clinical and Wood's lamp images.

The system employs a multi-stream deep learning model, utilizing the SigLIP vision model as its backbone. It first preprocesses the input images to extract and highlight edge features, then feeds these features into a series of trained neural network classifiers. Finally, it aggregates the results from multiple model heads to produce a final prediction through a voting mechanism.

All necessary models and weights are bundled directly within the executable file, requiring no separate download or configuration.

## 2. Environment Requirements

The program is designed to run on a Linux server with the following specifications:

-   **Operating System**: Linux
-   **CPU**: Intel Core i7-10700K or equivalent
-   **GPU**: NVIDIA GeForce RTX 3090 (24GB VRAM)
-   **CUDA Version**: 12.3 or higher (Compatible with PyTorch compiled for CUDA 12.3)
-   **Python Dependencies**: All required Python libraries are bundled within the executable. No manual installation of Python or packages is needed.

## 3. How to Run

Follow these steps to run the inference program:

### Step 1: Prepare Your Data

Ensure you have a CSV file containing the paths to the image pairs. The CSV file should have two columns: `Wood_path` and `Normal_path`, pointing to the Wood's lamp and clinical images, respectively.

Example `data.csv`:
```csv
Wood_path,Normal_path
picture/001_W.jpg,picture/001_N.jpg
picture/002_W.jpg,picture/002_N.jpg
```

### Step 2: Place Files

Place the generated executable file (`test`) and your data files (the CSV and the `picture` directory) in the same directory. The structure should look like this:

```
your_working_directory/
├── test                # The executable file
├── data.csv            # Your input CSV file
└── picture/            # Directory containing all your images
    ├── 001_W.jpg
    ├── 001_N.jpg
    └── ...
```

### Step 3: Execute the Program

Open a terminal in your working directory and run the following command.

**Command Format:**

```bash
./test --data_path <your_input_csv_path> --result_path <your_output_csv_path>
```

**Example Usage:**

```bash
文件夹目录为：
test_dist/
├── _internal/          #模型文件夹
├── test               # 可执行文件

进入可执行文件夹目录：test_dist
cd /test_dist 

批量测试文件: --data_path:测试数据路径  --result_path 存放结果的文件路径
./test --data_path data.csv --result_path result.csv
```

### Step 4: Get the Result

The program will start processing. It will print progress information to the console. Once finished, a file named `result.csv` (or the name you specified) will be created in the same directory. This file contains the final predictions.

Example `result.csv` output:
```csv
name,predicted
001,稳定期
002,进展期
```

---
*This README provides all necessary information for running the program as per the competition guidelines.* 

### important information
生成的文件中：
 dist:
    test_dist:
        test:主可执行文件，程序的入口，核心任务就是创建一个临时的python环境，可以运行python代码。
        _internal:包括其他的全部文件
            model_ckpt
            python库和依赖项：一大堆 .so, .pyd, .dll 文件和文件夹(import的所有库函数，pyinstaller把这些库函数从开发环境中直接复制了过来)
            base_liarary.zip:python模拟压缩包（把原来的脚本比如test.py,model.py等全部压缩到这个zip里面，减少体积，在上面给的可执行文件中会从这里进行解码并执行代码）

build：临时文件夹：
    1：分析文件: PyInstaller 分析你的代码后生成的依赖关系列表
    2：日志和警告文件 (warn-test.txt等): 这里记录了打包过程中遇到的问题。比如，如果打包失败，或者某个库没找到，来这个文件里看日志，是排查问题的首选。
    3：中间文件: 在生成最终的 exe 和 COLLECT 之前，所有被处理过的脚本和文件都会临时存放在这里
# vitiligo-ai-diagnosis_linux_test_exe
