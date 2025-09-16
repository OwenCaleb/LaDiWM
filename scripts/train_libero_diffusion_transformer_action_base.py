"""
训练 LIBERO 数据集上的扩散 Transformer 动作模型的启动脚本。

- 依据 --suite 选择数据集任务套件，自动收集其下所有任务的 train/val 目录
- 通过 Python 模块入口 engine.train_diffusion_transformer_action 启动训练（通常为 Hydra 配置驱动）
- 关闭 HuggingFace tokenizer 的并行以减少无关告警

用法示例：
    python scripts/train_libero_diffusion_transformer_action_base.py --suite libero_90

注意事项：
- 请将 root_dir 修改为本地实际的数据根目录，目录结构需满足 {suite_name}/*/(train|val)/
- gpu_ids 以列表形式写明可用 GPU 的编号，并原样传给 Hydra 配置项 train_gpus
- 该脚本仅收集数据路径与拼装命令，不负责具体的训练细节
"""

import os
import argparse
from glob import glob


# 关闭 tokenizer 并行日志，避免多进程冲突与不必要的警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 解析命令行参数：选择要使用的 LIBERO 任务套件
parser = argparse.ArgumentParser()
parser.add_argument(
    "--suite",
    default="libero_90",
    choices=[
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_90",
        "libero_100",
        "libero_long",
    ],
    help="期望使用的任务套件名称，libero_10 是 libero_long 的别名（若在其它脚本中引用）。",
)
args = parser.parse_args()

# 训练配置名称：与 conf/ 中的 Hydra 配置对应（例如 config-name 会选择相应 yaml 配置）
CONFIG_NAME = "libero_diff_transformer_action"

# 指定参与训练的 GPU 编号列表，例如 [0, 1] 使用两张卡
gpu_ids = [0]

# 数据根目录：需包含 {suite_name}/*/(train|val)/ 的层级结构
root_dir = "/media/huang/T7/data/atm_libero/"
suite_name = args.suite

# 可选示例：当 suite_name 为 "libero_100" 时，合并 libreo_90 与 libero_10 的数据并调整 epoch
# 保留为注释以供参考，避免改变当前默认行为
# if suite_name == "libero_100":
#     EPOCH = 301
#     train_dataset_list = glob(os.path.join(root_dir, "libero_90/*/train/")) + glob(os.path.join(root_dir, "libero_10/*/train/"))
#     val1_dataset_list = glob(os.path.join(root_dir, "libero_90/*/val/")) + glob(os.path.join(root_dir, "libero_10/*/val/"))
# else:
#     EPOCH = 101

# 根据所选套件，收集所有任务的 train 和 val 子目录，传递给训练引擎
train_dataset_list = glob(os.path.join(root_dir, f"{suite_name}/*/train/"))
val1_dataset_list = glob(os.path.join(root_dir, f"{suite_name}/*/val/"))

# 组装训练命令：
# - 通过 `python -m` 方式调用 engine.train_diffusion_transformer_action 模块
# - 传入 Hydra 的 config-name、train_gpus、数据集路径等配置
# - 注意：这里将 Python 列表以字符串形式（外部再包一层双引号）传入，以兼容下游解析逻辑
command = (
    f'python -m engine.train_diffusion_transformer_action --config-name={CONFIG_NAME} '
    f'train_gpus="{gpu_ids}" '
    # f'experiment={CONFIG_NAME}_{suite_name.replace("_", "-")}_ep{EPOCH} '
    # f'epochs={EPOCH} '
    f'train_dataset="{train_dataset_list}" val_dataset="{val1_dataset_list}" '
)

# 执行命令启动训练；如需更细粒度控制与错误处理，可改用 subprocess.run
os.system(command)
