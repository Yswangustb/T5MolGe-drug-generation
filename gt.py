if __name__ == "__main__":
    import os
    import subprocess
    import pandas as pd

    # 定义不同的n_blocks值
    n_blocks_values = [0, 1, 2, 3, 4]

    # 训练脚本的路径
    script_path = "transfer.py"
    # 模型权重的路径
    model_path = "cond_gpt/weights/scaffold_mamba_layer4/epoch_10_scaffold_mamba_layer4.pt"
    # 运行名称前缀
    run_name_prefix = "transfer_scaffold_mamba4_"

    # 迭代不同的n_blocks值
    for n_blocks in n_blocks_values:
        print(n_blocks)
        run_name = run_name_prefix + str(n_blocks)
        # 构建命令行命令
        command = [
            "python", script_path,
            "--run_name", run_name,
            "--model_path", model_path,
            "--n_blocks", str(n_blocks),
            "--n_layer",'4',
            "--scaffold",
        ]
        # 执行命令
        result = subprocess.run(command, capture_output=True, text=True)