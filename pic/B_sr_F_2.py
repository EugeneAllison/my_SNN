import os
import json
import numpy as np
import matplotlib.pyplot as plt

# 指定路径
base_dirs = [
    "/home/isi/li/my_SNN/output/B_sr_Fmnist/test1",
    "/home/isi/li/my_SNN/output/B_sr_Fmnist/test2",
]
save_path = "/home/isi/li/my_SNN/pic/acc_comparison_Fmnist_shaded.png"

# 存储 x 和 y 数据
acc_values = {}  # {folder_float: [max_acc_e1, max_acc_e2, ...]}
baseline_acc = None  # baseline (0.0) 的最大 acc_e

# 读取 baseline (0.0) 仅从 test1
baseline_path = os.path.join(base_dirs[0], "0.0", "records.json")
if os.path.isfile(baseline_path):
    with open(baseline_path, "r") as f:
        data = json.load(f)
    baseline_acc = max(data["acc_e"]) * 100  # 获取最大 acc_e 并转换为 %

# 遍历 test1 和 test2 目录
for base_dir in base_dirs:
    for folder in sorted(
        os.listdir(base_dir),
        key=lambda x: float(x) if x.replace(".", "", 1).isdigit() else float("inf"),
    ):
        try:
            folder_float = float(folder)
            if not (0.1 <= folder_float <= 4):  # 只选取 0.1 ~ 4
                continue
        except ValueError:
            continue  # 跳过非数字的文件夹

        folder_path = os.path.join(base_dir, folder)
        json_path = os.path.join(folder_path, "records.json")

        if os.path.isfile(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)

            max_acc_e = max(data["acc_e"]) * 100  # 取最大值并转换为 %

            # 存储最大值
            if folder_float not in acc_values:
                acc_values[folder_float] = []
            acc_values[folder_float].append(max_acc_e)

# 计算最大 10 个值的范围和均值
x_values = sorted(acc_values.keys())
y_avg = []
y_min_top10 = []
y_max_top10 = []

for x in x_values:
    values = sorted(acc_values[x], reverse=True)[:19]  # 取前 10 大值
    y_avg.append(np.mean(values))  # 平均值
    y_min_top10.append(min(values))  # 最大 10 个值中的最小值
    y_max_top10.append(max(values))  # 最大 10 个值中的最大值

# 绘制图像
plt.figure(figsize=(8, 5))

# 绘制浅色区域（最大最小范围）
plt.fill_between(x_values, y_min_top10, y_max_top10, color="#A0C4FF", alpha=0.3)

# 绘制深色折线（总体平均）
plt.plot(
    x_values,
    y_avg,
    marker="o",
    linestyle="-",
    color="#3674B5",
    linewidth=2,
    label="Average Accuracy (%)",
)

# 添加 baseline (红色虚线)
if baseline_acc is not None:
    plt.axhline(
        y=baseline_acc,
        color="#D2665A",
        linestyle="dotted",
        label="Original acc (without changing the singular value)",
    )

# 添加 aDFA 研究的最佳准确率 (红色实线)
best_adfa_acc = 87.34
plt.axhline(
    y=best_adfa_acc,
    color="red",
    linestyle="-",
    linewidth=2,
    label="Best accuracy in the aDFA previous research",
)


# 设置标题和标签
plt.xlabel("max Singular Value of B")
plt.ylabel("Avg Accuracy (%)")
plt.title("F-mnist accuracy")
plt.legend()

# 保存图片
os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保目录存在
plt.savefig(save_path, dpi=300)  # 以高分辨率保存

# 显示图像
plt.show()
