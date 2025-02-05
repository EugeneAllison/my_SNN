import os
import json
import matplotlib.pyplot as plt

# 指定路径
# base_dir = "/home/isi/li/my_SNN/output/B_sr_amp04/test1"
# save_path = "/home/isi/li/my_SNN/pic/acc_comparison_mnist.png"
base_dir = "/home/isi/li/my_SNN/output/B_sr_Fmnist/test1"
save_path = "/home/isi/li/my_SNN/pic/acc_comparison_Fmnist.png"
# base_dir = "/home/isi/li/my_SNN/output/B_sr"
# save_path = "/home/isi/li/my_SNN/pic/acc_comparison_NG.png"
# 存储 x 和 y 数据
x_values = []
y_values = []
baseline_acc = None  # 存储 baseline (0.0) 的 acc_e 最大值

# 先读取 baseline (0.0)
baseline_path = os.path.join(base_dir, "0.0", "records.json")
if os.path.isfile(baseline_path):
    with open(baseline_path, "r") as f:
        data = json.load(f)
    baseline_acc = max(data["acc_e"]) * 100  # 获取 acc_e 最大值，并转换为百分比

# 遍历 0.1 ~ 4 的文件夹
for folder in sorted(os.listdir(base_dir), key=lambda x: float(x)):
    try:
        folder_float = float(folder)
        if not (0.1 <= folder_float <= 4):  # 只选取 0.1 ~ 4
            continue
    except ValueError:
        continue  # 跳过非数字的文件夹

    folder_path = os.path.join(base_dir, folder)
    json_path = os.path.join(folder_path, "records.json")

    # 确保 records.json 存在
    if os.path.isfile(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)

        # 获取 acc_e 中的最大值，并转换为百分比
        max_acc_e = max(data["acc_e"]) * 100

        # 记录 x 和 y
        x_values.append(folder_float)
        y_values.append(max_acc_e)

# 绘制图像
plt.figure(figsize=(8, 5))

# 绘制数据点
plt.plot(
    x_values,
    y_values,
    marker="o",
    linestyle="-",
    color="#3674B5",
    label="Max accuracy (%)",
)

# 添加 baseline (红色虚线)
if baseline_acc is not None:
    plt.axhline(
        y=baseline_acc, color="#D2665A", linestyle="dotted", label="Original acc"
    )

# 设置标题和标签
plt.xlabel("Spectral Radius of B")
plt.ylabel("Max accuracy (%)")
plt.title("F-mnist accuracy")
plt.legend()

# 保存图片
os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保目录存在
plt.savefig(save_path, dpi=300)  # 以高分辨率保存

# 显示图像
plt.show()
