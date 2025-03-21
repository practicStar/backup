import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def check_class_distribution(loader, dataset_name="Dataset"):
    """
    检查数据集的类别分布
    :param loader: 数据加载器（DataLoader）
    :param dataset_name: 数据集名称（用于绘图标题）
    """
    all_labels = []
    for _, labels in loader:
        all_labels.extend(labels.numpy())  # 假设 labels 是 Tensor，转换为 numpy 数组
    class_counts = Counter(all_labels)  # 统计每个类别的样本数量

    # 打印类别分布
    print(f"{dataset_name} Class Distribution:")
    for class_id, count in class_counts.items():
        print(f"Class {class_id}: {count} samples")

    # 绘制类别分布图
    plt.figure(figsize=(10, 5))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title(f"{dataset_name} Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.show()



def check_feature_distribution(loader, dataset_name="Dataset", feature_index=0):
    """
    检查数据集的特征分布
    :param loader: 数据加载器（DataLoader）
    :param dataset_name: 数据集名称（用于绘图标题）
    :param feature_index: 要检查的特征索引（如果是多维数据）
    """
    all_features = []
    for inputs, _ in loader:
        if isinstance(inputs, (list, tuple)):  # 如果 inputs 是多个输入（如 eye 和 eeg）
            features = inputs[feature_index].flatten().numpy()  # 选择特定特征并展平
        else:
            features = inputs.flatten().numpy()  # 展平所有特征
        all_features.extend(features)

    # 打印特征统计信息
    print(f"{dataset_name} Feature Distribution (Feature {feature_index}):")
    print(f"Mean: {np.mean(all_features):.4f}")
    print(f"Std: {np.std(all_features):.4f}")
    print(f"Min: {np.min(all_features):.4f}")
    print(f"Max: {np.max(all_features):.4f}")

    # 绘制特征分布直方图
    plt.figure(figsize=(10, 5))
    plt.hist(all_features, bins=50, alpha=0.7, label=dataset_name)
    plt.title(f"{dataset_name} Feature Distribution (Feature {feature_index})")
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


# # 检查训练集和验证集的特征分布
# check_feature_distribution(train_loader, dataset_name="Train Set", feature_index=0)
# check_feature_distribution(val_loader, dataset_name="Validation Set", feature_index=0)
# # 检查训练集和验证集的类别分布
# check_class_distribution(train_loader, dataset_name="Train Set")
# check_class_distribution(val_loader, dataset_name="Validation Set")