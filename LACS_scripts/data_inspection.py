# scripts/lacs/data_inspection.py
"""
数据检查脚本：检查 VisRAG-Ret-Train 数据集的结构
"""
import os
from datasets import load_dataset
from PIL import Image
import json

def inspect_dataset(data_dir: str = "./data/VisRAG-Ret-Train-In-domain-data"):
    """检查数据集的基本结构和样本内容"""
    
    print("=" * 80)
    print("数据检查：VisRAG-Ret-Train-In-domain-data")
    print("=" * 80)
    
    # 加载数据集
    print("\n[1] 加载数据集...")
    dataset = load_dataset(
        "parquet",
        data_files=f"{data_dir}/data/train-*.parquet",
        split="train"
    )
    
    print(f"✓ 数据集大小: {len(dataset)} 个样本")
    print(f"✓ 数据集特征: {list(dataset.features.keys())}")
    
    # 检查第一个样本
    print("\n[2] 检查第一个样本...")
    sample = dataset[0]
    print(f"✓ 样本字段: {list(sample.keys())}")
    
    for key, value in sample.items():
        if key == "image":
            if isinstance(value, Image.Image):
                print(f"  - {key}: PIL.Image, size={value.size}, mode={value.mode}")
            else:
                print(f"  - {key}: {type(value)}")
        else:
            print(f"  - {key}: {type(value).__name__} = {str(value)[:100]}")
    
    # 统计信息
    print("\n[3] 数据集统计信息...")
    sources = {}
    for i in range(min(1000, len(dataset))):  # 采样检查
        source = dataset[i].get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1
    
    print(f"✓ 前1000个样本的 source 分布:")
    for source, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"    {source}: {count}")
    
    # 检查是否有重复的 query-image 对
    print("\n[4] 检查数据唯一性...")
    unique_pairs = set()
    for i in range(min(1000, len(dataset))):
        query = dataset[i]["query"]
        # 使用 query 作为唯一标识（因为 image 对象无法直接 hash）
        unique_pairs.add(query)
    
    print(f"✓ 前1000个样本中唯一 query 数量: {len(unique_pairs)}")
    
    print("\n" + "=" * 80)
    print("数据检查完成！")
    print("=" * 80)
    
    return dataset

if __name__ == "__main__":
    dataset = inspect_dataset()