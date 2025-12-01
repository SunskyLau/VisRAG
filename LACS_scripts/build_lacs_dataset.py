# scripts/lacs/build_lacs_dataset.py
"""
LACS 数据构建脚本：构建 Top-5 RAG 模拟样本
使用 VisRAG-Ret 检索模型挖掘难负样本
使用路径方式保存图片（而非 base64），节省存储空间
"""
import os
import json
import random
import hashlib
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from typing import List, Dict, Tuple, Set

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ============ 参考 demo.py 的实现 ============

def weighted_mean_pooling(hidden, attention_mask):
    """加权平均池化"""
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    d = attention_mask_.sum(dim=1, keepdim=True).float()
    reps = s / d
    return reps

@torch.no_grad()
def encode(text_or_image_list):
    """编码文本或图像（使用全局 model 和 tokenizer）"""
    global model, tokenizer
    
    if isinstance(text_or_image_list[0], str):
        inputs = {
            "text": text_or_image_list,
            'image': [None] * len(text_or_image_list),
            'tokenizer': tokenizer
        }
    else:
        inputs = {
            "text": [''] * len(text_or_image_list),
            'image': text_or_image_list,
            'tokenizer': tokenizer
        }
    
    outputs = model(**inputs)  # **是解包操作，将字典中的键值对解包为关键字参数
    attention_mask = outputs.attention_mask
    hidden = outputs.last_hidden_state
    
    reps = weighted_mean_pooling(hidden, attention_mask)
    embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
    return embeddings

# ============ 全局变量（模型和tokenizer） ============
model = None
tokenizer = None

def load_model(model_name_or_path: str = "openbmb/VisRAG"):
    """加载 VisRAG 检索模型"""
    global model, tokenizer
    
    print(f"[1] 加载检索模型: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to("cuda")
    model.eval()
    print("✓ 模型加载完成")

# ============ 数据构建逻辑 ============

class LACSDatasetBuilder:
    """LACS 数据集构建器"""
    
    def __init__(
        self,
        data_dir: str = "./data/VisRAG-Ret-Train-In-domain-data",
        topk: int = 5,
        batch_size: int = 32,
        max_samples: int = None,
        cache_dir: str = "./data/lacs_cache",
    ):
        self.data_dir = data_dir
        self.topk = topk
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.cache_dir = cache_dir
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 创建图片存储目录
        self.image_dir = os.path.join(self.cache_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        
        # 加载数据集
        print(f"\n[2] 加载数据集: {data_dir}")
        self.dataset = load_dataset(
            "parquet",
            data_files=f"{data_dir}/data/train-*.parquet",
            split="train"
        )
        
        if self.max_samples is not None:
            self.dataset = self.dataset.select(range(min(self.max_samples, len(self.dataset))))
        
        print(f"✓ 数据集大小: {len(self.dataset)} 个样本")
    
    def _get_image(self, idx: int) -> Image.Image:
        """获取并转换图像为 RGB 格式"""
        image = self.dataset[idx]["image"]
        return image.convert("RGB")
    
    def build_corpus_embeddings(self, save_path: str = None):
        """构建整个数据集的 corpus embeddings（支持断点续传）"""
        if save_path is None:
            save_path = os.path.join(self.cache_dir, "corpus_embeddings.npy")
        
        # 检查是否已存在
        if os.path.exists(save_path):
            print(f"\n[3] 发现已存在的 Corpus Embeddings: {save_path}")
            corpus_embeddings = np.load(save_path)
            print(f"✓ 加载完成，形状: {corpus_embeddings.shape}")
            return corpus_embeddings
        
        print(f"\n[3] 构建 Corpus Embeddings...")
        
        embeddings_list = []
        embedding_dim = None  # 动态获取维度
        
        # 批量编码图像
        for i in tqdm(range(0, len(self.dataset), self.batch_size), desc="编码图像"):
            batch_end = min(i + self.batch_size, len(self.dataset))
            batch_images = [self._get_image(j) for j in range(i, batch_end)]
            
            # 编码批次
            try:
                batch_embeddings = encode(batch_images)
                embeddings_list.append(batch_embeddings)
                if embedding_dim is None:
                    embedding_dim = batch_embeddings.shape[1]
            except Exception as e:
                print(f"\n警告: 批次 {i}-{batch_end} 编码失败: {e}")
                # 使用已知维度或跳过
                if embedding_dim is not None:
                    dummy_emb = np.zeros((len(batch_images), embedding_dim))
                    embeddings_list.append(dummy_emb)
                else:
                    raise RuntimeError(f"第一个批次编码失败，无法确定 embedding 维度: {e}")
        
        # 合并所有 embeddings
        corpus_embeddings = np.vstack(embeddings_list)
        print(f"✓ Corpus embeddings 形状: {corpus_embeddings.shape}")
        
        # 保存
        np.save(save_path, corpus_embeddings)
        print(f"✓ 已保存到: {save_path}")
        
        return corpus_embeddings
    
    def retrieve_topk(
        self, 
        query: str, 
        corpus_embeddings: np.ndarray,
        exclude_indices: List[int] = None,
        topk: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """检索 Top-K 图片（排除指定索引）"""
        # 编码 query（加上 instruction）
        INSTRUCTION = "Represent this query for retrieving relevant documents: "
        query_with_instruction = INSTRUCTION + query
        query_embedding = encode([query_with_instruction])
        
        # 计算相似度
        similarities = query_embedding @ corpus_embeddings.T  # [1, N]
        similarities = similarities[0]  # [N]
        
        # 排除指定索引
        if exclude_indices:
            for idx in exclude_indices:
                similarities[idx] = -1e9
        
        # 获取 Top-K
        actual_topk = min(topk, len(corpus_embeddings))
        topk_indices = np.argsort(similarities)[::-1][:actual_topk]
        topk_values = similarities[topk_indices]
        
        return topk_indices, topk_values
    
    def _compute_image_hash(self, image: Image.Image) -> str:
        """计算图片的 MD5 哈希值，用于去重"""
        return hashlib.md5(image.tobytes()).hexdigest()
    
    def _try_add_negative(
        self,
        neg_idx: int,
        images: List[Image.Image],
        labels: List[int],
        image_indices: List[int],
        used_image_hashes: Set[str],
    ) -> bool:
        """
        尝试添加一个负样本，如果图片不重复则添加成功。
        
        Args:
            neg_idx: 负样本在数据集中的索引
            images: 图片列表（会被修改）
            labels: 标签列表（会被修改）
            image_indices: 索引列表（会被修改）
            used_image_hashes: 已使用的图片哈希集合（会被修改）
        
        Returns:
            bool: 是否成功添加
        """
        # 跳过已使用的索引
        if neg_idx in image_indices:
            return False
        
        # 获取图片并计算哈希
        try:
            neg_image = self._get_image(neg_idx)
            neg_image_hash = self._compute_image_hash(neg_image)
        except Exception:
            return False
        
        # 如果图片内容重复，跳过
        if neg_image_hash in used_image_hashes:
            return False
        
        # 添加不重复的负样本
        images.append(neg_image)
        labels.append(0)
        image_indices.append(neg_idx)
        used_image_hashes.add(neg_image_hash)
        return True
    
    def build_sample(
        self,
        query_idx: int,
        corpus_embeddings: np.ndarray,
    ) -> Dict:
        """
        为单个 query 构建 Top-5 样本。
        
        去重逻辑：
        1. 按检索顺序遍历候选图片
        2. 使用图片内容的 MD5 哈希值判断是否重复
        3. 如果与已添加的图片（包括正样本）重复，跳过该图片，继续下一个
        4. 确保最终得到 top-k 个不重复的图片
        """
        query = self.dataset[query_idx]["query"]
        
        # 检索 Top-K（排除当前 query 的索引，因为它就是 GT）
        # 检索更多候选，以便去重后仍有足够的负样本
        topk_indices, topk_scores = self.retrieve_topk(
            query,
            corpus_embeddings,
            exclude_indices=[query_idx],
            topk=self.topk * 10  # 检索更多，以便去重后仍有足够选择
        )
        
        # 获取正样本（Ground-truth 图片）
        gt_image = self._get_image(query_idx)
        
        # 初始化：正样本
        images: List[Image.Image] = [gt_image]
        labels: List[int] = [1]
        image_indices: List[int] = [query_idx]
        
        # 用于去重：存储已使用的图片哈希值（初始化为正样本的哈希）
        used_image_hashes: Set[str] = {self._compute_image_hash(gt_image)}
        
        # 需要的负样本数量
        num_negatives_needed = self.topk - 1
        
        # ========== 阶段1：按检索顺序添加负样本（去重） ==========
        for i in range(len(topk_indices)):
            if len(images) - 1 >= num_negatives_needed:  # 已添加足够的负样本
                break
            
            neg_idx = int(topk_indices[i])
            self._try_add_negative(neg_idx, images, labels, image_indices, used_image_hashes)
        
        # ========== 阶段2：如果检索结果不够，从整个数据集按顺序查找 ==========
        if len(images) - 1 < num_negatives_needed:
            # 按索引顺序遍历整个数据集
            for candidate_idx in range(len(self.dataset)):
                if len(images) - 1 >= num_negatives_needed:
                    break
                
                # 跳过正样本索引和已在检索结果中尝试过的索引
                if candidate_idx == query_idx:
                    continue
                
                self._try_add_negative(candidate_idx, images, labels, image_indices, used_image_hashes)
        
        # ========== 阶段3：如果仍然不足，使用占位符 ==========
        if len(images) < self.topk:
            print(f"\n警告: query_idx {query_idx} 无法找到足够的唯一负样本，当前: {len(images)}/{self.topk}")
            while len(images) < self.topk:
                placeholder = Image.new('RGB', (224, 224), color='white')
                images.append(placeholder)
                labels.append(0)
                image_indices.append(-1)  # 使用 -1 表示占位符
        
        # 随机打乱图片和标签（保持对应关系）
        combined = list(zip(images, labels, image_indices))
        random.shuffle(combined)
        images_shuffled, labels_shuffled, indices_shuffled = zip(*combined)
        
        # 保存图片到磁盘，记录路径
        image_paths = []
        for idx, img in enumerate(images_shuffled):
            img_filename = f"{query_idx}_{idx}.png"
            img_path = os.path.join(self.image_dir, img_filename)
            img.save(img_path, format="PNG")
            relative_path = os.path.join("images", img_filename)
            image_paths.append(relative_path)
        
        return {
            "query": query,
            "query_idx": query_idx,
            "image_paths": image_paths,
            "labels": list(labels_shuffled),
            "image_indices": list(indices_shuffled),
            "source": self.dataset[query_idx].get("source", "unknown")
        }
    
    def build_dataset(
        self,
        corpus_embeddings: np.ndarray = None,
        output_path: str = "./data/lacs_train_dataset.jsonl",
        start_idx: int = 0,
        end_idx: int = None,
    ):
        """构建完整的 LACS 训练数据集（支持分批处理）"""
        print(f"\n[4] 构建 LACS 训练数据集...")
        
        # 如果没有提供，先构建 corpus embeddings
        if corpus_embeddings is None:
            corpus_embeddings = self.build_corpus_embeddings()
        
        if end_idx is None:
            end_idx = len(self.dataset)
        
        print(f"处理样本范围: {start_idx} - {end_idx}")
        print(f"图片保存目录: {self.image_dir}")
        
        # 构建样本
        samples = []
        failed_count = 0
        
        for i in tqdm(range(start_idx, end_idx), desc="构建样本"):
            try:
                sample = self.build_sample(i, corpus_embeddings)
                samples.append(sample)
            except Exception as e:
                print(f"\n警告: 处理样本 {i} 时出错: {e}")
                failed_count += 1
                continue
        
        print(f"✓ 成功构建 {len(samples)} 个样本")
        if failed_count > 0:
            print(f"⚠ 失败 {failed_count} 个样本")
        
        # 保存为 JSONL 格式
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 如果文件已存在且 start_idx > 0，追加模式；否则创建新文件
        mode = 'a' if os.path.exists(output_path) and start_idx > 0 else 'w'
        
        with open(output_path, mode, encoding='utf-8') as f:
            for sample in samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✓ 数据集已保存到: {output_path}")
        print(f"✓ 图片保存在: {self.image_dir}")
        print(f"  (JSONL 中的路径是相对于 {self.cache_dir} 的相对路径)")
        
        return samples

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="构建 LACS 训练数据集")
    parser.add_argument("--data_dir", type=str, 
                       default="./data/VisRAG-Ret-Train-In-domain-data",
                       help="数据集目录")
    parser.add_argument("--model_path", type=str,
                       default="openbmb/VisRAG",
                       help="检索模型路径")
    parser.add_argument("--output_path", type=str,
                       default="./data/lacs_train_dataset.jsonl",
                       help="输出文件路径")
    parser.add_argument("--topk", type=int, default=5,
                       help="Top-K 检索数量")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="批处理大小")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="最大处理样本数（用于测试）")
    parser.add_argument("--corpus_embeddings_path", type=str, default=None,
                       help="预计算的 corpus embeddings 路径")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="起始样本索引（用于分批处理）")
    parser.add_argument("--end_idx", type=int, default=None,
                       help="结束样本索引（用于分批处理）")
    parser.add_argument("--cache_dir", type=str, default="./data/lacs_cache",
                       help="缓存目录（图片也会保存在此目录下的 images/ 子目录）")
    
    args = parser.parse_args()
    
    # 加载模型（参考 demo.py）
    load_model(args.model_path)
    
    # 创建构建器
    builder = LACSDatasetBuilder(
        data_dir=args.data_dir,
        topk=args.topk,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        cache_dir=args.cache_dir
    )
    
    # 加载或构建 corpus embeddings
    if args.corpus_embeddings_path and os.path.exists(args.corpus_embeddings_path):
        print(f"\n[3] 加载预计算的 Corpus Embeddings: {args.corpus_embeddings_path}")
        corpus_embeddings = np.load(args.corpus_embeddings_path)
    else:
        corpus_embeddings = None
    
    # 构建数据集
    builder.build_dataset(
        corpus_embeddings=corpus_embeddings,
        output_path=args.output_path,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )

if __name__ == "__main__":
    main()