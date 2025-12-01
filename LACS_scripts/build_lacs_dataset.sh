# LACS_scripts/run_build_test.sh
#!/bin/bash
# 测试模式：只处理前100个样本，用于验证流程

python LACS_scripts/build_lacs_dataset.py \
    --data_dir ./data/VisRAG-Ret-Train-In-domain-data \
    --model_path openbmb/VisRAG-Ret \
    --output_path ./data/lacs_train_dataset_test.jsonl \
    --topk 5 \
    --batch_size 16 \
    --max_samples 100 \
    --cache_dir ./data/lacs_cache_test