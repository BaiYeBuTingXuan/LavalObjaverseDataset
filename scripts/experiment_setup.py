import os
import json
import random
import argparse
from tqdm import tqdm
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description="Generate N-to-N Image Relighting Experiment Pairs")
    
    # 路徑參數
    parser.add_argument("--test_path", type=str, default="/media/HDD1/hejun/LavalObjaverseDataset/rendered/testing", 
                        help="測試集根路徑")
    parser.add_argument("--output", type=str, default=None, 
                        help="輸出 JSON 文件名")
    
    parser.add_argument("-n", "--views", type=int, default=16, 
                        help="每個映射挑選的視角數量 (N)")
    
    # FOV 增強參數
    parser.add_argument("--fov_argument", type=int, choices=[0, 1], default=1,
                        help="是否開啟中心裁切增強 (1: 開啟, 0: 關閉)")
    
    # 隨機種子
    parser.add_argument("--seed", type=int, default=42, 
                        help="隨機種子")

    args = parser.parse_args()
    
    if args.output is None:
        args.output = f"./experimental_pair/{args.views}_to_{args.views}_mapping_pairs.json"
        
    return args

def generate_relighting_experiment(args):
    experiment_pairs = []
    skipped_objects = []
    
    if not os.path.exists(args.test_path):
        print(f"❌ 錯誤：路徑 '{args.test_path}' 不存在。")
        return

    # 排序確保對象處理順序固定
    object_ids = sorted([d for d in os.listdir(args.test_path) 
                  if os.path.isdir(os.path.join(args.test_path, d)) and d != 'temp'])
    
    print("\n" + "="*45)
    print(f"開始生成實驗配置 (1-Source-to-All-Targets)")
    print(f"  對象總數: {len(object_ids)}")
    print(f"  每組 View 數量 (N): {args.views}")
    print("="*45 + "\n")

    # 用於記錄每個 pair 的絕對索引
    pair_idx = 0

    for obj_idx, obj_id in enumerate(tqdm(object_ids, desc="Processing Objects")):
        json_file = os.path.join(args.test_path, obj_id, "info.json")
        
        if not os.path.exists(json_file):
            skipped_objects.append((obj_id, "Missing info.json"))
            continue
            
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            lighting_list = data.get("basic", {}).get("lighting", {}).get("testing", [])
            view_list = data.get("basic", {}).get("view", {}).get("testing", [])
            
            if not lighting_list or not view_list:
                skipped_objects.append((obj_id, "Empty list"))
                continue

            MAX_LIGHTING = len(lighting_list)
            MAX_VIEW = len(view_list)
            n = args.views

            # 1. Source Light 選取: 使用對象索引 obj_idx % MAX_LIGHTING
            source_light = lighting_list[obj_idx % MAX_LIGHTING]
            
            # 2. 遍歷所有的 Target Lighting
            for target_idx, target_light in enumerate(lighting_list):
                
                # --- View 輪換邏輯 ---
                # 基礎偏移 (obj_idx * n) 確保不同物件起始點不同
                # 加上 target_idx 讓同一物件的每個 pair 視角視窗逐步輪換
                start_v_idx = (obj_idx * n + target_idx) % MAX_VIEW
                
                selected_views = []
                for i in range(n):
                    v_curr = (start_v_idx + i) % MAX_VIEW
                    selected_views.append(view_list[v_curr])

                # Crop Ratio 生成 (保留原邏輯，若需每個 pair 獨立可移至此處)
                if args.fov_argument == 1:
                    crop_ratios = [round(random.uniform(0.4, 1.0), 4) for _ in range(n)]
                else:
                    crop_ratios = [1.0] * n

                pair = {
                    "object": obj_id,
                    "source_lighting": source_light,
                    "target_lighting": target_light,
                    "view": selected_views,
                    "crop_ratio": crop_ratios,
                    "idx": pair_idx  # 🔑 改為絕對索引
                }
                experiment_pairs.append(pair)
                pair_idx += 1
                
        except Exception as e:
            tqdm.write(f"❌ [錯誤] {obj_id}: {e}")
            continue

    # 保存文件
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(experiment_pairs, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 生成完畢！")
    print(f"總計 Pair 數: {len(experiment_pairs)}")
    print(f"結果已保存至: {args.output}")

if __name__ == "__main__":
    # 固定隨機種子確保可重現性
    random.seed(42)
    args = get_args()
    generate_relighting_experiment(args)