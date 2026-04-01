import os
import json

def compare_folders_with_json(json_path, target_dir):
    # 1. 讀取 JSON 檔案
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 假設 JSON 結構是 {"folder_names": ["name1", "name2", ...]}
            # 如果你的 JSON 只是單純的列表格式 [ "name1", ... ]，請修改下方這行
            json_folders = data
    except FileNotFoundError:
        print(f"錯誤：找不到 JSON 檔案 {json_path}")
        return
    except Exception as e:
        print(f"讀取 JSON 時發生錯誤: {e}")
        return

    # 2. 讀取路徑下的子文件夾名字
    if not os.path.exists(target_dir):
        print(f"錯誤：路徑 {target_dir} 不存在")
        return

    # 只列出目錄中的子文件夾 (排除檔案)
    actual_folders = set([
        name for name in os.listdir(target_dir) 
        if os.path.isdir(os.path.join(target_dir, name))
    ])

    # 3. 比對差異
    # 在子文件夾中，但不在 JSON 中的 (多出來的)
    not_in_json = actual_folders - set(json_folders)
    
    # 在 JSON 中，但不在子文件夾中的 (缺失的)
    missing_in_disk = set(json_folders) - actual_folders

    # 4. 輸出結果
    print(f"--- 統計結果 ---")
    print(f"JSON 記錄數量: {len(json_folders)}")
    print(f"實際文件夾數量: {len(actual_folders)}")
    print("\n--------------------------------")

    if not_in_json:
        print(f"⚠️  不在 JSON 記錄中，但存在於硬碟的文件夾 ({len(not_in_json)} 個):")
        for folder in sorted(list(not_in_json)):
            print(f"  - {folder}")
    else:
        print("✅ 所有硬碟文件夾皆已記錄在 JSON 中。")

    print("--------------------------------")

    if missing_in_disk:
        print(f"❌ JSON 中原本應有，但硬碟缺失的文件夾 ({len(missing_in_disk)} 個):")
        for folder in sorted(list(missing_in_disk)):
            print(f"  - {folder}")
    else:
        print("✅ JSON 中的記錄在硬碟皆有對應文件夾。")

# --- 設定區 ---
if __name__ == "__main__":
    # 請根據你的實際路徑修改這裡
    MY_JSON_FILE = "/media/HDD1/hejun/LavalObjaverseDataset/objaverse/info/full_validation_objects.json"  # JSON 路徑
    MY_TARGET_DIR = "/media/HDD1/hejun/LavalObjaverseDataset/rendered/validation"  # 要掃描的目錄路徑

    compare_folders_with_json(MY_JSON_FILE, MY_TARGET_DIR)