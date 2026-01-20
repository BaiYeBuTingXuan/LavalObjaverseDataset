import glob
import json
import multiprocessing
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional
import os
import random
import boto3
import tyro
import wandb
import signal
from multiprocessing import Process, Queue
import objaverse
import time

@dataclass
class Args:
    workers_per_gpu: int = 8
    """number of workers per gpu"""
    
    input_models_path: str = '../../dataset/Objaverse/objects.json'
    """Path to a json file containing a list of 3D object files"""

    output_dir: str = '/media/HDD1/hejun/dataset/relitObjaverse'
    """Path to save the rendered result"""

    lighting_dir: str = '/media/HDD1/hejun/dataset/resized_lighting'
    """Path to save the rendered result"""


    blender: str = '/media/HDD1/hejun/blender-4.3.2-linux-x64/blender'
    """Blender to launch"""

    upload_to_s3: bool = False
    """Whether to upload the rendered images to S3"""

    log_to_wandb: bool = True
    """Whether to log the progress to wandb"""

    gpus: list = field(default_factory=lambda: [0])
    """number of gpus to use."""

    skip_exist: bool = True
    """If we can skip existing result."""

    seed: int = 0
    "Random seed"

def list_subfolders(directory):
    """
    List all subfolder names in the specified directory.

    Parameters:
    - directory: The path to the directory to search.

    Returns:
    - A list of subfolder names.
    """
    # List to store subfolder names
    subfolders = []

    # Iterate over the entries in the directory
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        # Check if the entry is a directory
        if os.path.isdir(full_path):
            subfolders.append(entry)

    return subfolders

def write_to_file(file_name, content):
    with open(file_name, 'w') as file:  # 'w' mode opens the file for writing (overwrites existing content)
        file.write(content)
        print(f"Content written to {file_name}")

def append_to_file(file_name, content):
    with open(file_name, 'a') as file:  # 'a' mode opens the file for appending
        file.write(content + '\n')  # Adding a newline for better formatting
        print(f"Content appended to {file_name}")

def print_lines(lines):
    for line in lines.splitlines():
        if not line.startswith("Fra"):
            print(line)

def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
    BLENDER,
    OUT_DIR,
    SKIP_EXIST,
    LIGHTING
    # s3: Optional[boto3.client],
) -> None:
    while True:
        item = queue.get()
        if item is None:
            break

        if SKIP_EXIST and os.path.exists(os.path.join(OUT_DIR, 'objects', item, 'info.json')):
            print('========', item, 'rendered', '========')
            queue.task_done()
            with count.get_lock():
                count.value += 1
            continue
            # Perform some operation on the item
        print('========', 'rendering', item, "now", '========')

        command = (
            f" CUDA_VISIBLE_DEVICES={gpu} "
            f" {BLENDER} -b -P blender_script.py --"
            f" --object_name {item}"
            f" --output_dir {OUT_DIR}"
            f" --depth"
            # f" --normal"
            # f" --albedo"
            f" --lighting_dir {LIGHTING}"
        )

        if SKIP_EXIST and count.value == 0:
            # command += f" --env"
            pass

        if SKIP_EXIST:
            command += f" --skip_exist"

        start_time = time.time()
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        end_time = time.time()
        duration = end_time - start_time
        print(f'==Render item {item} using {duration:.2f} seconds==')
        print(f'Item {item} Standard Output:')
        print_lines(result.stdout.decode())

        print(f'Item {item} Error Report:')
        print_lines(result.stderr.decode())


        with count.get_lock():
            count.value += 1
        queue.task_done()
        
if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.log_to_wandb:
        wandb.login(key='f99bf05243fcbfa31ac750bdbc1675282b080eae')
        wandb.init(project="objaverse-rendering")
    s3 = boto3.client("s3") if args.upload_to_s3 else None
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    BLENDER = args.blender
    OUT_DIR = os.path.abspath(args.output_dir)
    SKIP_EXIST = args.skip_exist
    LIGHTING = args.lighting_dir

    print(f'We use blender:{BLENDER}')
    print(f'Output:{OUT_DIR}')
    print(f'Lighting files come from {LIGHTING}')
    if SKIP_EXIST:
        print('These processes will skip existing files')
    else:
        print('These processes will NOT skip existing files')

    processes = []
    # Start worker processes on each of the GPUs
    for gpu_i in args.gpus:
        for worker_i in range(args.workers_per_gpu):
            global_args = (BLENDER, OUT_DIR, SKIP_EXIST, LIGHTING)
            worker_i = gpu_i * args.workers_per_gpu + worker_i
            process = multiprocessing.Process(
                target=worker, args=(queue, count, gpu_i, *global_args)
            )
            process.daemon = True
            process.start()
            processes.append(process)

    try:
        # # Add items to the queue
        # with open(args.input_models_path, "r") as f:
        #     model_paths = json.load(f)
        # # import ipdb; ipdb.set_trace()
        # model_keys = list(model_paths.keys())
        
        # model_keys = model_keys
        # # random.shuffle(model_keys)
        # model_keys.sort()

        with open('./subset3.json', 'r') as f:
            info = json.load(f)
            if isinstance(info, dict):
                uids = list(info.keys())
            else:
                uids = list(info)
        # uids = list_subfolders(os.path.join(args.output_dir, "objects"))

        with open('./rendered.json', 'r') as f:
            rendered_uids = json.load(f)
        
        uids = [uid for uid in uids if uid not in rendered_uids]

        num = len(uids)
        if args.seed != 0:
            random.seed(args.seed)
            random.shuffle(uids)     # 打乱列表

        print(f"We will download {num} objects")

        for item in uids:
            queue.put(item)
            # queue.put(os.path.join('/share/phoenix/nfs05/S8/hj453/Objaverse/hf-objaverse-v1', model_paths[item]))

        # update the wandb count
        if args.log_to_wandb:
            while True:
                wandb.log(
                    {
                        "count": count.value,
                        # "total": len(model_paths),
                        "progress": count.value / len(uids),
                    }
                )
                if count.value == len(uids):
                    break
                time.sleep(60)
        wandb.finish()

        queue.join()
        # Add sentinels to the queue to stop the worker processes
        for i in range(args.num_gpus * args.workers_per_gpu):
            queue.put(None)
    except KeyboardInterrupt:
        print("Received keyboard interrupt. Terminating processes.")
        for p in processes:
            os.kill(p.pid, signal.SIGKILL)