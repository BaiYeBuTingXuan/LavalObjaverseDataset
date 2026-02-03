import json
import multiprocessing
import subprocess
import time
import os
import random
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional
import tyro
import wandb
import shlex
from utils import get_subset_json_files

# ===== LOGGING SETUP (DO THIS FIRST) =====
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"render_{TIMESTAMP}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(process)d | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging initialized. Full log: {LOG_FILE}")

OBJAVERSE_INFO = 'objaverse/info'
SAVE_ROOT = os.path.abspath('rendered')

@dataclass
class Args:
    workers_per_gpu: int = 1
    """number of workers per gpu"""

    split: str = 'training'

    blender: str = '/home/user/Documents/application/blender-4.3.2-linux-x64/blender'
    """Blender executable path"""

    log_to_wandb: bool = True
    """Whether to log progress to wandb"""

    gpus: list = field(default_factory=lambda: [0])
    """GPU indices to use"""

    skip_exist: bool = True
    """Skip existing renders (checks for finish.keep marker)"""

    seed: int = 0
    """Random seed"""

    upload: bool = False
    """Upload rendered subsets to remote server after completion"""

    upload_retries: int = 3
    """Number of retry attempts for failed uploads"""

    remote_ip: Optional[str] = None
    """Remote server IP (loaded from env/credentials file if not provided)"""

    remote_pwd: Optional[str] = None
    """Remote server password (loaded from env/credentials file if not provided)"""

    remote_path: Optional[str] = None
    """Remote base path (loaded from env/credentials file if not provided)"""

    credentials_file: Optional[str] = None
    """Path to credentials JSON file (default: ./credentials.json). NOT tracked by git."""

def upload_and_cleanup(
    local_path: str,
    remote_ip: str,
    remote_pwd: str,
    remote_base_path: str,
    save_root: str,  # ADDED: Base rendering directory for relative path extraction
    max_retries: int = 3
) -> bool:
    """
    Upload directory to remote server using PASSWORD authentication.
    Extracts subset name as relative path from save_root (e.g., 'training/subset_1').
    
    Returns:
        True on success, False if all retries fail.
    """
    if not os.path.exists(local_path):
        logger.warning(f"Upload skipped - path doesn't exist: {local_path}")
        return False
    
    # Extract RELATIVE path from SAVE_ROOT (e.g., 'training/subset_1')
    try:
        subset_relpath = os.path.relpath(local_path, save_root)
        if subset_relpath.startswith('..') or subset_relpath == '.':
            raise ValueError(f"local_path '{local_path}' is not under save_root '{save_root}'")
    except Exception as e:
        logger.error(f"Failed to compute relative path: {e}")
        subset_relpath = os.path.basename(local_path)  # Fallback
    
    # Construct remote path preserving hierarchy
    remote_full_path = os.path.join(remote_base_path, subset_relpath)
    remote_full_path_quoted = shlex.quote(remote_full_path)
    
    for attempt in range(1, max_retries + 1):
        logger.info(f"Upload attempt {attempt}/{max_retries} for {subset_relpath} -> ubuntu@{remote_ip}:{remote_full_path}")
        
        try:
            # Set up environment with password via SSHPASS (secure)
            env = os.environ.copy()
            env['SSHPASS'] = remote_pwd
            
            # === STEP 1: Create remote directory ===
            mkdir_cmd = [
                'sshpass', '-e', 'ssh',
                '-o', 'StrictHostKeyChecking=no',
                '-o', 'ConnectTimeout=10',
                '-o', 'ServerAliveInterval=30',
                f'ubuntu@{remote_ip}',
                f'mkdir -p {remote_full_path_quoted}'
            ]
            
            subprocess.run(
                mkdir_cmd,
                env=env,
                check=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # === STEP 2: Rsync data ===
            rsync_cmd = [
                'sshpass', '-e', 'rsync',
                '-avz',
                '--progress',
                '--exclude=finish.keep',
                '--rsh=ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=30',
                f'{local_path}/',
                f'ubuntu@{remote_ip}:{remote_full_path_quoted}/'
            ]
            
            process = subprocess.Popen(
                rsync_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream rsync output (filter noise)
            for line in process.stdout:
                stripped = line.strip()
                if stripped and not any(noise in stripped for noise in [
                    'sending incremental file list', 'total:', 'speedup is', 'bytes/sec'
                ]):
                    logger.debug(f"RSYNC: {stripped}")
            
            ret = process.wait(timeout=3600)
            if ret != 0:
                raise RuntimeError(f"RSYNC failed with exit code {ret}")
            
            # === SUCCESS: Perform cleanup ===
            logger.info(f"✓ Upload succeeded for {subset_relpath} on attempt {attempt}")
            
            # Remove local files (keep directory structure + finish.keep)
            removed_count = 0
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    if file == "finish.keep":
                        continue
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        removed_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to remove {file_path}: {e}")
            
            # Create finish.keep marker with full context
            marker_path = os.path.join(local_path, "finish.keep")
            with open(marker_path, 'w') as f:
                f.write(f"Upload completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Local path: {local_path}\n")
                f.write(f"Relative path: {subset_relpath}\n")
                f.write(f"Remote destination: ubuntu@{remote_ip}:{remote_full_path}\n")
                f.write(f"Files removed: {removed_count}\n")
                f.write(f"Upload attempts: {attempt}/{max_retries}\n")
            
            logger.info(f"✓ Cleanup complete. Created marker: {marker_path} ({removed_count} files removed)")
            return True
            
        except subprocess.TimeoutExpired as e:
            error_msg = f"Timeout after {e.timeout}s"
        except subprocess.CalledProcessError as e:
            error_msg = f"Command failed (exit {e.returncode}): {e.stderr[:300] if e.stderr else 'no stderr'}"
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)[:300]}"
        
        logger.warning(f"Upload attempt {attempt} failed for {subset_relpath}: {error_msg}")
        
        if attempt < max_retries:
            backoff = min(2 ** (attempt - 1), 30)  # Exponential backoff capped at 30s
            logger.info(f"Retrying in {backoff}s...")
            time.sleep(backoff)
        else:
            logger.error(f"✗ All {max_retries} upload attempts failed for {subset_relpath}. SKIPPING cleanup.")
            return False

def upload_and_cleanup(
    local_path: str,
    remote_ip: str,
    remote_auth: dict,  # {'method': 'password', 'value': 'pwd'} OR {'method': 'key', 'value': '/path/to/key'}
    remote_base_path: str,
    remote_user: str = "ubuntu",
    max_retries: int = 3
) -> bool:
    """
    Upload directory to remote server with retry logic.
    Supports BOTH password AND SSH key authentication securely.
    
    Args:
        remote_auth: Dict with keys:
            - method: 'password' or 'key'
            - value: password string OR path to private key file
    
    Returns:
        True on success, False if all retries fail.
    """
    if not os.path.exists(local_path):
        logger.warning(f"Upload skipped - path doesn't exist: {local_path}")
        return False
    
    subset_name = os.path.basename(local_path)
    remote_full_path = os.path.join(remote_base_path, subset_name)
    
    # Pre-validate auth method
    if remote_auth['method'] not in ('password', 'key'):
        raise ValueError(f"Invalid auth method: {remote_auth['method']}. Must be 'password' or 'key'")
    
    if remote_auth['method'] == 'key' and not os.path.exists(remote_auth['value']):
        raise FileNotFoundError(f"SSH key not found: {remote_auth['value']}")

    for attempt in range(1, max_retries + 1):
        logger.info(f"Upload attempt {attempt}/{max_retries} for {subset_name} -> {remote_user}@{remote_ip}:{remote_full_path}")
        
        try:
            # === STEP 1: Create remote directory ===
            if remote_auth['method'] == 'password':
                # SAFER: Use SSHPASS env var instead of -p flag (avoids ps leakage)
                mkdir_cmd = [
                    'sshpass', '-e', 'ssh',
                    '-o', 'StrictHostKeyChecking=no',
                    '-o', 'ConnectTimeout=10',
                    f'{remote_user}@{remote_ip}',
                    f'mkdir -p {shlex.quote(remote_full_path)}'
                ]
                env = os.environ.copy()
                env['SSHPASS'] = remote_auth['value']
            else:  # key auth
                mkdir_cmd = [
                    'ssh',
                    '-i', remote_auth['value'],
                    '-o', 'StrictHostKeyChecking=no',
                    '-o', 'ConnectTimeout=10',
                    f'{remote_user}@{remote_ip}',
                    f'mkdir -p {shlex.quote(remote_full_path)}'
                ]
                env = None
            
            subprocess.run(
                mkdir_cmd,
                env=env,
                check=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # === STEP 2: Rsync data ===
            if remote_auth['method'] == 'password':
                rsync_cmd = [
                    'sshpass', '-e', 'rsync',
                    '-avz',
                    '--progress',
                    '--exclude=finish.keep',
                    '--rsh=ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10',
                    f'{local_path}/',
                    f'{remote_user}@{remote_ip}:{remote_full_path}/'
                ]
                env = os.environ.copy()
                env['SSHPASS'] = remote_auth['value']
            else:  # key auth
                rsync_cmd = [
                    'rsync',
                    '-avz',
                    '--progress',
                    '--exclude=finish.keep',
                    '-e', f'ssh -i {shlex.quote(remote_auth["value"])} -o StrictHostKeyChecking=no -o ConnectTimeout=10',
                    f'{local_path}/',
                    f'{remote_user}@{remote_ip}:{remote_full_path}/'
                ]
                env = None
            
            # Stream rsync output with minimal noise
            process = subprocess.Popen(
                rsync_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in process.stdout:
                stripped = line.strip()
                if stripped and not any(noise in stripped for noise in [
                    'sending incremental file list', 'total:', 'speedup is', 'bytes/sec', '^M'
                ]):
                    logger.debug(f"RSYNC: {stripped}")
            
            ret = process.wait(timeout=3600)
            if ret != 0:
                raise RuntimeError(f"RSYNC failed with exit code {ret}")
            
            # === SUCCESS: Perform cleanup ===
            logger.info(f"✓ Upload succeeded for {subset_name} on attempt {attempt}")
            
            # Remove local files (keep directory structure)
            removed_count = 0
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    if file == "finish.keep":
                        continue
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        removed_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to remove {file_path}: {e}")
            
            # Create finish.keep marker
            marker_path = os.path.join(local_path, "finish.keep")
            with open(marker_path, 'w') as f:
                f.write(f"Upload completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Source: {local_path}\n")
                f.write(f"Destination: {remote_user}@{remote_ip}:{remote_full_path}\n")
                f.write(f"Auth method: {remote_auth['method']}\n")
                f.write(f"Files removed: {removed_count}\n")
                f.write(f"Upload attempts: {attempt}/{max_retries}\n")
            
            logger.info(f"✓ Cleanup complete. Created marker: {marker_path} ({removed_count} files removed)")
            return True
            
        except subprocess.TimeoutExpired as e:
            error_msg = f"Timeout after {e.timeout}s"
        except subprocess.CalledProcessError as e:
            error_msg = f"Command failed (exit {e.returncode}): {e.stderr[:300] if e.stderr else 'no stderr'}"
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)[:300]}"
        
        logger.warning(f"Upload attempt {attempt} failed for {subset_name}: {error_msg}")
        
        if attempt < max_retries:
            backoff = min(2 ** (attempt - 1), 30)  # Exponential backoff capped at 30s
            logger.info(f"Retrying in {backoff}s...")
            time.sleep(backoff)
        else:
            logger.error(f"✗ All {max_retries} upload attempts failed for {subset_name}. SKIPPING cleanup.")
            return False

def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
    args,
) -> None:
    worker_logger = logging.getLogger(f"worker.gpu{gpu}.{os.getpid()}")
    
    while True:
        item_save = queue.get()
        if item_save is None:
            queue.task_done()
            break

        item, save_path = item_save
        
        # Individual object skip check (info.json)
        if args.skip_exist and os.path.exists(os.path.join(save_path, item, 'info.json')):
            worker_logger.info(f"✓ Skipping {item} (info.json exists)")
            queue.task_done()
            with count.get_lock():
                count.value += 1
            continue

        worker_logger.info(f".Rendering {item} on GPU {gpu}")
        
        command = (
            f"CUDA_VISIBLE_DEVICES={gpu} "
            f"{args.blender} -b -P ./blender/blender_script.py -- "
            f"--object_name {item} "
            f"--output_dir '{save_path}' "
            f"--depth "
            f"--lighting_split {args.split} "
            f"--view_split {args.split} "
        )
        if args.skip_exist:
            command += " --skip_exist"

        start_time = time.time()
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=1800  # 30 min timeout
            )
            duration = time.time() - start_time
            worker_logger.info(f"✓ Rendered {item} in {duration:.1f}s")
            
            # Log non-frame Blender output
            for line in result.stdout.splitlines():
                if line.strip() and not line.startswith("Fra:"):
                    worker_logger.debug(f"[{item}] {line}")
                        
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            worker_logger.error(f"✗ TIMEOUT {item} after {duration:.1f}s")
        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            worker_logger.error(f"✗ FAILED {item} after {duration:.1f}s (exit {e.returncode})")
        except Exception as e:
            worker_logger.exception(f"✗ CRITICAL ERROR rendering {item}")

        with count.get_lock():
            count.value += 1
        queue.task_done()

def rendering(uids, save_path, queue, count, args, subset_idx=0, total_subsets=1, total_objects=0):
    """
    Queue all objects for a subset and wait for completion.
    Returns when all objects in this subset are rendered.
    """
    if args.seed != 0:
        random.seed(args.seed + subset_idx)  # Deterministic per-subset shuffling
        random.shuffle(uids)

    logger.info(f"Queueing {len(uids)} objects for {save_path} (subset {subset_idx+1}/{total_subsets})")
    start_count = count.value
    
    for item in uids:
        queue.put((item, save_path))
    
    # Wait for subset completion with progress reporting
    target_count = start_count + len(uids)
    last_report = start_count
    
    while count.value < target_count:
        current = count.value
        if current > last_report or (current == target_count):
            rendered_this = current - start_count
            progress = rendered_this / len(uids)
            global_progress = current / total_objects if total_objects else 0
            
            logger.info(
                f"Subset {subset_idx+1}/{total_subsets}: {rendered_this}/{len(uids)} ({progress:.1%}) | "
                f"Global: {current}/{total_objects} ({global_progress:.1%})"
            )
            
            if args.log_to_wandb:
                wandb.log({
                    "rendered_count": current,
                    "total_count": total_objects,
                    "subset_progress": progress,
                    "subset_index": subset_idx,
                    "global_progress": global_progress,
                    "current_subset": os.path.basename(save_path),
                })
            
            last_report = current
        
        time.sleep(30)  # Report every 30 seconds
    
    logger.info(f"✓ Subset completed: {len(uids)} objects rendered at {save_path}")
    return len(uids)

def main():
    args = tyro.cli(Args)
    assert args.split in ['training', 'validation', 'test'], f"Invalid split: {args.split}"
    
    # Load credentials ONLY if upload is requested
    if args.upload:
        args = load_remote_credentials(args)
        logger.warning("⚠️  Using password-based SSH authentication. For production, use SSH keys.")
    
    logger.info("="*70)
    logger.info(f"STARTING RENDERING SESSION | Split: {args.split}")
    logger.info(f"Blender: {args.blender}")
    logger.info(f"Output root: {SAVE_ROOT}")
    logger.info(f"GPUs: {args.gpus} | Workers/GPU: {args.workers_per_gpu}")
    logger.info(f"Skip existing: {args.skip_exist} | Upload: {args.upload}")
    if args.upload:
        logger.info(f"Upload retries: {args.upload_retries} | Remote target: {args.remote_ip}:{args.remote_path}")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("="*70)

    if args.log_to_wandb:
        wandb.login(key='f99bf05243fcbfa31ac750bdbc1675282b080eae')
        wandb.init(project="Laval-Objaverse-Dataset-rendering", config=vars(args))
    
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    # Start persistent worker pool
    processes = []
    for gpu_i in args.gpus:
        for _ in range(args.workers_per_gpu):
            p = multiprocessing.Process(
                target=worker,
                args=(queue, count, gpu_i, args)
            )
            p.start()
            processes.append(p)
            logger.info(f"Started worker on GPU {gpu_i} (PID: {p.pid})")

    try:
        # ===== PREPARE WORKLOAD =====
        if args.split == 'training':
            subsets_dir = f"{OBJAVERSE_INFO}/training_subsets"
            subsets = get_subset_json_files(subsets_dir)
            total_objects = 0
            subset_data = []
            
            for subset in subsets:
                with open(os.path.join(subsets_dir, subset), 'r') as f:
                    uids = json.load(f)
                    subset_name = os.path.splitext(subset)[0]
                    save_path = os.path.join(SAVE_ROOT, args.split, subset_name)
                    subset_data.append((subset_name, save_path, uids))
                    total_objects += len(uids)
                    
            logger.info(f"Found {len(subsets)} training subsets ({total_objects} total objects)")
        else:
            with open(f'{OBJAVERSE_INFO}/full_{args.split}_objects', 'r') as f:
                uids = json.load(f)
            save_path = os.path.join(SAVE_ROOT, args.split)
            subset_data = [(args.split, save_path, uids)]
            total_objects = len(uids)
            logger.info(f"Processing {args.split} split ({total_objects} objects)")

        # ===== PROCESS EACH SUBSET SEQUENTIALLY =====
        failed_uploads = []
        successful_uploads = []
        
        for subset_idx, (subset_name, save_path, uids) in enumerate(subset_data):
            os.makedirs(save_path, exist_ok=True)
            marker_path = os.path.join(save_path, "finish.keep")
            
            # SKIP CHECK: finish.keep at subset level
            if args.skip_exist and os.path.exists(marker_path):
                logger.info(f"✓ Skipping {subset_name} (finish.keep exists at {marker_path})")
                continue
            
            logger.info(f"\n{'='*70}")
            logger.info(f"PROCESSING SUBSET {subset_idx+1}/{len(subset_data)}: {subset_name}")
            logger.info(f"Objects: {len(uids)} | Path: {save_path}")
            logger.info('='*70)
            
            # Render the subset
            rendered_count = rendering(
                uids=uids,
                save_path=save_path,
                queue=queue,
                count=count,
                args=args,
                subset_idx=subset_idx,
                total_subsets=len(subset_data),
                total_objects=total_objects
            )
            
            # UPLOAD & CLEANUP (if enabled)
            if args.upload:
                success = upload_and_cleanup(
                    save_path,
                    args.remote_ip,
                    args.remote_pwd,
                    args.remote_path,
                    max_retries=args.upload_retries
                )
                
                if success:
                    successful_uploads.append(subset_name)
                    if args.log_to_wandb:
                        wandb.log({
                            f"upload_success_{subset_name}": 1,
                            "last_uploaded_subset": subset_name,
                            "upload_timestamp": time.time()
                        })
                else:
                    failed_uploads.append(subset_name)
                    logger.error(f"✗ Upload permanently failed for {subset_name} after {args.upload_retries} attempts. Continuing to next subset...")
                    if args.log_to_wandb:
                        wandb.alert(
                            title="Upload Failed",
                            text=f"Subset {subset_name} failed after {args.upload_retries} attempts. Continuing rendering..."
                        )
                    # CRITICAL: Continue to next subset - don't crash entire job
                    continue

        # ===== FINAL SYNCHRONIZATION =====
        logger.info("Waiting for all workers to finish processing queue...")
        queue.join()
        
        # Terminate workers gracefully
        logger.info("Terminating workers...")
        for _ in range(len(args.gpus) * args.workers_per_gpu):
            queue.put(None)
        
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                logger.warning(f"Worker {p.pid} hung - forcing termination")
                p.terminate()
                p.join(5)

        # Final summary
        logger.info("\n" + "="*70)
        logger.info("UPLOAD SUMMARY")
        logger.info(f"Successful uploads: {len(successful_uploads)}")
        if successful_uploads:
            logger.info(f"  {', '.join(successful_uploads[:5])}{'...' if len(successful_uploads) > 5 else ''}")
        logger.info(f"Failed uploads: {len(failed_uploads)}")
        if failed_uploads:
            logger.info(f"  {', '.join(failed_uploads[:5])}{'...' if len(failed_uploads) > 5 else ''}")
        logger.info("="*70)

        if args.log_to_wandb:
            wandb.finish()

    except KeyboardInterrupt:
        logger.error("KeyboardInterrupt received - terminating workers...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        raise
    finally:
        logger.info("\n" + "="*70)
        logger.info(f"SESSION COMPLETE | Total rendered: {count.value}")
        logger.info(f"Log file: {LOG_FILE}")
        logger.info("="*70)

if __name__ == "__main__":
    main()