import argparse
import os
import subprocess
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """Run a shell command and check for errors."""
    logger.info(f"Running: {description}")
    logger.debug(f"Command: {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    
    if result.returncode != 0:
        logger.error(f"Error during {description}:")
        logger.error(result.stderr)
        sys.exit(1)
    else:
        logger.info(f"Successfully {description}")
        if result.stdout.strip():
            logger.info(result.stdout)

def main():
    parser = argparse.ArgumentParser(description='Run evaluation on a remote training server')
    parser.add_argument('--host', type=str, required=True, help='Remote host (user@ip)')
    parser.add_argument('--dataset', type=str, default='ETTh2', help='Dataset to evaluate')
    parser.add_argument('--remote-dir', type=str, default='~/ts-sandbox', help='Project directory on remote')
    parser.add_argument('--key', type=str, default=None, help='Path to SSH private key (optional)')
    args = parser.parse_args()

    ssh_base = f"ssh -o StrictHostKeyChecking=no"
    if args.key:
        ssh_base += f" -i {args.key}"
    
    # 1. Sync updated scripts to remote
    # We need to send evaluate_latest.py and models/diffusion_tsf/visualize.py 
    # (visualize.py was patched locally)
    logger.info("=" * 60)
    logger.info("STEP 1: Uploading updated scripts to remote")
    logger.info("=" * 60)
    
    files_to_sync = [
        'evaluate_latest.py',
        'models/diffusion_tsf/visualize.py'
    ]
    
    for local_file in files_to_sync:
        if not os.path.exists(local_file):
            logger.error(f"Local file not found: {local_file}")
            sys.exit(1)
            
        remote_path = os.path.join(args.remote_dir, local_file)
        scp_cmd = f"scp -o StrictHostKeyChecking=no"
        if args.key:
            scp_cmd += f" -i {args.key}"
        
        scp_cmd += f" {local_file} {args.host}:{remote_path}"
        run_command(scp_cmd, f"uploading {local_file}")

    # 2. Run evaluation on remote
    logger.info("=" * 60)
    logger.info("STEP 2: Running evaluation on remote")
    logger.info("=" * 60)
    
    # Construct remote command
    # Assumes venv is in ./venv inside the project dir
    venv_python = os.path.join(args.remote_dir, 'venv', 'bin', 'python')
    script_path = os.path.join(args.remote_dir, 'evaluate_latest.py')
    
    remote_cmd = f"{venv_python} {script_path} --dataset {args.dataset}"
    
    ssh_cmd = f"{ssh_base} {args.host} 'cd {args.remote_dir} && {remote_cmd}'"
    
    # Stream output for long-running process
    process = subprocess.Popen(ssh_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Read stdout/stderr in real-time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            
    if process.returncode != 0:
        logger.error("Remote evaluation failed!")
        print(process.stderr.read())
        sys.exit(1)
        
    # 3. Sync results back
    logger.info("=" * 60)
    logger.info("STEP 3: Downloading results")
    logger.info("=" * 60)
    
    # We need to find the run name to know what to download.
    # evaluate_latest.py prints "Results synced to: .../synced_results/<RUN_NAME>"
    # Or we can just sync the whole synced_results folder.
    
    local_sync_dir = 'synced_results'
    os.makedirs(local_sync_dir, exist_ok=True)
    
    remote_results_dir = os.path.join(args.remote_dir, 'synced_results')
    
    # Rsync is better for directories
    rsync_cmd = f"rsync -avz -e '{ssh_base}' {args.host}:{remote_results_dir}/ {local_sync_dir}/"
    run_command(rsync_cmd, "downloading results")
    
    logger.info(f"Done! Results available in {local_sync_dir}")

if __name__ == '__main__':
    main()
