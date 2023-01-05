import subprocess
import time

def run_and_wait(cmd: str):
    s = time.time()
    print(f"Running: {cmd}")
    try:
        p = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
        )
        return_code = p.wait()
    except Exception as e:
        print(f"Error: {e}")
    print(f"Finished: {cmd} return {return_code} ({time.time() - s:.3f} seconds)")