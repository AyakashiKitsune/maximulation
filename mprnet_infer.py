import subprocess
import os 

BASEPATH = os.getcwd()
_PATH = os.path.join(BASEPATH, "MPRNet")

def denoise_infer(emit):
    command = ["python", "demo.py", "--task", "Denoising", "--input_dir", os.path.join(BASEPATH, "img", "input"), "--result_dir", os.path.join(BASEPATH, "img", "output", "denoise", "mprnet")]
    try:
        os.chdir(_PATH)
        process = subprocess.Popen(command,stdout=subprocess.PIPE,encoding="utf-8",cwd=_PATH)
        for l in iter(process.stdout.readline, ""):
            # await emit(l)
            print("denoise" , l)
        
    except Exception as e:
        print("error", e)