import subprocess
import os 

BASEPATH = os.getcwd()
_PATH = os.path.join(BASEPATH, "maxim", "maxim")
DEBLUR_WEIGHT = "deblur.npz"
DENOISE_WEIGHT = "denoise.npz"
ENHANCEMENT_WEIGHT = "enhancement.npz"
# python run_eval.py --task Deblurring --ckpt_path deblur.npz --input_dir ../../img/ --output_dir ../../img/output/ --has_target=False
def deblur_infer(emit):
    install_maxim = ["pip", "install", "."]
    command = ["python3", "run_eval.py", "--task", "Deblurring",  "--ckpt_path", DEBLUR_WEIGHT, "--input_dir", "../../img", "--output_dir", "../../img/output/deblur/maxim", "--has_target=False"]
    try:
        os.chdir(_PATH)
        subprocess.run(install_maxim , cwd=_PATH)
        process = subprocess.Popen(command,stdout=subprocess.PIPE,encoding="utf-8",cwd=_PATH)
        for l in iter(process.stdout.readline, ""):
            # await emit(l)
            print("deblur" , l)
    except Exception as e:
        print("error", e)

def denoise_infer(emit):
    install_maxim = ["pip", "install", "."]
    command = ["python3", "run_eval.py", "--task", "Denoising",  "--ckpt_path", DENOISE_WEIGHT, "--input_dir", "../../img", "--output_dir", "../../img/output/denoise/maxim", "--has_target=False"]
    try:
        os.chdir(_PATH)
        subprocess.run(install_maxim , cwd=_PATH)
        process = subprocess.Popen(command,stdout=subprocess.PIPE,encoding="utf-8")
        for l in iter(process.stdout.readline, ""):
            # await emit(l)
            print("denoise" , l)
    except Exception as e:
        print("error", e)

def enhancement_infer(emit):
    install_maxim = ["pip", "install", "."]
    command = ["python3", "run_eval.py", "--task", "Enhancement",  "--ckpt_path", ENHANCEMENT_WEIGHT, "--input_dir", "../../img", "--output_dir", "../../img/output/enhancement/maxim", "--has_target=False"]
    try:
        os.chdir(_PATH)
        subprocess.run(install_maxim , cwd=_PATH)
        process = subprocess.Popen(command,stdout=subprocess.PIPE,encoding="utf-8")
        for l in iter(process.stdout.readline, ""):
            # await emit(l)
            print("enhancement",l)
    except Exception as e:
        print("error", e)

