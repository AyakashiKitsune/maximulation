import shutil
import subprocess
import os 

BASEPATH = os.getcwd()
_PATH = os.path.join(BASEPATH, "UEGAN")
RAW_PATH = os.path.join(_PATH,"data","fivek","test","raw")
LABEL_PATH = os.path.join(_PATH,"data","fivek","test","label")
UEGAN_OUTPUT_DIR = os.path.join(_PATH, "results","UEGAN-FiveK","test","test_results")


def enhancement_infer(emit):
    command = ["python","main.py", "--mode", "test", "--version", "UEGAN-FiveK", "--pretrained_model", "92", "--is_test_nima", "True", "--is_test_psnr_ssim", "True"]
    try:
        os.chdir(_PATH)
        shutil.rmtree(RAW_PATH)
        shutil.rmtree(LABEL_PATH)
        os.mkdir(RAW_PATH)
        os.mkdir(LABEL_PATH)
        shutil.copyfile(os.path.join(BASEPATH, "img", "input", "input.png"),RAW_PATH)
        shutil.copyfile(os.path.join(BASEPATH, "img", "original", "input.png"),LABEL_PATH)
        process = subprocess.Popen(command,stdout=subprocess.PIPE,encoding="utf-8",cwd=_PATH)
        for l in iter(process.stdout.readline, ""):
            # await emit(l)
            print("enhancement" , l)
        output = os.listdir(UEGAN_OUTPUT_DIR)[0]
        shutil.copyfile(os.path.join(UEGAN_OUTPUT_DIR, output),os.path.join(BASEPATH, "img","output", "enhancement","uegan",output))
        os.rename(os.path.join(BASEPATH, "img","output", "enhancement","uegan",output),os.path.join(BASEPATH, "img","output", "enhancement","uegan","input.png"))
        
    except Exception as e:
        print("error", e)
