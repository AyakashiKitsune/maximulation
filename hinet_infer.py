import subprocess
import os 

BASEPATH = os.getcwd()
_PATH = os.path.join(BASEPATH, "HINet")
def deblur_infer(emit):
    command = ['python', 'basicsr/demo.py','-opt', 'options/demo/demo.yml']
    try:
        os.chdir(_PATH)
        process = subprocess.Popen(command,stdout=subprocess.PIPE,encoding="utf-8")
        for l in iter(process.stdout.readline, ""):
            # await emit(l)
            print("deblur" , l)
    except Exception as e:
        print("error", e)
