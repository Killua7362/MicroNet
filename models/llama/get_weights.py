import os
import requests
import sys
models_dir = 'Weights'
model_size = '1558M'
path = os.path.join(models_dir,model_size)
try:
    os.makedirs(path,exist_ok=False)
except FileExistsError:
    print('FileExistsError: The folder already exists. Delete the folder first then run it again')
    sys.exit(0)

for filename in [
       "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
]:
        url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
        r = requests.get(f"{url}/{model_size}/{filename}", stream=True)
        r.raise_for_status()
    
        with open(os.path.join(path,filename),'wb') as f:
            f.write(r.content)