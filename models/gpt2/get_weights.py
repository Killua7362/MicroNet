import os
import requests
import sys
script_dir = os.path.dirname(sys.path[0])
micro_net_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

##["124M", "355M", "774M", "1558M"]
def get_weights(models_dir='Weights',model_size='1558M'):
    path = os.path.join(micro_net_dir,models_dir,model_size)
    try:
        os.makedirs(path,exist_ok=False)
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
    except FileExistsError:
        print('FileExistsError: The folder already exists. Delete the folder first then run it again')


if __name__ == "__main__":
    get_weights()