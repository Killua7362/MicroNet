from zipfile import ZipFile
import os
import urllib
import urllib.request

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'
if not os.path.isfile(FILE):
    urllib.request.urlretrieve(URL, FILE)
with ZipFile(FILE) as zip_images:
    zip_images.extractall(FOLDER)
print('Done!')