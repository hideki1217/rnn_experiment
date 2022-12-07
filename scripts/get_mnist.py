from pathlib import Path
import urllib.request
import gzip
import os
import shutil

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

dataset_dir = Path(__file__).parent.parent / 'data' / 'mnist'
if not dataset_dir.exists():
    dataset_dir.mkdir()

for v in key_file.values():
    file_path = dataset_dir / v
    urllib.request.urlretrieve(url_base + v, file_path)

    with gzip.open(file_path) as f:
        with open(dataset_dir / v[:-3], "wb") as wf:
            shutil.copyfileobj(f, wf)
    os.remove(file_path)
