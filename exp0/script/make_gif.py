from pathlib import Path
import re

from PIL import Image
import parse

while (folder := Path(input("image directory(path/empty): "))).exists():
    pictures=[]
    for path in filter(lambda x: re.match(R"[0-9]*\.png", x.name), folder.iterdir()):
        (time, )=parse.parse("{:d}.png", path.name)
        img = Image.open(path)
        pictures.append((time, img))
    pictures = list(map(lambda x: x[1], sorted(pictures)))
    pictures[0].save(folder.parent / f'{folder.name}.gif',
                    save_all=True, 
                    append_images=pictures[1:], 
                    optimize=True, 
                    duration=500, 
                    loop=0)
