import os
from PIL import Image
    
for subdir, dirs, files in os.walk(r'E:\\GithubEDrive\\RealTimeObjectDetection\\Tensorflow\workspace\\images'):
    for filename in files:
        filepath = subdir + os.sep + filename

        if filepath.endswith(".jpg") or filepath.endswith(".png") or filepath.endswith(".PNG"):
            print(filepath)
            img = Image.open(filepath)
            width, height = img.size
            if(width != 320 or height != 320):
                img = img.resize((320, 320), Image.ANTIALIAS)
                img.save(filepath)