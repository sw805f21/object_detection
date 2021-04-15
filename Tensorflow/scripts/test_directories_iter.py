import os
import re

checkpoints = []
for subdir, dirs, files in os.walk(CHECKPOINT_PATH):
    for file_name in files:
        if(file_name.endswith("index")):
            checkpoints.append(file_name[:-6])


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

checkpoints.sort(key=natural_keys)
print(checkpoints)
print(checkpoints[-1])