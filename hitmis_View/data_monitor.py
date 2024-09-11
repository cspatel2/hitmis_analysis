#%% watches the folder thats collecting data. As soon as it collects a full day of data,
# it will run the data conversion so we can L1A data as data collection is happening.
#%%
from __future__ import annotations 
from collections.abc import Iterable
import os
import time
import subprocess
from datetime import datetime
import numpy as np
from tqdm import tqdm

#%%
def get_fn(dir:str)-> Iterable:
    dirlist = os.listdir(dir)
    folders = np.sort([f for f in dirlist if f.isnumeric()])
    # folder_paths = [os.path.join(dir,f) for f in folders]
    return folders

def run_conversion_script(script_path:str, data_folder:str, output_directory:str, prefix:str):
    command = f"python {script_path} {data_folder} {output_directory} --prefix {prefix}"
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(f"Output:\n{result.stdout}")
    print(f"Errors:\n{result.stderr}")

def timer(seconds : int):
    for _ in tqdm(range(seconds), desc="Next check in", unit="s"):
        time.sleep(1)

def watch(watch_directory:str, script_path:str, output_directory:str, prefix:str, polling_interval:str):
    previous_day_folder = None

    while True:
        folders = get_fn(watch_directory)
        latest_folder = folders[-1]

        if previous_day_folder is not None and latest_folder != previous_day_folder:
            previous_day_folder_path = os.path.join(watch_directory, previous_day_folder)
            run_conversion_script(script_path, previous_day_folder_path, output_directory, prefix)

        previous_day_folder = latest_folder

        timer(polling_interval)

if __name__ == "__main__":
    watch_directory = "/media/windowsshare"
    script_path = "/home/charmi/Projects/hitmis_analysis/hitmis_pipeline/hitmis_l1a_converter.py"
    output_directory = "/home/charmi/Projects/hitmis_analysis/hms1_FoxHall_L1A/"
    prefix = "Check"

    polling_interval = 6 * 60 * 60  # Check every 6 hours

    watch(watch_directory, script_path, output_directory, prefix, polling_interval)

