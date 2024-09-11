#%% find fit file that matches the jpg and move then into same folder
import os
import shutil
import re

# Define source and destination directories
png_dir = "/home/charmi/Projects/hitmis_analysis/aurora_052024/20240510/day/"
fits_dir = "/home/charmi/Projects/hitmis_analysis/aurora_052024/20240510/"

# Regular expression pattern to match timestamp in PNG file names
pattern = re.compile(r'(\d+)\.jpg')

# Iterate over PNG files
for png_file in os.listdir(png_dir):
    if png_file.endswith(".jpg"):
        # Extract timestamp from PNG file name using regular expression
        match = pattern.search(png_file)
        if match:
            timestamp = match.group(1)

            # Find matching FITS file
            for fits_file in os.listdir(fits_dir):
                if timestamp in fits_file:
                    # Move FITS file to the same directory as PNG file
                    fits_path = os.path.join(fits_dir, fits_file)
                    destination = os.path.join(png_dir, fits_file)
                    shutil.move(fits_path, destination)
                    print("Moved {} to {}".format(fits_file, png_dir))
                    break
            else:
                print("No matching FITS file found for {}".format(png_file))
        else:
            print("Timestamp not found in PNG file name: {}".format(png_file))
# %%
