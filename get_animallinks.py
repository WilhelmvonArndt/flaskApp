#Guitar classifier
# Classifer for Flying V, Stratocast, Les Paul

# Importing libraries

import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os
import numpy as np
import pathlib


current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

#%%

# Code for importing image:
    
# // Scroll to the bottom of the page to load all image results
# window.scrollTo(0, document.body.scrollHeight);

# // Wait for images to load
# setTimeout(() => {
#   // Extract image URLs from the page
#   const urls = Array.from(document.querySelectorAll('.rg_i')).map(img => img.dataset.src || img.src);

#   // Convert URLs to CSV format
#   const csvContent = "data:text/csv;charset=utf-8," + encodeURIComponent(urls.join('\n'));

#   // Create a temporary link element
#   const link = document.createElement('a');
#   link.href = csvContent;
#   link.download = 'image_urls.csv'; // Set the filename for the CSV file

#   // Append the link element to the document, click it, and remove it
#   document.body.appendChild(link);
#   link.click();
#   document.body.removeChild(link);
# }, 5000); // Adjust the timeout value as needed to ensure images are loaded

#%%

#Downloading images Strat

csv_file = 'animals.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_file, header=None, names=['links'])
#%%

#Download Strat pics

output_folder = 'stratocaster'  # Replace with the path to your desired output folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder)  


for index, row in df.iterrows():
    image_url = row['links']  # Replace 'image_url' with the column name that contains the image URLs in your CSV
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    try:
        img.save(os.path.join(output_folder, f'stratocaster_{index}.jpg'))  # Save the image with a unique filename in the output folder
        print("saved_{index}")
    except OSError as e:
        print(f"Error saving image at index {index}: {e}")

#%%

#Delete 0KB files

folder_path = output_folder

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    # Check if the file is a regular file and not a directory
    if os.path.isfile(file_path):
        # Get the size of the file in bytes
        file_size = os.path.getsize(file_path)
        # Check if the file size is 0 KB
        if file_size == 0:
            print(f'Deleting file: {file_path}')
            os.remove(file_path)  # Delete the file

#%%
#Downloading images Les Paul

csv_file = 'lespaul.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_file, header=None, names=['links'])
#%%

#Download Strat pics

output_folder = 'lespaul'  # Replace with the path to your desired output folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder)  


for index, row in df.iterrows():
    image_url = row['links']  # Replace 'image_url' with the column name that contains the image URLs in your CSV
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    try:
        img.save(os.path.join(output_folder, f'lespaul_{index}.jpg'))  # Save the image with a unique filename in the output folder
        print("saved_{index}")
    except OSError as e:
        print(f"Error saving image at index {index}: {e}")

#%%

#Delete 0KB files

folder_path = output_folder

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    # Check if the file is a regular file and not a directory
    if os.path.isfile(file_path):
        # Get the size of the file in bytes
        file_size = os.path.getsize(file_path)
        # Check if the file size is 0 KB
        if file_size == 0:
            print(f'Deleting file: {file_path}')
            os.remove(file_path)  # Delete the file
#%%
classes = ['stratocaster', 'lespaul']

#%%

run_path = '/Users/admin/Desktop/Python projects/Guitar classifier/run'

data = ImageDataLoaders.from_folder(run_path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4, backend='torch').normalize(imagenet_stats)







