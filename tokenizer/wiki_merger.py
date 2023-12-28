import os
from tqdm import tqdm


base_path = './wikiextractor/text'
folders = os.listdir(base_path)

merge_text = "./merge_text.txt"

for folder in tqdm(folders, desc='folder', position=0):
    file_path = f"{base_path}/{folder}"
    files = os.listdir(file_path)
    for file in tqdm(files, desc='file', position=1, leave=False):
        with open(merge_text, 'a') as destination_file, open(f'{file_path}/{file}', 'r') as departure_file:
            lines = departure_file.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith('<doc') or line.startswith('</doc') or line.startswith('&'):
                    continue
                destination_file.write(f'{line}\n')