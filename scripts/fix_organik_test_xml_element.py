import os
import xml.etree.cElementTree as ET

##########################
#SET ANNOTATION XML DIRECTORY
##########################
DIR_PREFIX = "Annotations/TEST/O"
PATH_PREFIX = "/wastedata/"
DESTINATION_DIR = "Annotations/TEST/O_update"
FILE_EXT = ".xml"
##########################

def modify_file(filename, pathfile):
    tree = ET.parse(pathfile)
    root = tree.getroot()
    filetag = root.find('filename')
    pathtag = root.find('path')
    filetag.text = filename
    pathtag.text = PATH_PREFIX + filename
    tree.write(f"{DESTINATION_DIR}/{filename}")

def start():
    if not os.path.exists(DESTINATION_DIR):
        os.makedirs(DESTINATION_DIR)
    for filename in os.listdir(DIR_PREFIX):
        if filename.endswith(FILE_EXT):
            pathfile = f"{DIR_PREFIX}/{filename}"
            if os.stat(pathfile).st_size > 0:
                modify_file(filename, pathfile)
            else:
                print(f"Empty file: {pathfile}")
        else:
            print(f"Skipping file: {pathfile}")

if __name__ == "__main__":
    start()
