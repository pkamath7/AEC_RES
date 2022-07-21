import os 
import json

DEBUG_PRINT=False
def get_file_descriptors(dataset_path):
    directory={}
    for i , (dirpath, dirname, filename) in enumerate(os.walk(dataset_path)):
        if(dirpath!=dataset_path):
            dirname=dirpath.split("/")[-1]
            files={}
            file_list=[]
            index=0
            for file in filename:
                filepath = os.path.join( dirpath, file)
                if ( (filepath.endswith('.wav'))):
                    if(file.startswith('.')):
                        pass
                    else:
                        file_list.append(filepath)
            file_list.sort()
            if(len(file_list)>0):
                for filepath in file_list:
                    files[index]=filepath
                    index+=1
                directory[dirname]=files
    return directory

directory=get_file_descriptors("../dataset_daps/daps")
json_path="./dir.json"
json_directory = json.dumps(directory,indent=4)
with open(json_path, "w") as outfile:
    outfile.write(json_directory)