import sys
sys.path.append("../")
import converter.xml2txt as xml2txt
from tqdm import tqdm
import os
from glob import glob
import shutil

def main():
    converter = xml2txt.xml2txt("levels")
    train_data, remove_file_list = converter.xml2vector(True)
    for i, data in enumerate(train_data):
        data_ = []
        for d in data:
            if sum(d) == 0:
                break
            data_.append(d)
        train_data[i] = data_
    train_data, test_data = train_data[:180], train_data[180:]
    with open("train.txt", "w") as f:
        for train_d in tqdm(train_data):
            train_d_ = ""
            for d in train_d:
                d = list(map(str, d))
                c = " ".join(d)
                train_d_ += c + "  "
            f.write(str(train_d_)+"\n")
    with open("valid.txt", "w") as f:
        for train_d in tqdm(test_data):
            train_d_ = ""
            for d in train_d:
                d = list(map(str, d))
                c = " ".join(d)
                train_d_ += c + "  "
            f.write(str(train_d_)+"\n")  


def remove_file():
    converter = xml2txt.xml2txt("../../../IratusAves/levels")
    train_data, remove_file_list = converter.xml2vector(True)
    print(len(train_data))
    print(remove_file_list)
    file_list = glob("../../../IratusAves/levels/*")
    save_file_list = list(set(file_list) - set(remove_file_list))
    for file_name in save_file_list:
        shutil.copy(file_name, "save_level/"+ os.path.basename(file_name))

if __name__ == "__main__":
    main()
