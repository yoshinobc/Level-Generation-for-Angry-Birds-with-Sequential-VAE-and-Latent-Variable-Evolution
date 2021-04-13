import glob
import numpy as np
import os
# x_min, x_max:-1.5 4.0


class xml2txt(object):
    def __init__(self, file_dir):
        self.material2code = {"wood": 0, "stone": 1, "ice": 2}
        self.type2code = {"empty": 0,
                          "Platform": 1,
                          "TNT": 2,
                          "Pig": 3,
                          "RectFat": 4,
                          "RectFat90": 5,
                          "RectSmall": 6,
                          "RectSmall90": 7,
                          "RectMedium": 8,
                          "RectMedium90": 9,
                          "RectBig": 10,
                          "RectBig90": 11,
                          "RectTiny": 12,
                          "RectTiny90": 13,
                          "SquareSmall": 14,
                          "SquareTiny": 15,
                          "SquareHole": 16,
                          "TriangleHole": 17,
                          "TriangleHole90": 18,
                          "Triangle": 19,
                          "Triangle90": 20,
                          "Circle": 21,
                          "CircleSmall": 22
                          }
        self.range2code = {0: 5,
                           1: 6,
                           2: 6,
                           3: 6,
                           4: 8,
                           5: 4,
                           6: 8,
                           7: 2,
                           8: 16,
                           9: 2,
                           10: 20,
                           11: 2,
                           12: 4,
                           13: 2,
                           14: 4,
                           15: 2,
                           16: 8,
                           17: 8,
                           18: 8,
                           19: 8,
                           20: 8,
                           21: 8,
                           21: 8,
                           22: 4
                           }
        self.block2size = {
            "Platform": [0.62, 0.62],
            "Pig": [0.5, 0.45],
            "TNT": [0.55, 0.55],
            "SquareHole": [0.84, 0.84],
            "RectFat": [0.85, 0.43],
            "RectFat90": [0.43, 0.85],
            "SquareSmall": [0.43, 0.43],
            "SquareTiny": [0.22, 0.22],
            "RectTiny": [0.43, 0.22],
            "RectTiny90": [0.22, 0.43],
            "RectSmall": [0.85, 0.22],
            "RectSmall90": [0.22, 0.85],
            "RectMedium": [1.68, 0.22],
            "RectMedium90": [0.22, 1.68],
            "RectBig": [2.06, 0.22],
            "RectBig90": [0.22, 2.06],
            "CircleSmall": [0.45, 0.45],
            "Circle": [0.8, 0.8],
            "Triangle": [0.82, 0.82],
            "Triangle90": [0.82, 0.82],
            "TriangleHole": [0.82, 0.82],
            "TriangleHole90": [0.82, 0.82]
        }
        self.line2vec = {}
        self.line2vec_index = 0
        self.y_range = 94
        self.n_line = 50
        self.file_list = glob.glob(file_dir+"/*")

    def xml2txt(self, xml_file, is_pre=False):
        vector = self.xml2vector_core(xml_file, is_pre)
        if vector == None:
            print("None")
        if len(vector) != 50:
            print(xml_file + " size is not 30")
        for vec in vector:
            if len(vec) != self.y_range:
                print(xml_file + " range is not " + self.y_range)
        return vector

    def xml2vector(self, is_pre=False, condition=False):
        train_data = []
        conds = []
        remove_file_list = []
        for file_name in self.file_list:
            vector = self.xml2vector_core(file_name, is_pre)
            cond = os.path.basename(os.path.dirname(file_name))
            if vector == None:
                remove_file_list.append(file_name)
                continue
            if len(vector) != 30:
                print("f", file_name)
            for vec in vector:
                if len(vec) != self.y_range:
                    print("f", file_name)
            train_data.append(vector)
            conds.append(cond)
        if condition:
            if is_pre:
                return train_data, conds
            else:
                return np.array(train_data), np.array(conds)
        if is_pre:
            return train_data, remove_file_list
        else:
            return np.array(train_data)

    def preprocess(self, text):
        return text[1:-3]

    def load_xml(self, xml_txt):
        contents = []
        xml_txt = xml_txt.split("\n")
        for text in xml_txt:
            block_name = ""
            x, y = 0, 0
            text = self.preprocess(text)
            content = text.split(" ")
            material = ""
            if len(content) >= 2:
                if content[0] == "Block":
                    material = content[2][10:-1]
                    if str(content[5][10:-1]) != "0" and str(content[5][10:-1]) != "0.0":
                        if content[5][10:-1] == "90.0" or content[5][10:-1] == "135.0":
                            block_name = content[1][6:-1] + content[5][10:-3]
                        else:
                            block_name = content[1][6:-1] + content[5][10:-1]
                    else:
                        block_name = content[1][6:-1]
                    x, y = float(content[3][3:-1]), float(content[4][3:-1])
                elif content[0] == "Platform" or content[0] == "Pig":
                    block_name = content[0]
                    x, y = float(content[3][3:-1]), float(content[4][3:-1])
                elif content[0] == "TNT":
                    block_name = content[0]
                    x, y = float(content[2][3:-1]), float(content[3][3:-1])
                else:
                    continue
                x = round(x, 1)
                contents.append((block_name, x, y, material))
        return contents

    def left_move(self, contents):
        x_range = sorted(
            contents, key=lambda x: x[1])[-1][1] - sorted(
            contents, key=lambda x: x[1])[0][1]
        most_left = sorted(
            contents, key=lambda x: x[1])[0][1]
        diff = 0 - most_left * 10
        #print(diff)
        return contents, x_range

    def xml2vector_core(self, file_name, is_pre):
        f = open(file_name, "r")
        xml_list = f.read()
        contents = self.load_xml(xml_list)
        if len(contents) <= 5:
            return None
        contents = sorted(
            contents, key=lambda x: x[2] - self.block2size[x[0]][1] / 2)
        contents, diff = self.left_move(contents)
        lines = [[] for _ in range(self.n_line)]
        ind = -1
        line_y = -1000
        for content in contents:
            y = content[2] - self.block2size[content[0]][1] / 2
            if y < -3.8:
                continue
            if line_y < y - 0.1:
                ind += 1
            line_y = y
            lines[ind].append(content)
        train_data = []
        for line in lines:
            vec = np.zeros(self.y_range)
            for l in line:
                if l[3] == "":
                    c = int(self.type2code[l[0]])
                else:
                    c = int(self.type2code[l[0]] + self.material2code[l[3]]*19)
                x_median = int(l[1] * 10 + diff)
                block_range = int(self.range2code[self.type2code[l[0]]] / 2)
                for x in range(x_median - block_range, x_median + block_range):
                    if int(x) < 0 or int(x) >= self.y_range:
                        #continue
                        return None
                    vec[int(x)] = int(c)
            if tuple(vec) in self.line2vec.keys():
                pass
            else:
                self.line2vec[tuple(vec)] = self.line2vec_index
                self.line2vec_index += 1
            if is_pre:
                vec = list(map(int, vec))
                train_data.append(vec)
            else:
                train_data.append(self.line2vec[tuple(vec)])
        if is_pre:
            pass
        else:
            train_data = np.array(train_data)
        return train_data

    def maketest(self):
        file_name = self.file_list[0]
        vector = self.xml2vector_core(file_name, True)
        test_d = "  ".join(map(str, vector))
        return test_d
