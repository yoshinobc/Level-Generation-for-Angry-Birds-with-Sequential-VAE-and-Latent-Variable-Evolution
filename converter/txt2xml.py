import glob
from icecream import ic

class txt2xml(object):
    def __init__(self):
        self.material2code = {"wood": 0, "stone": 1, "ice": 2}
        self.code2material = {0: "wood", 1: "stone", 2: "ice"}
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
    '''
    def stract_key_rotation_object_name_material(self, v):
        if v >= 4:
            v_ = (v - 4) % 19 + 4
        else:
            v_ = v
        key = [k for k, va in self.type2code.items() if va == v_][0]
        if "90" in key:
            rotation = 90
            object_name = key[:-2]
        else:
            rotation = 0
            object_name = key
        if object_name in ["empty", "TNT", "Pig"]:
            material = "BasicSmall"
        elif object_name == "Platform":
            material = "Platform"
        else:
            material = self.re_material2code[(v - 4) // 19]
        return key, rotation, object_name, material
    '''
    def stract_key_rotation_objectname_material_from_vec(self, vec):
        objects = []
        vec = vec.split(" ")
        for i, v in enumerate(vec):
            try:
                v = int(v)
            except:
                v = 0
            if v != 0:
                object_key = v if v < 4 else (v - 4) % 19 + 4
                threshold_num = self.range2code[object_key]
                if threshold_num <= vec[i: i + threshold_num].count(v) and vec[i] != 1:
                    continue
                for j in range(i, i + threshold_num):
                    if j < 0 or j >= len(vec):
                        continue
                    if vec[j] != 1:
                        vec[j] = 0
                object_name_ = [
                    k for k, va in self.type2code.items() if va == object_key][0]
                x_median = i + int(threshold_num / 2)
                if "90" in object_name_:
                    rotation = 90
                    object_name = object_name_[:-2]
                else:
                    rotation = 0
                    object_name = object_name_
                if object_name in ["empty", "TNT", "Pig"]:
                    material = "BasicSmall"
                elif object_name == "Platform":
                    material = "Platform"
                else:
                    material = self.code2material[(v - 4) // 19]
                objects.append(
                    [x_median, object_name_, rotation, object_name, material])
        return objects

    def stract_min_max_list(self, other_contents):
        range_covers = []
        for other_content in other_contents:
            x_min, x_max = other_content[1] - self.block2size[other_content[0]
                                                          ][0] / 2, other_content[1] + self.block2size[other_content[0]][0] / 2
            y_min, y_max = other_content[2] - self.block2size[other_content[0]
                                                          ][1] / 2, other_content[2] + self.block2size[other_content[0]][1] / 2
            range_covers.append((x_min, x_max, y_min, y_max))
        return range_covers

    def stract_min_max_content(self, content):
        x_min, x_max = content[1] - self.block2size[content[0]
                                                ][0] / 2, content[1] + self.block2size[content[0]][0] / 2
        y_min, y_max = content[2] - self.block2size[content[0]
                                                ][1] / 2, content[2] + self.block2size[content[0]][1] / 2
        range_content = (x_min, x_max, y_min, y_max)
        return range_content

    def is_under(self, x, key, range_content, content):
        x_min, x_max = x - self.block2size[key][0]/2, x + self.block2size[key][0]/2
        if content[0] == "Platform" and key == "Platform":
            return False
        if range_content[0] + 0.1 <= x_max and x_min + 0.1 <= range_content[1]:
            return True
        else:
            return False

    def calc_y(self, x, key, contents, current_platform_y):
        range_covers = self.stract_min_max_list(contents)
        y = -3.6 + self.block2size[key][1]/2
        for range_cover, content in zip(range_covers, contents):
            if self.is_under(x, key, range_cover, content):
                y = range_cover[3] + self.block2size[key][1] / 2 + 0.05
        if key == "Platform":
            if y == -3.6 + self.block2size[key][1]/2 and not self.platform_first:
                y = current_platform_y
            current_platform_y = y
            self.platform_first = False
        return current_platform_y, y

    def create_text(self, object_name, x, y, material, rotation):
        if object_name == "Pig":
            text = '<' + object_name + ' type="' + "BasicSmall" + '" material="' + "" + '" x="' + \
                str(x) + '" y="' + str(y) + \
                '" rotation="' + str(rotation) + '" />\n'
        elif object_name == "TNT":
            text = '<' + object_name + ' type="' + "" + '" x="' + \
                str(x) + '" y="' + str(y) + \
                '" rotation="' + str(rotation) + '" />\n'
        elif object_name == "Platform":
            text = '<Platform type="' + "Platform" + '" material="' + "" + '" x="' + \
                str(x) + '" y="' + str(y) + \
                '" rotation="' + str(rotation) + '" />\n'
        else:
            text = '<Block type="' + object_name + '" material="' + material + '" x="' + \
                str(x) + '" y="' + str(y) + \
                '" rotation="' + str(rotation) + '" />\n'
        return text

    def txt2xml(self, text):
        contents = []
        texts = '<?xml version="1.0" encoding="utf-16"?>\n'
        texts += '<Level width ="2">\n'
        texts += '<Camera x="0" y="2" minWidth="20" maxWidth="30">\n'
        texts += '<Birds>\n'
        texts += '<Bird type="BirdBlue"/>\n'
        texts += '<Bird type="BirdRed"/>\n'
        texts += '<Bird type="BirdYellow"/>\n'
        texts += '<Bird type="BirdBlue"/>\n'
        texts += '<Bird type="BirdBlue"/>\n'
        texts += '<Bird type="BirdBlue"/>\n'
        texts += '<Bird type="BirdRed"/>\n'
        texts += '</Birds>\n'
        texts += '<Slingshot x="-8" y="-2.5">\n'
        texts += '<GameObjects>\n'
        self.platform_first = True
        current_platform_y = None
        vector = text.split("  ")
        for vec in vector:
            vec = vec.split(" ")
            if "\n" in vec:
                continue
            else:
                objects = self.stract_key_rotation_objectname_material_from_vec(
                    vec)
                for x_median, key, rotation, object_name, material in objects:
                    if key == "empty":
                        continue
                    x = (x_median - 20) / 10
                    current_platform_y, y = self.calc_y(
                        x, key, contents, current_platform_y)
                    content = (key, x, y, material)
                    contents.append(content)
                    text = self.create_text(
                        object_name, x, y, material, rotation)
                    texts += text
        texts += "</GameObjects>\n"
        texts += "</Level>\n"
        return texts

    def vector2xml(self, vector):
        contents = []
        texts = '<?xml version="1.0" encoding="utf-16"?>\n'
        texts += '<Level width ="2">\n'
        texts += '<Camera x="0" y="2" minWidth="20" maxWidth="30">\n'
        texts += '<Birds>\n'
        texts += '<Bird type="BirdBlue"/>\n'
        texts += '<Bird type="BirdRed"/>\n'
        texts += '<Bird type="BirdYellow"/>\n'
        texts += '<Bird type="BirdBlue"/>\n'
        texts += '<Bird type="BirdBlue"/>\n'
        texts += '<Bird type="BirdBlue"/>\n'
        texts += '<Bird type="BirdRed"/>\n'
        texts += '</Birds>\n'
        texts += '<Slingshot x="-8" y="-2.5">\n'
        texts += '<GameObjects>\n'
        self.platform_first = True
        current_platform_y = None
        for vec in vector:
            if vec == "<unk>":
                continue
            elif vec == "":
                continue
            elif vec == "<s>" or vec == "</s>":
                continue
            else:
                objects = self.stract_key_rotation_objectname_material_from_vec(vec)
                for center, key, rotation, object_name, material in objects:
                    if key == "empty":
                        continue
                    x = (center - 20) / 10
                    current_platform_y, y = self.calc_y(
                        x, key, contents, current_platform_y)
                    content = (key, x, y, material)
                    contents.append(content)
                    text = self.create_text(
                        object_name, x, y, material, rotation)
                    texts += text
        texts += "</GameObjects>\n"
        texts += "</Level>\n"
        return texts
