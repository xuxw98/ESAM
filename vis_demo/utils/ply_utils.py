import  numpy as np
import  struct
import math

label_mapper=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])


def create_color_palette():
    return [
       (0, 0, 0),
       (174, 199, 232),		# wall0
       (152, 223, 138),		# floor1
       (31, 119, 180), 		# cabinet2
       (255, 187, 120),		# bed3
       (188, 189, 34), 		# chair4
       (140, 86, 75),  		# sofa5
       (255, 152, 150),		# table6
       (214, 39, 40),  		# door7
       (197, 176, 213),		# window8
       (148, 103, 189),		# bookshelf9
       (196, 156, 148),		# picture10
       (23, 190, 207), 		# counter11
       (178, 76, 76),
       (247, 182, 210),		# desk13
       (66, 188, 102),
       (219, 219, 141),		# curtain15
       (140, 57, 197),
       (202, 185, 52),
       (51, 176, 203),
       (200, 54, 131),
       (92, 193, 61),
       (78, 71, 183),
       (172, 114, 82),
       (255, 127, 14), 		# refrigerator23
       (91, 163, 138),
       (153, 98, 156),
       (140, 153, 101),
       (158, 218, 229),		# shower curtain27
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		# toilet32
       (112, 128, 144),		# sink33
       (96, 207, 209),
       (227, 119, 194),		# bathtub35
       (213, 92, 176),
       (94, 106, 211),
       (82, 84, 163),  		# otherfurn38
       (100, 85, 144)
    ]

label_name=[
"wall",
"floor",
"cabinet",
"bed",
"chair",
"sofa",
"table",
"door",
"window",
"bookshelf",
"picture",
"counter",
"desk",
"curtain",
"refrigerator",
"shower curtain",
"toilet",
"sink",
"bathtub",
"otherfurn",
]

def write_ply(point_cloud,rgb_cloud=None,label_cloud=None,output_dir="./",name="test",hasrgb=False,haslabel=False):
    point_count=point_cloud.shape[0]
    ply_file = open(output_dir+name+ ".ply", 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex " + str(point_count) + "\n")
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")

    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    if haslabel:
        ply_file.write("property uchar label\n")


    ply_file.write("end_header\n")
    color=create_color_palette()
    for i in range(point_count):
        ply_file.write(str(point_cloud[i, 0]) + " " +
                       str(point_cloud[i, 1]) + " " +
                       str(point_cloud[i, 2]))
        if hasrgb:
            ply_file.write(" "+str(int(rgb_cloud[i, 0])) + " " +
                           str(int(rgb_cloud[i, 1])) + " " +
                           str(int(rgb_cloud[i, 2])))
        if haslabel:
            ply_file.write(" "+str(color[int(label_cloud[i])][0]) + " " +
                           str(color[int(label_cloud[i])][1]) + " " +
                           str(color[int(label_cloud[i])][2]))
            ply_file.write((" "+str(int(label_cloud[i]))))

        ply_file.write("\n")
    ply_file.close()
    print("save result to "+output_dir+name+ ".ply")

