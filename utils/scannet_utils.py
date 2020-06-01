import os
import csv
import json
import numpy as np
import open3d as o3d

CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
NO_CLASS_ID = np.array([0])
VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
COLOR_PALETTE = np.array([
       (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),
       (247, 182, 210),		# desk
       (66, 188, 102),
       (219, 219, 141),		# curtain
       (140, 57, 197),
       (202, 185, 52),
       (51, 176, 203),
       (200, 54, 131),
       (92, 193, 61),
       (78, 71, 183),
       (172, 114, 82),
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),
       (153, 98, 156),
       (140, 153, 101),
       (158, 218, 229),		# shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),
       (227, 119, 194),		# bathtub
       (213, 92, 176),
       (94, 106, 211),
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)
    ])


def list_dir_single(directory):
    for root, dirs, files in os.walk(directory):
        return dirs


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_scannet(scans_dir, scene_name):
    class_dict = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
                  "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
                  "11": 11, "12": 12, "14": 13, "16": 14,
                  "24": 15, "28": 16, "33": 17, "34": 18,
                  "36": 19, "39": 20}

    label_map = {}
    # label_map_file = os.path.join(args.input_folder, "scannet-labels.combined.tsv")
    label_map_file = os.path.join(scans_dir, "scannetv2-labels.combined.tsv")
    with open(label_map_file, 'r') as f:
        lines = csv.reader(f, delimiter='\t')
        cnt = 0
        for line in lines:
            if cnt == 0:
                print(line)
            else:
                if len(line[4]) > 0:
                    label_map[line[1]] = line[4]
                else:
                    label_map[line[1]] = '0'
            cnt += 1

    print(scene_name)
    aggregation_file = os.path.join(scans_dir, scene_name, scene_name + ".aggregation.json")
    seg_file = os.path.join(scans_dir, scene_name, scene_name + "_vh_clean_2.0.010000.segs.json")
    ply_file = os.path.join(scans_dir, scene_name, scene_name + "_vh_clean_2.ply")
    pcd = o3d.io.read_point_cloud(ply_file)
    mesh = load_mesh(ply_file, True)

    with open(aggregation_file) as f:
        aggregation_data = json.load(f)

    with open(seg_file) as f:
        seg_data = json.load(f)

    str_segments = seg_data["segIndices"]
    int_segments = np.asarray(str_segments, dtype='int32')
    out_labels = np.zeros((len(int_segments)), dtype='int32')

    num_objects = len(aggregation_data["segGroups"])
    for obj in aggregation_data["segGroups"]:
        str_lbl = obj["label"]
        for seg in obj["segments"]:
            int_seg = int(seg)
            ind = int_segments == int_seg
            if str_lbl in label_map:
                lb = label_map[str_lbl]
            else:
                lb = '-'
            if lb in class_dict.keys():
                out_labels[ind] = class_dict[lb]
            else:
                out_labels[ind] = 0

    return mesh, pcd, out_labels


def load_mesh(path, compute_normals=False):
    print('Loading mesh from path', path)
    mesh = o3d.io.read_triangle_mesh(path, True)
    # mesh.remove_non_manifold_edges()
    # mesh.remove_duplicated_vertices()
    # mesh.remove_duplicated_triangles()
    # mesh.remove_degenerate_triangles()
    # mesh.remove_unreferenced_vertices()
    # mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(mesh)

    if compute_normals:
        if not mesh.has_vertex_normals():
            print('Computing vertex normals')
            mesh.compute_vertex_normals()

        if not mesh.has_triangle_normals():
            print('Computing triangle normals')
            mesh.compute_triangle_normals()
        mesh.normalize_normals()

    mesh.compute_adjacency_list()

    print("Try to render a mesh with normals (exist: " +
          str(mesh.has_vertex_normals()) + ") and colors (exist: " +
          str(mesh.has_vertex_colors()) + ") and uvs (exist: " +
          str(mesh.has_adjacency_list()) + ") and adjacency list (exist: " +
          str(mesh.has_triangle_uvs()) + ")")

    return mesh


# color palette for nyu40 labels
def color_palette():
    return COLOR_PALETTE


def valid_color_palette():
    return COLOR_PALETTE[np.concatenate((NO_CLASS_ID, VALID_CLASS_IDS))]


def get_class_name(idx):
    return CLASS_LABELS[idx]


class ScannetParams:
    def __init__(self):
        # 200 scenes, 0.1
        self.class_count = np.asarray([290043, 257756, 36185, 18419, 70674, 27485, 44351, 38429, 29290, 18326, 4295,
                                       2843, 16040, 9923, 4008, 1915, 2598, 1943, 2562, 26012])

        # 200 scenes, 0.05
        #self.class_count = np.asarray([1078745, 966338, 131619, 68992, 257120, 102138, 166306, 146127, 106970, 69564,
         #                              16464, 11048, 59853, 39364, 15043, 6937, 9383, 7042, 9162, 99354])

        # 200 scenes, 0.2
        #self.class_count = np.asarray([76572, 69439, 9458, 4829, 19395, 6894, 11660, 10061, 7893, 4651, 1093, 702,
        #                               3835, 2335, 1000, 476, 607, 527, 677, 6658])

        # 8 scenes (6 train), 0.05
        #self.class_count = np.asarray([36650, 38207, 12950, 6670, 9245, 8260, 6745, 7547, 6697, 0, 546, 261, 2585, 7019,
        #                               1158, 0, 290, 234, 0, 4367])

        # 32 scenes (37 train), 0.05
        #self.class_count = np.asarray([159948, 173820, 21164, 6670, 48023, 29758, 33373, 19366, 15902, 10662, 3453, 803,
        #                               5250, 9249, 3021, 1050, 1743, 1936, 1444, 15780])

        # scene 0004_00, 0.4
        #self.class_count = np.asarray([32, 694, 0, 0, 287, 0, 125, 0, 334, 0, 0, 0, 0, 0, 0, 0, 0,
        #                               0, 0, 0])
        # scene 0004_00, 0.05
        #self.class_count = np.asarray([344, 10044, 0, 0, 4054, 0, 1502, 0, 4242, 0, 0, 0, 0, 0, 0, 0, 0,
        #                               0, 0, 0])
        self.class_freq = 100.0 * self.class_count / self.class_count.sum()
        #self.class_freq = np.asarray([40.82, 27.79, 2.96, 2.81, 5.04, 2.94, 2.81, 2.51, 1.06, 2.25,
        #                              0.42, 0.73, 1.86, 1.43, 0.46, 0.20, 0.3, 0.38, 0.36, 2.84])
        self.class_weights = np.asarray([-np.log(x / 100.0) if x > 0 else 0 for x in self.class_freq])
        self.num_classes = len(self.class_freq) + 1
