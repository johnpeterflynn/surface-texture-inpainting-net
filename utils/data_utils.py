from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import glob
import os
from torch_geometric.data import Data
from utils import vector_math


class HierarchicalData(Data):
    def __init__(self, x=None, color=None, pos=None, mask=None, labels=None, edge_index=None, name=None):#, num_vertices=None):
        super().__init__()
        self.x = x
        self.color = color
        self.pos = pos
        self.mask = mask
        self.labels = labels
        self.edge_index = edge_index
        self.name = name
        #self.num_vertices = num_vertices

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'num_vertices':
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.num_vertices[0]
        elif key == 'x' or key == 'color' or key == 'pos' or key == 'mask' or key == 'labels':
            return 0
        else:
            for level in range(1, len(self.num_vertices)):
                if key == f"hierarchy_edge_index_{level}":
                    return self.num_vertices[level]
                # TODO: Are traces incremented correctly?
                elif key == f"hierarchy_trace_index_{level}":
                    return self.num_vertices[level]

            return super().__inc__(key, value, *args, **kwargs)


def get_train_val_test_split(data_ids, num_in_train_step, num_in_val_step, data_select_path=None, slice_start=0,
                             slice_step=1, slice_end=None):
    """Separates data_ids into three separate lists of ids for train, val and test respectively.
    """
    train_flag = 0
    val_flag = 1
    test_flag = 2

    if slice_end is None:
        slice_end = len(data_ids)

    if data_select_path is not None:
        with open(data_select_path) as csv_file:
            data = pd.read_csv(csv_file, delimiter=' ', index_col=None, header=None)
            use_indices = np.array(data.values).squeeze()
    else:
        use_indices = np.arange(slice_end)

    data_ids = [data_ids[i] for i in use_indices]
    data_ids = data_ids[slice(slice_start, slice_end, slice_step)]

    data_ids = np.array(data_ids)
    category_indices = [(test_flag if (i // (num_in_train_step + num_in_val_step)) % 2 == 0 else val_flag)
                        if (i % (num_in_train_step + num_in_val_step)) >= num_in_train_step else train_flag for i
                        in range(len(data_ids))]

    category_indices = np.array(category_indices)
    train_ids = data_ids[category_indices == train_flag]
    val_ids = data_ids[category_indices == val_flag]
    test_ids = data_ids[category_indices == test_flag]

    return train_ids, val_ids, test_ids


# NOTE: DEPRECATED
def get_train_val_split(pose_files, skip, max_index=None, stride=1):
    if max_index is None:
        max_index = len(pose_files)
    pose_files = pose_files[0:max_index:stride]
    train_filenames = [pose_files[i] for i in range(len(pose_files)) if (i % skip) != 0]
    val_filenames = [pose_files[i] for i in range(len(pose_files)) if (i % skip) == 0]

    return train_filenames, val_filenames


def load_poses(pose_files):
    rots = []
    ts = []
    for file in pose_files:
        with open(file) as csv_file:
            data = pd.read_csv(csv_file, delimiter=' ', index_col=None, header=None)

            rot = np.array(data.values[0:3, 0:3])
            t = np.array(data.values[0:3, -1])
            rots.append(rot)
            ts.append(t)
    return rots, ts


def get_nn_indices(neighbors, items):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(neighbors)

    neigh_dist, neigh_index = neigh.kneighbors(items)

    return neigh_index


def get_rotvecs_from_matrices(rots):
    rotvecs = []
    for rot in rots:
        rotvecs.append(R.from_matrix(rot).as_rotvec())

    return rotvecs


# Load train and validation poses
def get_val_nn_train_angles(train_rots, val_rots, unit='deg'):
    # Transform to axis representation
    train_rotvecs = get_rotvecs_from_matrices(train_rots)
    val_rotvecs = get_rotvecs_from_matrices(val_rots)

    # Find nearest neighbors by angle
    nn_indices = get_nn_indices(train_rotvecs, val_rotvecs)

    # Get angles to nearest neighbords in degrees
    angles = []
    for i, val_rotvec in enumerate(val_rotvecs):
        angle = vector_math.angle_between(val_rotvec, train_rotvecs[nn_indices[i, 0]])
        if unit == 'deg':
            angle = np.rad2deg(angle)
        angles.append(angle)

    return angles


def load_filenames_sorted(directory):
    filenames = load_filenames(directory)
    filenames.sort()
    return filenames

def load_filenames(directory, sorted=True):
    filenames = glob.glob(os.path.join(directory, '*'))
    return filenames


def load_filename_pairs(file_dir_a, file_dir_b):
    a_filenames = load_filenames_sorted(file_dir_a)
    b_filenames = load_filenames_sorted(file_dir_b)

    filename_pairs = list(zip(a_filenames, b_filenames))

    return filename_pairs
