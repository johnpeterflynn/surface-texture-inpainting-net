from collections import defaultdict
import glob
import hashlib
import random
import numpy as np
from torch.utils.data import Dataset


SCANNET_TRAIN_FILE = 'datasets/meta/scannet/scannetv2_train.txt'
SCANNET_VAL_FILE = 'datasets/meta/scannet/scannetv2_val.txt'
SCANNET_TEST_FILE = 'datasets/meta/scannet/scannetv2_test.txt'


class ScanNetDataset(Dataset):
    def __init__(self, root_dir, end_level, is_train, benchmark, index2allfilenames, no_train_cropped, num_crops_per_scene, transform, max_num_scenes, used_repeated_reconsts):
        self._root_dir = root_dir
        self._is_train = is_train
        self._benchmark = benchmark
        self._end_level = end_level
        self._no_train_cropped = no_train_cropped
        self._transform = transform
        self._max_num_scenes = max_num_scenes
        self._use_repeated_reconsts = used_repeated_reconsts
        self._num_crops_per_scene = num_crops_per_scene
        self._legal_filenames = np.asarray(list(set(index2allfilenames)))

    def get_approved_filenames(self):
        if self._is_train:
            file_path = SCANNET_TRAIN_FILE
        else:
            if not self._benchmark:
                file_path = SCANNET_VAL_FILE
            else:
                file_path = SCANNET_TEST_FILE

        with open(file_path, 'r') as f:
            set_file_paths = f.read().splitlines()

        set_file_paths = [x for x in set_file_paths if x in self._legal_filenames]

        if not self._use_repeated_reconsts:
            set_file_paths = [x for x in set_file_paths if int(x.split('/')[-1].split('_')[1]) == 0]
        return set_file_paths[:self._max_num_scenes] if self._max_num_scenes >= 0 else set_file_paths

    def select_rand_crops_per_scene(self, scene_names_to_cropped_file_paths):
        dirs = []
        for scene_name, scene_crop_paths in scene_names_to_cropped_file_paths.items():
            # Apply pseudorandom shuffle by seeding with hash of scene name so that
            #  crops are selected from uniform distribution but identical between program executions.
            #  And shuffle indices so that a larger number of crops per scene contains the sets of
            #  smaller numbers of crops per scene.
            scene_crop_paths.sort()
            seed = int(hashlib.sha1(scene_name.encode("utf-8")).hexdigest(), 16) % (10 ** 8)
            indices = np.arange(0, len(scene_crop_paths) - 1, dtype=int)
            random.Random(seed).shuffle(indices)
            num_crops = min(self._num_crops_per_scene,
                            len(scene_crop_paths)) if self._num_crops_per_scene >= 0 else len(scene_crop_paths)
            indices = indices[:num_crops]
            dirs += [scene_crop_paths[i] for i in indices]
        return dirs