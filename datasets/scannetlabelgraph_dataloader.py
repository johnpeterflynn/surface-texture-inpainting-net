from collections import defaultdict
import glob
import numpy as np
import torch
import open3d
from torch_geometric.data import Data, DataListLoader
from torch_geometric.loader import DataLoader as GraphLevelDataLoader
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import transform
from typing import List
from easydict import EasyDict
from datasets import scannet_dataset
from utils import data_utils


def unit_test_compare_train_val(train_dataset, val_dataset):
    val_set = set(val_dataset.index2filenames)
    train_list = train_dataset.index2filenames

    for item in train_list:
        assert item not in val_set and item.split('/')[-1].rsplit('_', 1)[0] + '.pt' not in val_set, \
            'ERROR: Validation dataset contains the same data as train dataset for: {}'.format(item)


class ScanNetLabelDataSet(scannet_dataset.ScanNetDataset):
    def __init__(self, root_dir, end_level, is_train, benchmark, index2allfilenames, no_train_cropped=False,
                 num_crops_per_scene=None, transform=None, max_num_scenes=None, used_repeated_reconsts=True):
        super(ScanNetLabelDataSet, self).__init__(root_dir, end_level, is_train, benchmark, index2allfilenames,
                                                  no_train_cropped, num_crops_per_scene, transform, max_num_scenes,
                                                  used_repeated_reconsts)
        self.index2filenames = self._load()
        self.index2filenames = np.asarray(self.index2filenames)

    def _load(self) -> List[str]:
        set_file_paths = self.get_approved_filenames()

        if self._is_train:
            if self._no_train_cropped:
                filenames = [x.split('/')[-1] for x in glob.glob(f"{self._root_dir}/*.pt")
                             if x.split('/')[-1].replace('.pt', '') in set_file_paths]
            else:
                existing_cropped_file_paths = defaultdict(list)
                for x in glob.glob(f"{self._root_dir}/*.pt"):
                    scene_name = x.split('/')[-1].replace('.pt', '').rsplit('_', 1)[0]
                    if scene_name in set_file_paths:
                        existing_cropped_file_paths[scene_name].append(x.split('/')[-1])

                filenames = self.select_rand_crops_per_scene(existing_cropped_file_paths)
        else:
            filenames = [x.split('/')[-1] for x in glob.glob(f"{self._root_dir}/*.pt")
                         if x.split('/')[-1].replace('.pt', '') in set_file_paths]
        return filenames

    def __len__(self):
        return len(self.index2filenames)

    @staticmethod
    def collate_fn(batch):
        return batch

    def __getitem__(self, index: int):
        name = self.index2filenames[index]
        file_path = f"{self._root_dir}/{name}"
        saved_tensors = torch.load(file_path)

        coords = saved_tensors['vertices'][:self._end_level]

        if not self._benchmark:
            labels = saved_tensors['labels']
        else:
            labels = None

        edges = saved_tensors['edges'][:self._end_level]

        # TODO: Normalize position, maybe normals
        sample = data_utils.HierarchicalData(
            x=torch.cat([coords[0][:, 3:9], coords[0][:, :3]], dim=-1),
            edge_index=edges[0].t().contiguous(),
            labels=labels,
            name=name,
            )

        if self._is_train:
            traces = saved_tensors['traces'][:self._end_level - 1]
        else:
            sample.original_index_traces = saved_tensors['traces'][0]
            traces = saved_tensors['traces'][1:self._end_level]

        sample.num_vertices = [coords[0].shape[0]]
        for level in range(1, len(edges)):
            setattr(sample, f"hierarchy_edge_index_{level}", edges[level].t().contiguous())
            setattr(sample, f"hierarchy_trace_index_{level}", traces[level - 1])
            #setattr(sample, f"pos_{level}", coords[level][:, :3])
            sample.num_vertices.append(
                int(sample[f"hierarchy_trace_index_{level}"].max() + 1))

        if self._transform:
            sample = self._transform(sample)

        return sample


class ScanNetGraphDataLoader:
    def __init__(self, config, multi_gpu):
        self.classes = ['unannotated', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                        'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refridgerator', 'shower curtain',
                        'toilet', 'sink', 'bathtub', 'otherfurniture']

        self.config = EasyDict(config)
        self.index2allfilenames = self._load_all_scene_names()
        # print('Computing train class weights...')
        # self.train_class_weights = self.compute_class_weights()
        # print('Train class weights:', self.train_class_weights)

        self.train_class_weights = torch.FloatTensor([0.000000000000000000e+00,
                                                      3.508061818168880297e+00,
                                                      4.415242725535003743e+00,
                                                      1.929816058226905895e+01,
                                                      2.628740008695115193e+01,
                                                      1.212917345982307893e+01,
                                                      2.826658055253028934e+01,
                                                      2.148932725385034459e+01,
                                                      1.769486222014486643e+01,
                                                      1.991481374929695747e+01,
                                                      2.892054111644061365e+01,
                                                      6.634054658350238753e+01,
                                                      6.669804496207542854e+01,
                                                      3.332619576690268559e+01,
                                                      3.076747790368030167e+01,
                                                      6.492922584696864874e+01,
                                                      7.542849603844955197e+01,
                                                      7.551157920875556329e+01,
                                                      7.895305324715594963e+01,
                                                      7.385072181024294480e+01,
                                                      2.166310943989462956e+01])

        def get_instance_list(module, config, *args):
            return getattr(module, config['type'])(*args, **config['args'])

        transf_list_train = []
        transf_list_valid = []

        for transform_config in self.config['train_transform']:
            transf_list_train.append(
                get_instance_list(transform, transform_config))
            # TODO: Replace with logger
            print('train:', transform_config)
        for transform_config in self.config['valid_transform']:
            transf_list_valid.append(
                get_instance_list(transform, transform_config))
            # TODO: Replace with logger
            print('valid:', transform_config)

        transf_train = transforms.Compose(transf_list_train)
        transf_valid = transforms.Compose(transf_list_valid)

        if multi_gpu:
            dataloader_class = DataListLoader
        else:
            dataloader_class = GraphLevelDataLoader

        self.train_dataset = ScanNetLabelDataSet(self.config.train_root_dir, self.config.end_level, is_train=True,
                                            used_repeated_reconsts=self.config.train_use_repeated_reconsts,
                                            transform=transf_train,
                                            no_train_cropped=self.config.no_train_cropped, benchmark=False,
                                            num_crops_per_scene=self.config.num_crops_per_train_scene,
                                            index2allfilenames=self.index2allfilenames,
                                            max_num_scenes=self.config.max_num_train_scenes)
        print('train dataset len', len(self.train_dataset))
        self.train_loader = dataloader_class(self.train_dataset, batch_size=self.config.train_batch_size,
                                             shuffle=True, pin_memory=True,
                                             num_workers=self.config.num_workers)

        self.val_dataset = ScanNetLabelDataSet(self.config.val_root_dir, self.config.end_level, is_train=False,
                                          used_repeated_reconsts=self.config.val_use_repeated_reconsts,
                                          transform=transf_valid, benchmark=False,
                                          num_crops_per_scene=self.config.num_crops_per_val_scene,
                                          index2allfilenames=self.index2allfilenames,
                                          max_num_scenes=self.config.max_num_val_scenes)
        print('val dataset len', len(self.val_dataset))

        unit_test_compare_train_val(self.train_dataset, self.val_dataset)

        self.val_loader = dataloader_class(self.val_dataset, batch_size=self.config.test_batch_size, shuffle=False,
                                            pin_memory=True,
                                            num_workers=self.config.num_workers)

    def _load_all_scene_names(self):
        filenames = set()
        for file_path in [scannet_dataset.SCANNET_TRAIN_FILE, scannet_dataset.SCANNET_VAL_FILE,
                          scannet_dataset.SCANNET_TEST_FILE]:
            with open(file_path, 'r') as f:
                set_file_paths = f.read().splitlines()
            filenames.update(set_file_paths)
        return list(sorted(filenames))

    def compute_class_weights(self):
        class_count = self.count_train_labels()
        class_count[self.ignore_classes] = 0
        class_count = class_count.float()
        class_freq = 100.0 * class_count / class_count.sum()
        return torch.FloatTensor([-torch.log(x / 100.0) if x > 0 else 0 for x in class_freq])

    def count_train_labels(self):
        train_counter_dataset = ScanNetLabelDataSet(self.config.root_dir, self.config.end_level, is_train=True,
                                               no_train_cropped=True, benchmark=False,
                                               index2allfilenames=self.index2allfilenames,
                                               max_num_scenes=self.config.max_num_scenes, transform=None)
        train_counter_loader = GraphLevelDataLoader(train_counter_dataset, batch_size=self.config.train_batch_size,
                                                    shuffle=False,
                                                    pin_memory=False, num_workers=self.config.num_workers)

        total_label_count = torch.zeros(self.num_classes, dtype=torch.uint8)
        for data in train_counter_loader:
            count = torch.nn.functional.one_hot(data.y, num_classes=self.num_classes).type(torch.uint8).sum(axis=0)
            total_label_count = total_label_count + count

        return total_label_count

    def get_mesh(self, mesh_name):
        mesh_name = mesh_name.replace('.pt', '')
        mesh_rgb_path = f"{self.config.original_meshes_dir}/{mesh_name}/{mesh_name}_vh_clean_2.ply"
        return open3d.io.read_triangle_mesh(mesh_rgb_path)

    @property
    def class_names(self):
        return self.classes

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def ignore_classes(self) -> int:
        return 0

    @property
    def color_map(self):
        return torch.FloatTensor(
            [[255, 255, 255],  # unlabeled
             [174, 199, 232],  # wall
             [152, 223, 138],  # floor
             [31, 119, 180],  # cabinet
             [255, 187, 120],  # bed
             [188, 189, 34],  # chair
             [140, 86, 75],  # sofa
             [255, 152, 150],  # table
             [214, 39, 40],  # door
             [197, 176, 213],  # window
             [148, 103, 189],  # bookshelf
             [196, 156, 148],  # picture
             [23, 190, 207],  # counter
             [247, 182, 210],  # desk
             [219, 219, 141],  # curtain
             [255, 127, 14],  # refrigerator
             [158, 218, 229],  # shower curtain
             [44, 160, 44],  # toilet
             [112, 128, 144],  # sink
             [227, 119, 194],  # bathtub
             [82, 84, 163]])  # otherfurn

    pos_neg_map = torch.FloatTensor(
        [
            [200, 200, 200],
            [0, 255, 0],
            [255, 0, 0]])
