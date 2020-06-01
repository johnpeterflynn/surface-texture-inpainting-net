from collections import defaultdict
import glob
import random
import numpy as np
import torch
import open3d
from torch_geometric.data import DataListLoader
from torch_geometric.loader import DataLoader as GraphLevelDataLoader
from torchvision import transforms
import transform
from easydict import EasyDict
from datasets import scannet_dataset
from utils import data_utils, unit_tests


class ScanNetGraphColorDataSet(scannet_dataset.ScanNetDataset):
    def __init__(self, root_dir, mask_name, end_level, is_train, benchmark, index2allfilenames,# colors,
                 no_train_cropped=False, num_crops_per_scene=None, transform=None, max_num_scenes=None, enabled_mask_ids=None,
                 used_repeated_reconsts=True):
        super(ScanNetGraphColorDataSet, self).__init__(root_dir, end_level, is_train, benchmark, index2allfilenames,
                                                       no_train_cropped, num_crops_per_scene, transform, max_num_scenes,
                                                       used_repeated_reconsts)
        self._mask_name = mask_name
        self._enabled_mask_ids = enabled_mask_ids
        self.index2filenames, self.index2maskfilenames = self._load()
        self.index2filenames = np.asarray(self.index2filenames)

    def _load(self):
        set_file_paths = self.get_approved_filenames()

        if self._is_train:
            if self._no_train_cropped:
                dirs = [x for x in glob.glob(f"{self._root_dir}/masks/{self._mask_name}/*")
                        if x.split('/')[-1] in set_file_paths]
            else:
                existing_cropped_file_paths = defaultdict(list)
                for x in glob.glob(f"{self._root_dir}/masks/{self._mask_name}/*"):
                    scene_name = x.split('/')[-1].rsplit('_', 1)[0]
                    if scene_name in set_file_paths:
                        existing_cropped_file_paths[scene_name].append(x)

                dirs = self.select_rand_crops_per_scene(existing_cropped_file_paths)

        else:
            dirs = [x for x in glob.glob(f"{self._root_dir}/masks/{self._mask_name}/*")
                    if x.split('/')[-1] in set_file_paths]

        dirs = list(sorted(dirs))

        scene_filenames = []
        scene_to_mask_filenames = []
        for dir in dirs:
            scene_id = len(scene_filenames)
            scene_name = f"{dir.split('/')[-1]}"
            mask_filenames = [x.split('/')[-1] for x in glob.glob(f"{dir}/*.npz")]

            # If scene has masks and those masks belong to the list of self._enabled_mask_ids
            #  then register them as masks for training
            added = False
            if len(mask_filenames) > 0:
                mask_filenames = list(sorted(mask_filenames))
                for filename in mask_filenames:
                    mask_id = int(filename.split('.')[0])
                    if self._enabled_mask_ids is None or mask_id in self._enabled_mask_ids:
                        # If scene_to_mask_filenames has not been updated with a new dict yet
                        if len(scene_to_mask_filenames) <= len(scene_filenames):
                            scene_to_mask_filenames.append({})
                        scene_to_mask_filenames[scene_id][mask_id] = filename
                        added = True

                if added:
                    scene_filenames.append(scene_name)

        return scene_filenames, scene_to_mask_filenames

    def __len__(self):
        return len(self.index2filenames)

    @staticmethod
    def collate_fn(batch):
        return batch

    def __getitem__(self, file_color_index: int):
        file_index = file_color_index
        scene_name = self.index2filenames[file_index]
        mask_id, mask_file = random.choice(list(self.index2maskfilenames[file_index].items()))
        file_path = f"{self._root_dir}/graphs/{scene_name}.pt"
        mask_path = f"{self._root_dir}/masks/{self._mask_name}/{scene_name}/{mask_file}"
        saved_tensors = torch.load(file_path)

        # coords: 0:3 -> pos, 3:6 -> color, 6:9 -> normals, 9 -> indices
        coords = saved_tensors['vertices'][:self._end_level]
        #coords[0][:, 3:6] = color
        # Color normalization
        coords[0][:, 3:6] = coords[0][:, 3:6] * 2.0 - 1.0

        edges = saved_tensors['edges'][:self._end_level]

        if 'dilated_edges' in saved_tensors and 'dilation_dists' in saved_tensors:
            dilated_edges = saved_tensors['dilated_edges'][:self._end_level]
            dilation_dists = saved_tensors['dilation_dists']
        else:
            dilated_edges = None
            dilation_dists = None

        with open(mask_path, 'rb') as f:
            mask = np.load(f, allow_pickle=True)['vertex_mask']
        # NOTE: We accept masks which are 0 in ground truth regions and > 0 for inpainting regions
        #  where the value represents that node's distance from the nearest ground truth pixel.
        mask = torch.from_numpy(mask).unsqueeze(1)
        mask_bool = (mask == 0)

        # TODO: Normalize position, maybe normals
        sample = data_utils.HierarchicalData(
            x=torch.cat([coords[0][:, 3:6] * mask_bool, coords[0][:, 6:9], coords[0][:, :3], mask_bool], dim=-1),
            color=coords[0][:, 3:6],
            mask=mask,
            edge_index=edges[0].t().contiguous(),
            #labels=labels,
            name=scene_name,
            )

        # Full-mesh traces contain the trace back to the original mesh at position 0 while cropped traces don't.
        if self._is_train and not self._no_train_cropped:
            traces = saved_tensors['traces'][:self._end_level - 1]
        else:
            #sample.original_index_traces = saved_tensors['traces'][0]
            traces = saved_tensors['traces'][1:self._end_level]

        num_vertices = [coords[0].shape[0]]
        for level in range(1, len(edges)):
            setattr(sample, f"hierarchy_edge_index_{level}", edges[level].t().contiguous())
            # TODO: Don't load dilated edges that model won't use
            if dilated_edges is not None and dilated_edges[level] is not None:
                for i, dist in enumerate(dilation_dists):
                    if len(dilated_edges[level][i]) > 0:
                        setattr(sample, f"hierarchy_dil_{dist}_edge_index_{level}", dilated_edges[level][i].t().contiguous())
                    elif i == 0:
                        print('--ERROR: No dilation for', scene_name)
                    else:
                        print('--WARNING: {} has no dilation dist {} for level {}. Using dilation {}.'.format(scene_name, dist, level, dilation_dists[i-1]))
                        setattr(sample, f"hierarchy_dil_{dist}_edge_index_{level}", dilated_edges[level][i-1].t().contiguous())
            elif level == 2:
                print('--WARNING: No dilation for', scene_name, level, dilated_edges is None, (dilated_edges[level] is None) if dilated_edges is not None else False)

            setattr(sample, f"hierarchy_trace_index_{level}", traces[level - 1])
            num_vertices.append(traces[level - 1].max() + 1)
            #setattr(sample, f"pos_{level}", coords[level][:, :3])
            #sample.num_vertices.append(
            #    int(sample[f"hierarchy_trace_index_{level}"].max() + 1))
        sample.num_vertices = torch.tensor(num_vertices, dtype=torch.int)

        if self._transform:
            sample = self._transform(sample)

        return sample


class ScanNetGraphColorDataLoader:
    def __init__(self, config, multi_gpu):
        self.config = EasyDict(config)
        self.index2allfilenames = self._load_all_scene_names()

        train_mask_ids = np.arange(0, self.config.num_train_masks)
        val_mask_ids = np.arange(0, self.config.num_val_masks)

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

        self.train_dataset = ScanNetGraphColorDataSet(self.config.train_root_dir, self.config.mask_name,
                                                      self.config.end_level, is_train=True,
                                                      enabled_mask_ids=train_mask_ids,
                                                      used_repeated_reconsts=self.config.train_use_repeated_reconsts,
                                            transform=transf_train,
                                            no_train_cropped=self.config.no_train_cropped,
                                            num_crops_per_scene=self.config.num_crops_per_train_scene,
                                            benchmark=False,
                                            index2allfilenames=self.index2allfilenames,
                                            #colors=self.train_colors,
                                            max_num_scenes=self.config.max_num_train_scenes)

        print('train dataset len', len(self.train_dataset))
        self.train_loader = dataloader_class(self.train_dataset, batch_size=self.config.train_batch_size,
                                             shuffle=True, pin_memory=True,
                                             persistent_workers=self.config.num_workers > 0,
                                             num_workers=self.config.num_workers)

        self.val_dataset = ScanNetGraphColorDataSet(self.config.val_root_dir, self.config.mask_name,
                                                    self.config.end_level, is_train=False,
                                                    enabled_mask_ids=val_mask_ids,
                                                    used_repeated_reconsts=self.config.val_use_repeated_reconsts,
                                          transform=transf_valid, benchmark=False,
                                          no_train_cropped=self.config.no_train_cropped,
                                          num_crops_per_scene=self.config.num_crops_per_val_scene,
                                          index2allfilenames=self.index2allfilenames,
                                          #colors=self.val_colors,
                                          max_num_scenes=self.config.max_num_val_scenes)
        print('val dataset len', len(self.val_dataset))

        #unit_tests.compare_train_val(self.train_colors, self.val_colors)
        unit_tests.compare_train_val(self.train_dataset.index2filenames, self.val_dataset.index2filenames,
                                     train_cropped=not self.config.no_train_cropped)

        self.val_loader = dataloader_class(self.val_dataset, batch_size=self.config.test_batch_size, shuffle=False,
                                           pin_memory=True,
                                           persistent_workers=self.config.num_workers > 0,
                                           num_workers=self.config.num_workers)

    def _load_all_scene_names(self):
        filenames = set()
        for file_path in [scannet_dataset.SCANNET_TRAIN_FILE, scannet_dataset.SCANNET_VAL_FILE,
                          scannet_dataset.SCANNET_TEST_FILE]:
            with open(file_path, 'r') as f:
                set_file_paths = f.read().splitlines()
            filenames.update(set_file_paths)
        return list(sorted(filenames))

    def get_mesh(self, mesh_name):
        mesh_name = mesh_name.replace('.pt', '')
        mesh_rgb_path = f"{self.config.original_meshes_dir}/{mesh_name}/{mesh_name}_vh_clean_2.ply"
        return open3d.io.read_triangle_mesh(mesh_rgb_path)
