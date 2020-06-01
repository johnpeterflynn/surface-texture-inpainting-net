import os
import glob
from abc import abstractmethod

import torch
import numpy as np
from torch.utils.data import DataLoader
from base import BaseDataLoader


class SceneLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, scene_args, scene_defaults, val_scene_ids, test_scene_ids,
                 train_display_indices=[], val_display_indices=[], num_workers=1):

        self.data_dir = data_dir
        self.train_display_indices = train_display_indices
        self.val_display_indices = val_display_indices

        train_datasets = []
        val_datasets = []
        test_datasets = []
        for scene_id, params in scene_args.items():
            is_train_set = scene_id not in val_scene_ids and scene_id not in test_scene_ids

            # Use default params if not explicitly specified
            all_params = scene_defaults.copy()
            all_params.update(params)

            dataset = self.create_dataset(self.data_dir, scene_id, **all_params, training=is_train_set)

            if is_train_set:
                train_datasets.append(dataset)
            elif scene_id in val_scene_ids:
                val_datasets.append(dataset)
            else:
                test_datasets.append(dataset)

        # Concatenate partitioned datasets
        self.train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        self.val_dataset = torch.utils.data.ConcatDataset(val_datasets)
        self.test_dataset = None#torch.utils.data.ConcatDataset(test_datasets)

        print('Train dataset length: ', len(self.train_dataset))
        print('Val dataset length: ', len(self.val_dataset))
        #print('Test dataset length: ', len(self.test_dataset))

        self.collate_fn = dataset.collate_fn if callable(getattr(dataset, "collate_fn", None)) else None

        super().__init__(self.train_dataset, batch_size, shuffle, num_workers, collate_fn=self.collate_fn, pin_memory=True)
        pass

    @abstractmethod
    def create_dataset(self, data_dir, scene_id, training):
        pass

    def split_validation(self):
        batch_size = self.init_kwargs['batch_size']
        num_workers = self.init_kwargs['num_workers']
        return DataLoader(self.val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=self.collate_fn, pin_memory=True)

    def split_train_sampler(self):
        self.train_display_indices = self.format_display_indices(self.train_display_indices, self.train_dataset)
        subset = torch.utils.data.Subset(self.train_dataset, self.train_display_indices)
        batch_size = self.init_kwargs['batch_size']
        num_workers = self.init_kwargs['num_workers']
        return DataLoader(subset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=self.collate_fn, pin_memory=True)

    def split_validation_sampler(self):
        self.val_display_indices = self.format_display_indices(self.val_display_indices, self.val_dataset)
        subset = torch.utils.data.Subset(self.val_dataset, self.val_display_indices)
        batch_size = self.init_kwargs['batch_size']
        num_workers = self.init_kwargs['num_workers']
        return DataLoader(subset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=self.collate_fn, pin_memory=True)

    def format_display_indices(self, display_indices, dataset):
        if isinstance(display_indices, int):
            return np.linspace(0, len(dataset), num=display_indices, endpoint=False, dtype=np.int)

    def split_test(self):
        batch_size = self.init_kwargs['batch_size']
        num_workers = self.init_kwargs['num_workers']
        return DataLoader(self.test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=self.collate_fn, pin_memory=True)



