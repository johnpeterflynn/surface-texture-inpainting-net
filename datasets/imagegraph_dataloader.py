import os
import glob
import random
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import open3d
from skimage import io, img_as_float32
from scipy import ndimage
from torch_geometric.data import Data, DataListLoader
from torch_geometric.loader import DataLoader as GraphLevelDataLoader
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import transform
from typing import List
from easydict import EasyDict
from utils import math_utils, data_utils, unit_tests


class ImageGraphTextureDataSet(Dataset):
    def __init__(self, image_files, end_level, is_train, benchmark, img_size, crop_half_width, circle_radius, num_circles=4, max_items=None,
                 no_train_cropped=False, transform=None, random_mask=False):
        self._is_train = is_train
        self._benchmark = benchmark
        self.img_size = img_size
        self.crop_half_width = crop_half_width
        self._end_level = end_level
        self._transform = transform
        self._no_train_cropped = no_train_cropped
        self.image_files = np.array(image_files)
        self.random_mask = random_mask
        self.circle_radius = circle_radius
        self.num_circles = num_circles
        self.circle = torch.zeros((self.circle_radius * 2, self.circle_radius * 2, 1), dtype=torch.bool)
        for row in range(self.circle.shape[0]):
            for col in range(self.circle.shape[1]):
                if abs(row - self.circle_radius) ** 2 + abs(col - self.circle_radius) ** 2 <= self.circle_radius ** 2:
                    self.circle[row, col] = True

        self.traces_list = []
        self.edge_indices_list = []
        self.num_vertices_list = []
        self.decimation = 2
        # Build fake traces
        for level in range(self._end_level):
            level_img_size = self.img_size // (self.decimation ** level)
            num_verties = level_img_size ** 2
            self.num_vertices_list.append(num_verties)
            if level > 0:
                trace = np.arange(num_verties).reshape(level_img_size, level_img_size)
                trace = np.repeat(trace, self.decimation, axis=1).repeat(self.decimation, axis=0)
                trace = np.reshape(trace, (-1,))
                #trace = torch.from_numpy(trace)
                #trace = torch.cat((trace, trace + level_img_size * level_img_size), dim=0)
                print(level, 'Trace shape:', trace.shape)
                self.traces_list.append(trace)

        # Build fake decimated edges
        for level in range(self._end_level):
            level_img_size = self.img_size // (self.decimation ** level)
            edge_indices = self._generate_image_graph_edges(level_img_size)
            #edge_indices = torch.from_numpy(edge_indices)
            #edge_indices = torch.cat((edge_indices, edge_indices + level_img_size * level_img_size), dim=0)
            #edge_indices = edge_indices.t().contiguous()
            print(level, 'Number of edge indices:', edge_indices.shape)
            self.edge_indices_list.append(edge_indices)

    def _generate_image_graph_edges(self, img_size):
        def double_set_add(s, a, b):
            s.add((a, b))
            s.add((b, a))

        def get_neighbor_coords_list(r, c, max_size):
            coords_list = []
            # TODO: Should we add self-loops?
            # Maybe not since most graph algorithms explicitly include the vertex they're operating on
            #coords_list.append((r, c))
            if r > 0:
                coords_list.append((r - 1, c + 0))
                #if c > 0:
                #    coords_list.append((r - 1, c - 1))
                #if c < max_size - 1:
                #    coords_list.append((r - 1, c + 1))
            if c > 0:
                coords_list.append((r + 0, c - 1))
            if c < max_size - 1:
                coords_list.append((r + 0, c + 1))

            if r < max_size - 1:
                coords_list.append((r + 1, c + 0))
                #if c > 0:
                #    coords_list.append((r + 1, c - 1))
                #if c < max_size - 1:
                #    coords_list.append((r + 1, c + 1))
            return coords_list

        edge_indices = set()
        for r in range(img_size):
            for c in range(img_size):
                index = r * img_size + c
                neighbor_coords = get_neighbor_coords_list(r, c, img_size)
                for neighbor_coord in neighbor_coords:
                    neighbor_index = neighbor_coord[0] * img_size + neighbor_coord[1]
                    double_set_add(edge_indices, index, neighbor_index)

        edge_indices = np.asarray(list(edge_indices))
        return edge_indices

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index: int):
        img_path = self.image_files[index]
        img = io.imread(img_path)
        img = np.array(img)

        sample = {'color': img}

        if self._transform:
            sample = self._transform(sample)

        img = sample['color']

        # Create circular masks
        mask = torch.zeros((self.img_size, self.img_size, 1), dtype=torch.bool)
        for i in range(self.num_circles):
            if self._is_train and self.random_mask:
                x_offset = int((self.img_size / 2 - self.crop_half_width) * (random.random() * 2.0 - 1.0) * 0.95)
                y_offset = int((self.img_size / 2 - self.crop_half_width) * (random.random() * 2.0 - 1.0) * 0.95)
            else:
                x_offset = ((i % 2) * 2 - 1) * self.img_size // 4
                y_offset = ((i // 2) * 2 - 1) * self.img_size // 4

            row_start = self.img_size//2-self.circle_radius + x_offset
            row_end = self.img_size//2+self.circle_radius + x_offset
            col_start = self.img_size//2-self.circle_radius + y_offset
            col_end = self.img_size//2+self.circle_radius + y_offset
            mask[row_start:row_end, col_start:col_end] += self.circle

        img = torch.reshape(img, (-1, 3))
        mask = torch.reshape(mask, (-1, 1))

        sample = data_utils.HierarchicalData(x=torch.cat([img * ~mask, mask], dim=-1),
                                  color=img,
                                  mask=mask,
                                  edge_index=torch.from_numpy(self.edge_indices_list[0]).t().contiguous(),
                                  #num_vertices=self.num_vertices_list,
                                  )

        ##sample.num_vertices = torch.tensor(self.num_vertices_list)
        num_vertices = [sample.x.shape[0]]
        sample.num_vertices = torch.tensor(self.num_vertices_list, dtype=torch.int)
        for level in range(1, self._end_level):
            setattr(sample, f"hierarchy_edge_index_{level}", torch.from_numpy(self.edge_indices_list[level]).t().contiguous())
            setattr(sample, f"hierarchy_trace_index_{level}", torch.from_numpy(self.traces_list[level - 1]))
            num_vertices.append(int(sample[f"hierarchy_trace_index_{level}"].max() + 1))
        sample.num_vertices = torch.tensor(num_vertices, dtype=torch.int)

        return sample


class Normalize(object):
    """Normalize color images between [-1,1]."""

    def __call__(self, sample):
        color_image = sample['color']
        # NOTE: Don't normalize input_image. It's just a matrix of coordinates

        color_image = img_as_float32(color_image)
        color_image = (color_image * 2.0) - 1
        #color_image = color_image - 0.5

        return {'color': color_image}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, min_size, max_size):
        # For now size is defined as the smaller size of an image
        assert isinstance(min_size, int)
        assert isinstance(max_size, int)
        assert min_size <= max_size
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, sample):
        input_image = sample['color']

        h, w = input_image.shape[:2]

        output_size = np.random.randint(self.min_size, self.max_size + 1)

        if isinstance(output_size, int):
            if h > w:
                new_h, new_w = output_size * h / w, output_size
            else:
                new_h, new_w = output_size, output_size * w / h
        else:
            new_h, new_w = output_size

        new_h, new_w = int(new_h), int(new_w)

        # TODO: Use pillow library for resizing images
        # Nearest neighbor for input_image since we can't interpolate across discontinuities in uv coordinates
        #input_image = transform.resize(input_image, (new_h, new_w))
        #input_image = cv2.resize(input_image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        input_image = cv2.resize(input_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return {'color': input_image}


class CenterCrop(object):
    def __init__(self, crop_size):
        assert isinstance(crop_size, tuple)
        self.crop_size = crop_size

    def __call__(self, sample):
        input_image = sample['color']

        # Assuming input_image and color_image are the same shape
        h, w, _ = input_image.shape

        size_crop_h, size_crop_w = self.crop_size

        # Get a valid starting and end positions
        h_start = int((h - size_crop_h) / 2)
        w_start = int((w - size_crop_w) / 2)
        h_end = h_start + size_crop_h
        w_end = w_start + size_crop_w

        # Crop the input and target
        input_image = input_image[h_start:h_end, w_start:w_end, :]

        return {'color': input_image}


class RandomRotation(object):
    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, sample):
        input_image = sample['color']
        angle = random.choice(self.angles)
        input_image = ndimage.rotate(input_image, angle, reshape=False, mode='constant')
        return {'color': input_image}


class RandomFlip(object):
    def __init__(self, flip_axis):
        self.flip_axis = flip_axis

    def __call__(self, sample):
        input_image = sample['color']

        if np.random.choice(a=[False, True]):
            input_image = np.flip(input_image, axis=self.flip_axis).copy()

        return {'color': input_image}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        input_image = sample['color']

        # NOTE: Axis swapping is not necessary for uv coords since
        #  it is not an image, but rather a matrix of coordinates

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #input_image = input_image.transpose((2, 0, 1))
        return {'color': torch.from_numpy(input_image)}


class ImageGraphTextureDataLoader:
    def __init__(self, config, multi_gpu):
        self.config = EasyDict(config)

        self.train_files = self._load(os.path.join(self.config.root_dir, 'train'))
        self.val_files = self._load(os.path.join(self.config.root_dir, 'val'))
        len_train_files, len_val_files = len(self.train_files), len(self.val_files)
        total_num_files = len_train_files + len_val_files
        frac_train_files = len_train_files / total_num_files
        if 0 <= self.config.max_items <= total_num_files:
            max_train_files = int(self.config.max_items * frac_train_files)
            max_val_files = int(self.config.max_items * (1 - frac_train_files))
        else:
            max_train_files = int(total_num_files * frac_train_files)
            max_val_files = int(total_num_files * (1 - frac_train_files))
        self.train_files = self.train_files[:max_train_files]
        self.val_files = self.val_files[:max_val_files]

        transf_list_train = [
            Normalize(),
            Rescale(self.config.img_size, self.config.img_size),
            CenterCrop((self.config.img_size, self.config.img_size)),
        ]
        if self.config.random_augmentation:
            transf_list_train += [
                RandomRotation(),
                RandomFlip(flip_axis=1),
            ]
        transf_list_train.append(ToTensor())

        # Build val/test transformation
        transf_list_valid = [
            Normalize(),
            Rescale(self.config.img_size, self.config.img_size),
            CenterCrop((self.config.img_size, self.config.img_size)),
            #RandomFlip(flip_axis=1),
            ToTensor()
        ]

        transf_train = transforms.Compose(transf_list_train)
        transf_valid = transforms.Compose(transf_list_valid)

        if multi_gpu:
            dataloader_class = DataListLoader
        else:
            dataloader_class = GraphLevelDataLoader

        self.train_dataset = ImageGraphTextureDataSet(self.train_files, self.config.end_level, is_train=True,
                                            circle_radius=self.config.circle_radius,
                                            transform=transf_train, random_mask=self.config.random_mask,
                                            no_train_cropped=self.config.no_train_cropped, benchmark=False,
                                            img_size=self.config.img_size, max_items=self.config.max_items,
                                                    crop_half_width=self.config.crop_half_width)

        print('train dataset len', len(self.train_dataset))
        self.train_loader = dataloader_class(self.train_dataset, batch_size=self.config.train_batch_size,
                                             shuffle=True, pin_memory=True, persistent_workers=self.config.num_workers > 0,
                                             num_workers=self.config.num_workers)
        self.sample_train_loader = dataloader_class(self.train_dataset, batch_size=self.config.train_batch_size,
                                             shuffle=False, pin_memory=True,
                                             num_workers=self.config.num_workers)
        self.sample_train_dataset = torch.utils.data.Subset(self.train_dataset,
                                                            np.arange(min(self.config.num_static_samples,
                                                                          len(self.train_dataset))))
        self.sample_train_loader = dataloader_class(self.sample_train_dataset, batch_size=self.config.train_batch_size,
                                             shuffle=False, pin_memory=True,
                                             num_workers=self.config.num_workers)

        # TODO: Update val dataset so that it doesn't have to be treated like a train dataset
        #  includes is_train=False and no_train_cropped=self.config.no_train_cropped
        self.val_dataset = ImageGraphTextureDataSet(self.val_files, self.config.end_level, is_train=False,
                                          circle_radius=self.config.circle_radius,
                                          transform=transf_valid, benchmark=False,
                                          no_train_cropped=self.config.no_train_cropped,
                                          img_size=self.config.img_size, max_items=self.config.max_items,
                                                    crop_half_width=self.config.crop_half_width)
        print('val dataset len', len(self.val_dataset))

        #unit_tests.compare_train_val(self.train_colors, self.val_colors)

        self.val_loader = dataloader_class(self.val_dataset, batch_size=self.config.test_batch_size, shuffle=False,
                                            pin_memory=True, persistent_workers=self.config.num_workers > 0,
                                            num_workers=self.config.num_workers)
        self.sample_val_dataset = torch.utils.data.Subset(self.val_dataset,
                                                          np.arange(min(self.config.num_static_samples,
                                                                        len(self.val_dataset))))
        self.sample_val_loader = dataloader_class(self.sample_val_dataset, batch_size=self.config.test_batch_size,
                                             shuffle=False, pin_memory=True,
                                             num_workers=self.config.num_workers)

    def _load(self, root_dir, seed=42) -> List[str]:
        filenames = glob.glob(f"{root_dir}/*.png")
        filenames = sorted(filenames)
        random.Random(seed).shuffle(filenames)
        return filenames