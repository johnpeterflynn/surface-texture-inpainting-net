import io
import os
import time
import random

import argparse
from argparse import RawTextHelpFormatter
import configparser
import torch
import glob
import json
import heapq
import open3d as o3d
import matplotlib.pyplot as plt
from termcolor import colored

#from pytorch3d.utils import ico_sphere
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm

# Util function for loading meshes
#from pytorch3d.io import load_objs_as_meshes, save_obj, load_ply, IO
#from pytorch3d.renderer.cameras import PerspectiveCameras, FoVPerspectiveCameras

#from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments

# Data structures and functions for rendering
#from pytorch3d.structures import Meshes
#from pytorch3d.renderer import (
#    FoVPerspectiveCameras,
#    #PerspectiveCameras,
#    Materials,
#    RasterizationSettings,
#    MeshRasterizer,
#    SoftPhongShader,
#)

#from pytorch3d.renderer import TexturesAtlas, TexturesUV
#from pytorch3d.io.obj_io import load_obj

# add path for demo utils functions
import sys
import os
sys.path.append(os.path.abspath(''))
from utils.plot_image_grid import image_grid
sys.path.append(".")
from utils.pretty_print import pretty_print_arguments
from utils.clear_folder import clear_folder
from sklearn.neighbors import BallTree


MESH_FILENAME = "{}_vh_clean_2.{}"
MIN_FRAC_MASKED_VERTS = 0.02


def load_camera_poses(path, max_num_poses=None, cpp_sens_reader=True):
    def read_camera_pose(filename):
        loaded_extrinsic = np.loadtxt(filename, dtype=np.float64)
        extrinsic = np.linalg.inv(loaded_extrinsic)
        R = np.transpose(extrinsic[:3, :3])
        T = extrinsic[:3, 3]
        return R, T

    if cpp_sens_reader:
        poses_path = os.path.join(path, '*.pose.txt')
    else:
        poses_path = os.path.join(path, '*.txt')
    filenames = glob.glob(poses_path)
    filenames.sort()
    if max_num_poses is not None:
        filenames = filenames[:max_num_poses]

    poses = [read_camera_pose(filename) for filename in filenames]
    R = torch.cat([torch.from_numpy(p[0]).unsqueeze(0) for p in poses], dim=0)
    T = torch.cat([torch.from_numpy(p[1]).unsqueeze(0) for p in poses], dim=0)
    pose_ids = np.arange(0, len(poses))

    return R, T, pose_ids


def load_scan_config(path, scan_name, cpp_sens_reader=True):
    """Handle importing ScanNet config from either c++ or python SensReader"""
    if cpp_sens_reader:
        filepath = os.path.join(path, '_info.txt')
    else:
        filepath = os.path.join(path, '{}.txt'.format(scan_name))

        intrinsic = np.loadtxt(os.path.join(path, 'intrinsic_color.txt'), dtype=np.float32)
        intrinsic = torch.from_numpy(intrinsic)

    # configparser only reads config files containing headers so we apply a dummy header as a workaround
    with open(filepath, 'r') as f:
        config_string = '[dummy_section]\n' + f.read()
    buf = io.StringIO(config_string)
    config = configparser.ConfigParser()
    config.read_file(buf)
    config = dict(config['dummy_section'])

    if cpp_sens_reader:
        intrinsic = np.array([float(i) for i in config['m_calibrationcolorintrinsic'].split()]).reshape(4, 4)
        config_out = {
            'colorheight': int(config['m_colorheight']),
            'colorwidth': int(config['m_colorwidth']),
            'colorintrinsic': intrinsic
        }
    else:
        config_out = {
            'colorheight': int(config['colorheight']),
            'colorwidth': int(config['colorwidth']),
            'colorintrinsic': intrinsic
        }

    return config_out


def compute_projection_matrix(intrinsic, camera_height, camera_width):
    near, far = 1.0, 100.0
    z_sign = -1.0
    max_x = near * camera_width / (2.0 * intrinsic[0, 0])
    max_y = near * camera_height / (2.0 * intrinsic[1, 1])
    min_x = -max_x
    min_y = -max_y

    # Build transform directly from camera coordinates to NDC
    K = torch.zeros((4, 4), dtype=torch.float32)
    K[0, 0] = 2.0 * near / (max_x - min_x)
    K[1, 1] = 2.0 * near / (max_y - min_y)
    K[2, 2] = z_sign * (far + near) / (far - near)
    K[3, 2] = z_sign
    K[2, 3] = -2.0 * far * near / (far - near)

    return K


def observed_faces_to_observed_vertices(observed_faces, mesh_faces, batch_size):
    observed_verts_list = []
    for b in range(batch_size):
        single_observed_faces = observed_faces[b]

        # Ignore faceless pixels (face id = -1)
        single_observed_faces = single_observed_faces[single_observed_faces >= 0]

        single_observed_faces = torch.unique(single_observed_faces)

        # Undo stacking of face ids within batch
        single_observed_faces = single_observed_faces - b * mesh_faces.shape[0]

        observed_verts = mesh_faces[single_observed_faces]

        # Observed faces to observed vertex IDs (# pixels * # verts per face)
        observed_verts = torch.reshape(observed_verts, (-1,))
        observed_verts = torch.unique(observed_verts)
        observed_verts_list.append(observed_verts)

    return observed_verts_list


def compute_observed_vertex_map(device, root_path, sens_path, scene_name, batch_size=50, max_num_poses=None, is_obj=False):
    # Setup
    #if torch.cuda.is_available():
    #    device = torch.device("cuda:0")
    #    torch.cuda.set_device(device)
    #else:
    #    device = torch.device("cpu")

    # Load obj file
    mesh_filename = MESH_FILENAME.format(scene_name, 'obj' if is_obj else 'ply')
    mesh_path = os.path.join(root_path, mesh_filename)
    if is_obj:
        mesh = load_objs_as_meshes([mesh_path], device=device)#, load_textures=False)
    else:
        mesh = IO().load_mesh(path=mesh_path, device=device)

    # All Faces to Vertex IDs of mesh (# faces in mesh, # verts per face)
    mesh_faces = mesh.faces_list()[0]

    num_verts = mesh.verts_list()[0].shape[0]
    global_observed_vert_pose_lists = [[] for i in range(num_verts)]
    print('Mesh contains', num_verts, 'vertices')

    scan_config = load_scan_config(sens_path, scene_name)
    camera_height = scan_config['colorheight']
    camera_width = scan_config['colorwidth']
    intrinsic = scan_config['colorintrinsic']
    intrinsic = torch.from_numpy(intrinsic)

    # The different viewpoints from which we want to render the mesh.
    R, T, pose_ids = load_camera_poses(sens_path, max_num_poses)
    num_poses = len(pose_ids)
    print('Loaded', num_poses, 'camera poses')

    num_batches = int(np.ceil(num_poses / batch_size))

    K = compute_projection_matrix(intrinsic, camera_height, camera_width)

    raster_settings = RasterizationSettings(
        image_size=(256, 256),
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Create a Phong renderer by composing a rasterizer and a shader. The textured
    # Phong shader will interpolate the texture uv coordinates for each vertex,
    # sample from a texture image and apply the Phong lighting model
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device
        )
    )

    meshes = mesh.extend(batch_size)

    with torch.no_grad():
        print('Executing', num_batches, 'batches of size', batch_size)
        for batch_idx in range(num_batches):
            # Adjust for last batch if it is not full-sized
            last_batch_size = num_poses - (num_batches - 1) * batch_size
            if batch_idx == num_batches - 1 and last_batch_size != batch_size:
                current_batch_size = last_batch_size
                meshes = mesh.extend(current_batch_size)
            else:
                current_batch_size = batch_size
            R_batch = R[(batch_size * batch_idx):(batch_size * batch_idx + current_batch_size)]
            T_batch = T[(batch_size * batch_idx):(batch_size * batch_idx + current_batch_size)]

            #print('Batch', batch_idx, 'poses', (batch_size * batch_idx),
            #      'to', (batch_size * batch_idx + current_batch_size))

            cur_K = K.unsqueeze(0).expand(current_batch_size, -1, -1)
            cameras = PerspectiveCameras(device=device, R=R_batch, T=T_batch, K=cur_K)
            target_images, target_fragments = renderer(meshes, cameras=cameras)

            #image_grid(target_images.cpu().numpy(), rows=4, cols=5, rgb=True)
            #plt.show()
            #input()

            # Pixels to Face IDs per sample (N, # pixels)
            observed_faces = target_fragments.pix_to_face
            observed_verts_list = observed_faces_to_observed_vertices(observed_faces, mesh_faces, current_batch_size)
            observed_verts_list = [lis.cpu().numpy() for lis in observed_verts_list]
            pose_ids_batch = pose_ids[(batch_size * batch_idx):(batch_size * batch_idx + current_batch_size)]
            for i, observed_verts in enumerate(observed_verts_list):
                pose_id = pose_ids_batch[i].item()
                for vertex_id in observed_verts:
                    global_observed_vert_pose_lists[vertex_id].append(pose_id)

    return global_observed_vert_pose_lists, pose_ids


def select_pose_random_subset(pose_ids, keep_probability=1.0):
    random_mask = (np.random.rand(len(pose_ids)) <= keep_probability)
    return pose_ids[random_mask]


def generate_mask_from_vertex_observing_poses(global_observed_vert_pose_lists, visible_pose_ids, min_num_poses):
    global_observed_verts = np.zeros(len(global_observed_vert_pose_lists), dtype=np.int)
    visible_pose_ids_set = set(visible_pose_ids)
    for i, vert_pose_ids in enumerate(global_observed_vert_pose_lists):
        # Keep if visible_pose_ids.shape[0] > 0 to protect against =0?
        vert_visible_pose_ids = [p for p in vert_pose_ids if p in visible_pose_ids_set]
        global_observed_verts[i] = 1 if len(vert_visible_pose_ids) >= min_num_poses else 0

    return global_observed_verts


def load_o3d_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path, True)

    print("Try to render a mesh with colors (exist: " +
          str(mesh.has_vertex_colors()) + ") and uvs (exist: " +
          str(mesh.has_triangle_uvs()) + ")")

    if not mesh.is_edge_manifold():
        print('WARNING: Mesh is not an edge manifold')

    if not mesh.is_vertex_manifold():
        print('WARNING: Mesh is not a vertex manifold')

    #if mesh.is_self_intersecting():
    #    print('WARNING: Mesh is self-intersecting')

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    return mesh


class MaskVisualizer:
    """Visualize vertex masks from various datasets with open3D. Key Events show RGB, ground truth, prediction or differences."""

    def __init__(self, select_pose_random_subset, generate_mask_from_vertex_observing_poses, keep_probability=0.5, prob_increment=0.05, min_num_poses=1, pose_increment=5):
        """Initialize Mask Visualizer

        Arguments:
            compute_mask_func  {method} -- Method to compute the observed vertex mask
        """
        #self._compute_mask_func = compute_mask_func
        self._select_pose_random_subset = select_pose_random_subset
        self._generate_mask_from_vertex_observing_poses = generate_mask_from_vertex_observing_poses
        self._rgb_colors = None
        self._mask_colors = None
        self._keep_probability = keep_probability
        self._prob_increment = prob_increment
        self._min_num_poses = min_num_poses
        self._pose_increment = pose_increment
        self._visible_pose_ids = None

    def visualize_result(self, mesh_filepath, observed_poses_per_vert, valid_pose_ids):
        mesh = load_o3d_mesh(mesh_filepath)
        #mesh.textures = []

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(width=1600, height=1200)

        # PREPARE RGB COLOR SWITCH
        self._rgb_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))

        def colorize_rgb(visu):
            mesh.vertex_colors = self._rgb_colors
            visu.update_geometry(mesh)
            visu.update_renderer()

        vis.register_key_callback(ord('H'), colorize_rgb)

        # PREPARE MASK COLOR SWITCH
        self._mask_colors = None

        def generate_mask():
            print('Generating new mask with min num poses', self._min_num_poses, 'keep probability',
                  self._keep_probability)

            observed_verts = self._generate_mask_from_vertex_observing_poses(observed_poses_per_vert, self._visible_pose_ids, self._min_num_poses)
            #observed_verts = self._compute_mask_func(observed_poses_per_vert, valid_pose_ids, self._keep_probability,
            #                                         self._min_num_poses)

            colors = np.zeros_like(np.asarray(mesh.vertex_colors))
            print('Num observed verts:', (observed_verts > 0).sum())
            colors[observed_verts > 0, 0] = 1.0
            colors = o3d.utility.Vector3dVector(np.asarray(colors))

            return colors

        def colorize_mask(visu):
            if self._mask_colors is None:
                self._visible_pose_ids = self._select_pose_random_subset(valid_pose_ids, self._keep_probability)
                self._mask_colors = generate_mask()
            mesh.vertex_colors = self._mask_colors
            visu.update_geometry(mesh)
            visu.update_renderer()

        vis.register_key_callback(ord('J'), colorize_mask)

        def swap_new_mask(visu):
            self._visible_pose_ids = self._select_pose_random_subset(valid_pose_ids, self._keep_probability)
            self._mask_colors = generate_mask()
            colorize_mask(visu)

        vis.register_key_callback(ord('G'), swap_new_mask)

        def swap_new_mask_down(visu):
            self._keep_probability = max(self._keep_probability - self._prob_increment, 0.0)
            self._visible_pose_ids = self._select_pose_random_subset(valid_pose_ids, self._keep_probability)
            self._mask_colors = generate_mask()
            colorize_mask(visu)

        vis.register_key_callback(ord('B'), swap_new_mask_down)

        def swap_new_mask_up(visu):
            self._keep_probability = min(self._keep_probability + self._prob_increment, 1.0)
            self._visible_pose_ids = self._select_pose_random_subset(valid_pose_ids, self._keep_probability)
            self._mask_colors = generate_mask()
            colorize_mask(visu)

        vis.register_key_callback(ord('T'), swap_new_mask_up)

        def swap_new_mask_posenum_down(visu):
            self._min_num_poses = max(self._min_num_poses - self._pose_increment, 0)
            self._mask_colors = generate_mask()
            colorize_mask(visu)

        vis.register_key_callback(ord('V'), swap_new_mask_posenum_down)

        def swap_new_mask_posenum_up(visu):
            self._min_num_poses = max(self._min_num_poses + self._pose_increment, 0)
            self._mask_colors = generate_mask()
            colorize_mask(visu)

        vis.register_key_callback(ord('R'), swap_new_mask_posenum_up)

        vis.add_geometry(mesh)
        vis.get_render_option().light_on = True
        vis.run()
        vis.destroy_window()


class MaskCircleVisualizer:
    """Visualize vertex masks from various datasets with open3D. Key Events show RGB, ground truth, prediction or differences."""

    def __init__(self):
        """Initialize Mask Visualizer

        Arguments:
            compute_mask_func  {method} -- Method to compute the observed vertex mask
        """
        self._rgb_colors = None
        self._mask_colors = None

    def visualize_result(self, mesh_filepath, mask):
        mesh = load_o3d_mesh(mesh_filepath)
        #mesh.textures = []

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(width=1600, height=1200)

        # PREPARE RGB COLOR SWITCH
        self._rgb_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))

        def colorize_rgb(visu):
            mesh.vertex_colors = self._rgb_colors
            visu.update_geometry(mesh)
            visu.update_renderer()

        vis.register_key_callback(ord('H'), colorize_rgb)

        # PREPARE MASK COLOR SWITCH
        self._mask_colors = None

        def generate_mask():
            colors = np.zeros_like(np.asarray(mesh.vertex_colors))
            print('Num masked:', (~mask).sum())
            colors[mask > 0, 0] = 1.0
            colors = o3d.utility.Vector3dVector(np.asarray(colors))

            return colors

        def colorize_mask(visu):
            self._mask_colors = generate_mask()
            mesh.vertex_colors = self._mask_colors
            visu.update_geometry(mesh)
            visu.update_renderer()

        vis.register_key_callback(ord('J'), colorize_mask)

        vis.add_geometry(mesh)
        vis.get_render_option().light_on = True
        vis.run()
        vis.destroy_window()


def plot_vertices_per_observer(observed_poses_per_vert, num_observers):
    vertices_per_observer = [0] * num_observers
    for pose_ids in observed_poses_per_vert:
        for p_idx in pose_ids:
            vertices_per_observer[p_idx] += 1

    x = np.arange(num_observers)
    vertices_per_observer = np.array(vertices_per_observer)
    plt.figure()
    plt.plot(x, vertices_per_observer)
    plt.title('Number of Vertices Observed vs Observer ID (temporal)')
    plt.xlabel('Observer IDs')
    plt.ylabel('# Vertices')
    #plt.show()


def plot_observer_dist(observed_poses_per_vert):
    num_observers_per_vert = [len(pose_ids) for pose_ids in observed_poses_per_vert]
    plt.figure()
    plt.hist(num_observers_per_vert, color='blue', edgecolor='black', bins=100)
    plt.title('Histogram of Number of Camera Observations Per Vertex')
    plt.xlabel('# Camera Observers')
    plt.ylabel('# Vertices')
    # plt.show()


def plot_statistics(observed_poses_per_vert, num_observers):
    plot_vertices_per_observer(observed_poses_per_vert, num_observers)
    plot_observer_dist(observed_poses_per_vert)
    plt.show()


def process_frame_observers(scan_name, device, args):
    scan_root_path = os.path.join(args.in_path, scan_name)
    scan_sens_path = os.path.join(args.in_sens_path, scan_name)
    cached_obs_per_vert_path = os.path.join(args.out_path, 'observers_per_vert',  '{}.npz'.format(scan_name))
    if not args.no_load_observers and os.path.isfile(cached_obs_per_vert_path):
        print('Loading vertex observer data for', scan_name, 'at', cached_obs_per_vert_path)
        with open(cached_obs_per_vert_path, 'rb') as f:
            data = np.load(f, allow_pickle=True)['data'].item()
        observed_poses_per_vert = data['observed_poses_per_vert']
        valid_pose_ids = data['valid_pose_ids']
    else:
        print('no cached observer data for {}.'.format(scan_name), 'Computing from mesh at', scan_root_path)
        t1 = time.time()
        observed_poses_per_vert, valid_pose_ids = compute_observed_vertex_map(device, scan_root_path, scan_sens_path, scan_name, 2)
        print('Total processing time for {}:'.format(scan_name), time.time()-t1)

        data = {}
        data['observed_poses_per_vert'] = observed_poses_per_vert
        data['valid_pose_ids'] = valid_pose_ids
        Path(args.out_path).mkdir(parents=True, exist_ok=True)
        with open(cached_obs_per_vert_path, 'wb') as f:
            np.savez_compressed(f, data=data)

    visible_pose_ids = select_pose_random_subset(valid_pose_ids, 0.5)
    mask = generate_mask_from_vertex_observing_poses(observed_poses_per_vert, visible_pose_ids, 40)

    if args.display:
        # TODO: Num observers should be the original number of camera poses
        #plot_statistics(observed_poses_per_vert, np.max(data['valid_pose_ids']) + 1)

        # Callback method to generate a vertex mask from vertex observers
        #def compute_observed_verts(observed_poses_per_vert, valid_pose_ids, keep_probability, min_num_poses):
        #    visible_pose_ids = select_pose_random_subset(valid_pose_ids, keep_probability)
        #    observed_verts = generate_mask_from_vertex_observing_poses(observed_poses_per_vert, visible_pose_ids, min_num_poses)
        #    return observed_verts

        print('Opening vertex mask visualizer')
        visualizer = MaskVisualizer(select_pose_random_subset, generate_mask_from_vertex_observing_poses)
        mesh_path = os.path.join(args.in_path, scan_name, MESH_FILENAME.format(scan_name, 'ply'))
        visualizer.visualize_result(mesh_path, observed_poses_per_vert, valid_pose_ids)

    return mask


def process_frame_circles(scan_name, device, args):
    mesh_filepath = os.path.join(args.in_path, scan_name, MESH_FILENAME.format(scan_name, 'ply'))
    mesh = load_o3d_mesh(mesh_filepath)
    random.seed(10 * args.number)

    print('Uniformly sampling mask points')
    #print('Surface area:', mesh.get_surface_area())
    vertices = np.asarray(mesh.vertices)
    masks = []

    for mask_num in range(args.masks_per_scene):
        task_name = '{} [{}/{}]'.format(scan_name, mask_num + 1, args.masks_per_scene)
        total_num_sampled_points = 0
        mask = np.zeros(vertices.shape[0], dtype=int)

        # It's hard to know how many sampled points is enough to account for 'args.frac_masked_vertices'
        # of the total vertices, so we iterate by predicting a few at a time until we reach the threshold
        sample_num_points = 10
        finish = False
        while not finish:

            print(task_name, 'sampling', sample_num_points, 'points')
            # TODO: Sample using poisson disk
            #pcd = mesh.sample_points_uniformly(number_of_points=100)
            ##pcd = mesh.sample_points_poisson_disk(number_of_points=100)
            #pcd = np.asarray(pcd.points)

            ## Unfortunately sample_points_uniformly returns points and not indices into mesh, so as a quick fix we use
            ## NN to find the indices
            #ball_tree = BallTree(vertices)
            #_, ind = ball_tree.query(pcd, k=1)
            #ind = ind.flatten()
            ind = random.sample(list(np.arange(0, len(mesh.vertices))), sample_num_points)

            if not mesh.has_adjacency_list():
                print(task_name, 'Mesh has no adjacency list. Computing one...')
                mesh.compute_adjacency_list()
            adjacency_list = mesh.adjacency_list

            print(task_name, 'Computing masked neighborhoods')
            for index in ind:
                seen = set()
                visited = []
                heapq.heappush(visited, (0, index))
                while len(visited) > 0:
                    next_dist, next_idx = heapq.heappop(visited)
                    seen.add(next_idx)
                    # Mask value is node dist (in edges) from nearest *unmasked* node. (args.radius - next_dist) is
                    #  the distance from the node to the edge of the circle. We use the maximum distance-to-circle
                    #  value assigned to each node since that circle covers the largest area around the node.
                    mask[next_idx] = max(args.radius - next_dist, mask[next_idx])
                    if next_dist <= args.radius - 1:
                        for neighbor_idx in adjacency_list[next_idx]:
                            if neighbor_idx not in seen:
                                heapq.heappush(visited, (next_dist + 1, neighbor_idx))
            total_num_sampled_points += sample_num_points
            num_masked_vertices = (mask > 0).sum()
            curent_frac_masked_verts = num_masked_vertices / len(mesh.vertices)
            sample_num_points = int(total_num_sampled_points * (args.frac_masked_vertices / curent_frac_masked_verts - 1))
            finish = num_masked_vertices / len(mesh.vertices) >= args.frac_masked_vertices or sample_num_points <= 0
            print(task_name,
                  'Num masked vertices:', num_masked_vertices,
                  'Total vertices:', len(mesh.vertices),
                  'frac: {:.2f}'.format(num_masked_vertices / len(mesh.vertices)))

        masks.append(mask)


        if args.display:
            visualizer = MaskCircleVisualizer()
            mesh_path = os.path.join(args.in_path, scan_name, MESH_FILENAME.format(scan_name, 'ply'))
            visualizer.visualize_result(mesh_path, (mask == 0))

    return masks

def process_frame(scan_name, device, args):
    if args.mask_type == 'observers':
        vertex_masks = process_frame_observers(scan_name, device, args)
    elif args.mask_type == 'circles':
        vertex_masks = process_frame_circles(scan_name, device, args)
    else:
        raise NotImplementedError("Processing now implemented for mask type", args.mask_type)

    approve_and_write_out_mask(scan_name, vertex_masks, args)


def approve_and_write_out_mask(scan_name, vertex_masks, args):
    for type in ['graph_levels', 'cropped']:
        for mode in ['train', 'val']:
            mode_dir = os.path.join('data', 'generated', type, args.preprocess_name, mode)
            scene_dir = os.path.join(mode_dir, 'graphs')
            filenames = [x.split('/')[-1] for x in glob.glob(f"{scene_dir}/*.pt")
                         if scan_name in x.split('/')[-1]]

            for filename in filenames:
                file_path = os.path.join(scene_dir, filename)
                print('Generating mask files for', file_path)
                saved_tensors = torch.load(file_path)
                vertex_indices = saved_tensors['vertices'][0][:, -1].numpy()

                # Correcting for what was a bit of a hack. Indices are passed along with other
                #  vertex properties and are therefore converted into floats. Rounding to the
                #  nearest integer is our best effort at precenting an error.
                vertex_indices = np.round(vertex_indices).astype(int)

                for mask_num, vertex_mask in enumerate(vertex_masks):
                    vertex_mask_out = vertex_mask[vertex_indices]

                    # Test if enough vertices exist
                    num_masked_unmasked_verts = np.bincount((vertex_mask_out == 0).astype(int))
                    frac_masked_verts = num_masked_unmasked_verts[0] / num_masked_unmasked_verts.sum()
                    if frac_masked_verts < MIN_FRAC_MASKED_VERTS:
                        print(colored('Warning: {:.3f}% masked vertices is too few for crop -> reject'.format(frac_masked_verts * 100), 'red'))
                        continue

                    filename_no_ext = filename.replace('.pt', '')
                    masks_directory = os.path.join(mode_dir, 'masks', args.mask_name, filename_no_ext)
                    mask_filepath = os.path.join(masks_directory, '{:06d}.npz'.format(mask_num))
                    Path(masks_directory).mkdir(parents=True, exist_ok=True)
                    with open(mask_filepath, 'wb') as f:
                        np.savez_compressed(f, vertex_mask=vertex_mask_out)

    print('Finished', args.number)


def write_args(args):
    with open(os.path.join(args.out_path, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def main(args):
    mapping = dict()

    # No individual concept of train/val/test necessary here
    considered_rooms_paths = ['datasets/meta/scannet/scannetv2_train.txt',
                              'datasets/meta/scannet/scannetv2_val.txt',
                              'datasets/meta/scannet/scannetv2_test.txt']
    considered_rooms = []
    for path in considered_rooms_paths:
        with open(path, 'r') as f:
            considered_rooms += f.read().splitlines()

    file_paths = sorted([x for
                         x in glob.glob(f"{args.in_path}/*/*.ply")
                         if 'clean_2.ply' in x
                         and x.split('/')[-1].rsplit('_', 3)[0] in considered_rooms])

    scan_path = file_paths[args.number]
    scan_name = scan_path.split('/')[-1].rsplit('_', 3)[0]

    if torch.cuda.device_count() > 1:
        if args.number % 2 == 0:
            device = 'cuda:0'
        else:
            device = 'cuda:1'
    else:
        device = 'cuda:0'

    process_frame(scan_name, device, args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                     description="Create mask labels for each vertex in a mesh based on which poses observe each vertex")

    parser.add_argument('--in_path', type=str, required=True,
                        help='path to the root directory of folders containing the trimeshes')

    parser.add_argument('--out_path', type=str, required=True,
                        help='path to the directory where the per-vertex observer data and vertex masks should be stored')

    parser.add_argument('--mask_name', type=str, required=True,
                        help='name of the masks to be generated')

    parser.add_argument('--number', const=-1, default=-1, type=int, nargs='?',
                        help='number of task id (used for parallization)')

    parser.add_argument('--preprocess_name', type=str, required=False,
                        help='name given to the generated collection of graph_levels and cropped files')

    parser.add_argument('--display', dest='display', action='store_true',
                        help='True if Open3D instance showing mask on the mesh should be loaded')
    parser.set_defaults(display=False)

    subparsers = parser.add_subparsers(title='mask type', dest="mask_type",
                                       help='select the type of mask to process, e.g. [observers, circles]')

    parser_observer = subparsers.add_parser('observers', help='a help')
    parser_observer.add_argument('--in_sens_path', type=str, required=True,
                                 help='path to the root directory of folders containing uncompressed raw sensor data')

    parser_observer.add_argument('--no_load_observers', dest='no_load_observers', action='store_true',
                                 help='True if per-vertex observer data should be loaded from disk')
    parser_observer.set_defaults(no_load_observers=False)

    parser_circle = subparsers.add_parser('circles', help='a help')
    parser_circle.add_argument('--radius', type=int, required=True,
                               help='number of hops away from center vertex')
    parser_circle.add_argument('--frac_masked_vertices', type=float, required=True,
                               help='the fraction of vertices of the mesh to be masked (result is approximate)')
    # TODO: This should be a param of all masking types. Implement also for observer
    parser_circle.add_argument('--masks_per_scene', type=int, required=True,
                               help='number of random masks to generate per scene')

    args = parser.parse_args()
    pretty_print_arguments(args)
    main(args)
    #if args.number == 0:
    #    write_args(args)
