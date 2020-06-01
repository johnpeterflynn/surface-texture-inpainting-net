"""Compute an optimal mesh texture given color images and camera poses
"""
import os
import argparse
import cv2
import glob
import numpy as np
import open3d as o3d
import re
from tqdm import tqdm


def load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path, True)
    #mesh.remove_non_manifold_edges()
    #mesh.remove_duplicated_vertices()
    #mesh.remove_duplicated_triangles()
    #mesh.remove_degenerate_triangles()
    #mesh.remove_unreferenced_vertices()
    #mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(mesh)

    #if not mesh.has_vertex_normals():
    #    print('Computing vertex normals')
    #    mesh.compute_vertex_normals()

    #if not mesh.has_triangle_normals():
    #    print('Computing triangle normals')
    #    mesh.compute_triangle_normals()
    #mesh.normalize_normals()

    #mesh.vertex_colors = o3d.utility.Vector3dVector([])

    print("Try to render a mesh with normals (exist: " +
          str(mesh.has_vertex_normals()) + ") and colors (exist: " +
          str(mesh.has_vertex_colors()) + ") and uvs (exist: " +
          str(mesh.has_triangle_uvs()) + ")")

    #if not mesh.has_vertex_colors():
    #    print('Adding blank color')
    #    vertices = np.asarray(mesh.vertices)
    #    vertex_colors = np.zeros_like(vertices)
    #    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    #    #vertex_colors = np.asarray(mesh.vertex_colors)

    return mesh


def load_dataset(path, scene_name, height, width, trajectory_slice):

    def get_file_list(path, extension=None):

        def sorted_alphanum(file_list_ordered):
            convert = lambda text: int(text) if text.isdigit() else text
            alphanum_key = lambda key: [
                convert(c) for c in re.split('([0-9]+)', key)
            ]
            return sorted(file_list_ordered, key=alphanum_key)

        if extension is None:
            file_list = [
                path + f
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
            ]
        else:
            file_list = [
                path + f
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f)) and
                os.path.splitext(f)[1] == extension
            ]
        file_list = sorted_alphanum(file_list)
        return file_list

    depth_image_path = get_file_list(os.path.join(path, "depth/"), extension=".png")[trajectory_slice]
    color_image_path = get_file_list(os.path.join(path, "color/"), extension=".jpg")[trajectory_slice]

    assert (len(depth_image_path) == len(color_image_path))

    orig_height, orig_width, _ = cv2.imread(os.path.join(color_image_path[0]), cv2.COLOR_BGR2RGB).shape

    rgbd_images = []
    for i in tqdm(range(len(depth_image_path))):
        # TODO: Is depth scaled correctly?
        # TODO: bilateral filter
        depth = cv2.imread(os.path.join(depth_image_path[i]), cv2.IMREAD_ANYDEPTH)  # TODO: Explicitly specify image type
        assert depth.shape[0] == height and depth.shape[1] == width, 'ERROR: Window height/width != depth height/width'
        depth[depth == 65535] = 0
        #depth = cv2.bilateralFilter(depth.)
        #depth = (depth / 1000).astype(np.uint16)
        color = cv2.imread(os.path.join(color_image_path[i]), cv2.COLOR_BGR2RGB)
        color[:, :, [2, 0]] = color[:, :, [0, 2]]
        color = cv2.resize(color, (depth.shape[1], depth.shape[0]))  # TODO: ensure correct interpolation used
        depth = o3d.geometry.Image(depth)
        color = o3d.geometry.Image(color)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, convert_rgb_to_intensity=False)
        rgbd_images.append(rgbd_image)

    filenames = glob.glob(os.path.join(path, 'pose', '*.txt'))
    filenames.sort()
    # TODO: What about depth intrinsics?
    ic = np.loadtxt(os.path.join(path, 'intrinsic', 'intrinsic_color.txt'))
    ic = np.array([[ic[0, 0] * width / orig_width, 0, width / 2.0 - 0.5],
                   [0, ic[1, 1] * height / orig_height, height / 2.0 - 0.5],
                   [0, 0, 1]])

    def read_camera_params(filename):
        # TODO: Account for skew
        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic.set_intrinsics(width, height, ic[0, 0], ic[1, 1], ic[0, 2], ic[1, 2])
        loaded_extrinsic = np.loadtxt(filename, dtype=np.float64)
        params.extrinsic = np.linalg.inv(loaded_extrinsic)

        return params

    camera_trajectory = o3d.camera.PinholeCameraTrajectory()
    camera_trajectory.parameters = [read_camera_params(filename) for filename in filenames][trajectory_slice]

    mesh = load_mesh(os.path.join(path, "{}_vh_clean_2.ply".format(scene_name)))

    return mesh, rgbd_images, camera_trajectory


def main():
    path = './data/sensor_data/scene0000_00'
    scene_name = 'scene0000_00'
    trajectory_slice = slice(None, None, 10)
    main.index = 0
    height = 480  # 968
    width = 640  # 1296
    mesh, rgbd_images, camera_trajectory = load_dataset(path, scene_name, height, width, trajectory_slice)
    print('length images, cam traj:', len(rgbd_images), len(camera_trajectory.parameters))
    print('Optimizer started')
    #mesh_optimized = o3d.pipelines.color_map.run_rigid_optimizer(
    #    mesh, rgbd_images, camera_trajectory,
    #    o3d.pipelines.color_map.RigidOptimizerOption(maximum_iteration=0))
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh_optimized = o3d.pipelines.color_map.run_non_rigid_optimizer(
            mesh, rgbd_images, camera_trajectory,
            o3d.pipelines.color_map.NonRigidOptimizerOption(
                maximum_iteration=0))
    o3d.io.write_triangle_mesh(os.path.join(path, "out.ply"), mesh_optimized)
    print('Optimized finished')
    input()

    import time

    def move_forward(vis):
        ctr = vis.get_view_control()
        print(camera_trajectory.parameters[0].intrinsic)
        if main.index < len(camera_trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(
                camera_trajectory.parameters[main.index])
        else:
            vis.register_animation_callback(None)
        time.sleep(0.1)
        main.index += 1

        #params = ctr.convert_to_pinhole_camera_parameters()
        #print('View control intrinsic:', params.intrinsic.intrinsic_matrix)
        #print('View control extrinsic:', params.extrinsic)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    vis.add_geometry(mesh_optimized)
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()

    #o3d.visualization.draw_geometries([mesh_optimized],
    #                                  zoom=0.5399,
    #                                  front=[0.0665, -0.1107, -0.9916],
    #                                  lookat=[0.7353, 0.6537, 1.0521],
    #                                  up=[0.0136, -0.9936, 0.1118])


if __name__ == '__main__':
    main()
