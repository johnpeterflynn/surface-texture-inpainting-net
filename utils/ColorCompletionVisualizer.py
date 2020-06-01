"""General purpose visualizer for semantic segmentation results on various datasets super-fueled by open3D.
"""
import os
import open3d
import torch
import numpy as np
from termcolor import colored


class ColorCompletionVisualizer:
    """Visualize meshes from various datasets with open3D. Key Events show RGB, ground truth, prediction or differences."""

    def __init__(self, dataset, save_dir: str = ""):
        """Initialize Semantic Segmentation Visualizer which shows meshes with optional prediction and ground truth

        Arguments:
            dataset  {BaseDataSet} -- Examples from which dataset we want to visualize
            save_dir {str}         -- Directory in which .ply files should be saved
        """

        # keep a pointer to the dataset to retrieve relevant information, such as color mapping
        self._dataset = dataset
        
        if not os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))

        assert os.path.isdir(save_dir) or save_dir == ""
        self._save_dir = save_dir

    def visualize_result(self, mesh_name, prediction=None, gt=None, mask=None):
        mesh = self._dataset.get_mesh(mesh_name)
        mesh.compute_vertex_normals()

        vis = open3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(width=1600, height=1200)

        # PREPARE RGB COLOR SWITCH
        rgb_colors = open3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))

        def colorize_rgb(visu):
            mesh.vertex_colors = rgb_colors
            visu.update_geometry(mesh)
            visu.update_renderer()

        vis.register_key_callback(ord('H'), colorize_rgb)

        if type(prediction) == torch.Tensor:
            # PREPARE PREDICTION COLOR SWITCH
            pred_colors = open3d.utility.Vector3dVector(np.asarray(prediction))

            def colorize_pred(visu):
                mesh.vertex_colors = pred_colors
                visu.update_geometry(mesh)
                visu.update_renderer()

            vis.register_key_callback(ord('J'), colorize_pred)

        if type(gt) == torch.Tensor:
            # PREPARE GROUND TRUTH COLOR SWITCH
            gt_shape = np.asarray(gt).shape
            rgb_shape = np.asarray(rgb_colors).shape
            if gt_shape != rgb_shape:
                print('WARNING: ground truth shape', gt_shape, 'differs from loaded rgb shape', rgb_shape)
            gt_colors = open3d.utility.Vector3dVector(np.asarray(gt))

            def colorize_gt(visu):
                mesh.vertex_colors = gt_colors
                visu.update_geometry(mesh)
                visu.update_renderer()

            vis.register_key_callback(ord('K'), colorize_gt)

        if type(mask) == torch.Tensor:
            # PREPARE GROUND TRUTH COLOR SWITCH
            mask_colors = torch.where(mask.expand_as(gt), torch.zeros_like(gt), gt)
            mask_colors = open3d.utility.Vector3dVector(np.asarray(mask_colors))

            def colorize_mask(visu):
                mesh.vertex_colors = mask_colors
                visu.update_geometry(mesh)
                visu.update_renderer()
                print('Rendering mask')

            vis.register_key_callback(ord('M'), colorize_mask)

        if type(gt) == torch.Tensor and type(prediction) == torch.Tensor:
            # PREAPRE DIFFERENCE COLOR SWITCH
            def generate_heatvector(v, min=0.0, max=0.5):
                out = torch.zeros_like(v).unsqueeze(1).repeat(1, 3)
                ratio = 2 * (v - min) / (max - min)
                out[:, 0] = torch.maximum(torch.zeros_like(ratio), (ratio - 1))
                out[:, 2] = torch.maximum(torch.zeros_like(ratio), (1 - ratio))
                out[:, 1] = 1.0 - out[:, 0] - out[:, 2]
                return out

            d = torch.abs(gt - prediction).mean(dim=1)
            differences = generate_heatvector(d)
            differences = torch.where(mask.expand_as(differences), differences, torch.zeros_like(differences)+0.25)
            diff_colors = open3d.utility.Vector3dVector(np.asarray(differences))

            def colorize_diff(visu):
                mesh.vertex_colors = diff_colors
                visu.update_geometry(mesh)
                visu.update_renderer()

            vis.register_key_callback(ord('F'), colorize_diff)

        def take_screenshot(visu):
            path = f"{self._save_dir}/SemSegVisualizer_img.png"
            visu.capture_screen_image(path)
            print('Saving screenshot to', path)

        vis.register_key_callback(ord('P'), take_screenshot)

        def save_room(visu):
            mesh.vertex_colors = rgb_colors
            open3d.io.write_triangle_mesh(
                f"{self._save_dir}/SemSegVisualizer_{mesh_name}_rgb.ply", mesh)

            if type(prediction) == torch.Tensor:
                mesh.vertex_colors = pred_colors
                open3d.io.write_triangle_mesh(
                    f"{self._save_dir}/SemSegVisualizer_{mesh_name}_pred.ply", mesh)

            if type(gt) == torch.Tensor:
                mesh.vertex_colors = gt_colors
                open3d.io.write_triangle_mesh(
                    f"{self._save_dir}/SemSegVisualizer_{mesh_name}_gt.ply", mesh)

            if type(mask) == torch.Tensor:
                mesh.vertex_colors = mask_colors
                open3d.io.write_triangle_mesh(
                    f"{self._save_dir}/SemSegVisualizer_{mesh_name}_mask.ply", mesh)

            if type(gt) == torch.Tensor and type(prediction) == torch.Tensor:
                mesh.vertex_colors = diff_colors
                open3d.io.write_triangle_mesh(
                    f"{self._save_dir}/SemSegVisualizer_{mesh_name}_diff.ply", mesh)
            print(colored(
                f"PLY meshes successfully stored in {os.path.abspath(self._save_dir)}", 'green'))

        # Start with unlit prediction
        colorize_pred(vis)
        vis.get_render_option().light_on = False

        vis.register_key_callback(ord('D'), save_room)
        vis.add_geometry(mesh)
        vis.run()
        vis.destroy_window()
