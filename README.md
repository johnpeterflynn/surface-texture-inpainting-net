# Surface Texture Inpainting Network
Inpainting with deep learning looks great these days but what if the pixels don't live on a flat, rectangular image?

Completing surface texture on partially textured 3D meshes using graph neural networks.

# Getting Started

## Installation

Clone this repo.
```bash
https://github.com/johnpeterflynn/surface-texture-inpainting-net
cd surface-texture-inpainting-net
```

Create a new conda environment containing PyTorch, replacing <env_name> with a name of your choice. 
```bash
conda create -n <env_name> -c pytorch pytorch torchvision python=3.9
```

Ensure you are using pip within your virtual environment.
```bash
conda install pip
```

Install packages from requirements.txt. This will also install necessary libraries such as OpenCV and Open3D. PyTorch Geometric (pyg) must be installed separately. 

```bash
pip install -r requirements.txt
conda install pyg -c pyg
```

To generate mesh simplification levels we utilize tridecimator during preprocessing. Tridecimator is contained in our fork of VCGLib which is a submodule of this repository.


```bash
git submodule update --init --recursive
scripts/install_vcglib.sh
```

## Dataset Preparation

### 3D Dataset: ScanNet

In our experiments we use ScanNet, a dataset containing 3D reconstructions of 1500 indoor scenes.

#### Download ScanNet

Please refer to https://github.com/ScanNet/ScanNet to gain access to the ScanNet dataset. It could take a few days to be granted access.

Our scripts only rely on ScanNet's low-resolution meshes. You can download these by specifying the file type _\_vh_clean_2.ply_ in the download script provided by ScanNet.

#### Symbolic link pointing to dataset

Once you've downloaded ScanNet you should link this project to them by placing a symbolic link within this project directory.

```bash
ln -s /path/to/downloaded/scannet/files data/scannet
```

### 2D Dataset: Images of Textures

Unfortunately our texture dataset is not publically available. If you would like to build your own you can aggregate images of textures from the following smaller datasets.

## Run Preprocessing and Training

We provide scripts to preprocess ScanNet scenes and train STINet. Each experiment subfolder contains a configuration file used for training. Modify these scripts and the configuration files to set up your own training and inference pipelines.

#### 2D Image Inpainting

```bash
experiments/2d_inpainting/run_2d_inpainting.sh
```

#### 3D Surface Inpainting

NOTE: Preprocessing each ScanNet scene takes roughly 30 minutes but processing scenes can be distributed amongst available CPU cores.

```bash
experiments/3d_inpainting/preprocess_3d_inpainting.sh
experiments/3d_inpainting/run_3d_inpainting.sh
```

## Preprocessing Pipeline

#### Step 1

`scripts/generate_graph_levels.sh` parses ScanNet meshes into graphs and computes various versions of the graphs used in STINet. For each mesh it computes mesh simplification levels, pooling maps between vertices of consecutive levels and vertex neighborhoods for dilated convolutions.

This outputs files containing graphs as serialized tensor objects to data/generated/graph_levels/<preprocess_name>/train and data/generated/graph_levels/<preprocess_name>/val.

#### Step 2

`scripts/generate_crops.sh` divides each mesh into smaller meshes of a fixed spatial size to better fit into GPU memory. 

This outputs files containing cropped graphs as serialized tensor objects to data/generated/cropped/<preprocess_name>/train. Cropping is not computed for the vaidation set.

#### Step 3

`scripts/generate_masks.sh` computes masks for each mesh. 

This outputs binary files containing vertex masks as serialized numpy arrays to data/generated/graph_levels/<preprocess_name>/train/masks/<mask_name>/, data/generated/graph_levels/<preprocess_name>/val/masks/<mask_name>/ and data/generated/cropped/<preprocess_name>/train/masks/<mask_name>/.
