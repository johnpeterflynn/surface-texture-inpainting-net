# Surface Texture Inpainting Network
Completing surface texture on partially textured 3D meshes using graph neural networks.

# Installation

Create a new conda environment, replacing <env_name> with a name of your choice. 
```
conda create -n <env_name> -c pytorch pytorch torchvision python=3.9
```

Ensure you are using pip within your virtual environment.
```
conda install pip
```

Install packages from requirements.txt. This will also install necessary libraries such as OpenCV and Open3D. PyTorch Geometric (pyg) must be installed separately. 

```
pip install -r requirements.txt
conda install pyg -c pyg
```

To generate mesh simplification levels we utilize tridecimator during preprocessing. Tridecimator is contained in our fork of VCGLib which is a submodule of this repository.


```
git submodule update --init --recursive
scripts/install_vcglib.sh
```

# Dataset Preparation

ln -s /mnt/raid/datasets/scannet scannet
