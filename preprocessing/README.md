

# Preprocessing Pipeline

### Step 1

`scripts/generate_graph_levels.sh` parses ScanNet meshes into graphs and computes various versions of the graphs used in STINet. For each mesh it computes mesh simplification levels, pooling maps between vertices of consecutive levels and vertex neighborhoods for dilated convolutions.

This outputs files containing graphs as serialized tensor objects to data/generated/graph_levels/<preprocess_name>/train and data/generated/graph_levels/<preprocess_name>/val.

### Step 2

`scripts/generate_crops.sh` divides each mesh into smaller meshes of a fixed spatial size to better fit into GPU memory. 

This outputs files containing cropped graphs as serialized tensor objects to data/generated/cropped/<preprocess_name>/train. Cropping is not computed for the vaidation set.

### Step 3

`scripts/generate_masks.sh` computes masks for each mesh. 

This outputs binary files containing vertex masks as serialized numpy arrays to data/generated/graph_levels/<preprocess_name>/train/masks/<mask_name>/, data/generated/graph_levels/<preprocess_name>/val/masks/<mask_name>/ and data/generated/cropped/<preprocess_name>/train/masks/<mask_name>/.
