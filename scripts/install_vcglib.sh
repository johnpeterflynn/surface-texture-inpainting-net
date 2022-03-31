#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# QUADRIC ERROR METRICS
cd ${SCRIPT_DIR}/../vcglib/apps/tridecimator/
qmake
make

# VERTEX CLUSTERING
cd ../sample/trimesh_clustering
qmake
make