import sys

open3d_path = '/home/flynn/workspace/thesis/Open3D/build/lib/python_package/'
tc_path = '/home/flynn/workspace/thesis/remote/tangent_conv/'

sys.path.append(open3d_path)
from open3d import *

def get_tc_path():
	return tc_path
