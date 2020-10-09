import argparse

from sidewalk.robots.jackal.data.jackal_hdf5_visualizer import JackalHDF5Visualizer
from sidewalk.utils import file_utils


parser = argparse.ArgumentParser()
parser.add_argument('-folders', nargs='+', help='list of folders containing hdf5s')
parser.add_argument('-horizon', type=int, default=8)
args = parser.parse_args()

hdf5_fnames = file_utils.recursive_get_files_ending_with(args.folders, '.hdf5')
assert len(hdf5_fnames) > 0
JackalHDF5Visualizer(hdf5_fnames, args.horizon).run()
