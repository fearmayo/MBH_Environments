#!/usr/bin/env python3
"""
Opens an HDF5 catalogue and prints all available groups/datasets,
along with their shape and dtype.
"""
import sys
import h5py


def print_fields(filename):
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name:50s}  shape={obj.shape}  dtype={obj.dtype}")
        else:
            print(f"{name}/")

    with h5py.File(filename, "r") as f:
        print(f"Available fields in: {filename}\n")
        f.visititems(visitor)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        # default path - change as needed
        filename = "/home/regan/Dropbox/LISA_GW/MBHEnvironments/Catalogues/SEEDZ/MBH_Environment_Catalog_SEEDZ.hdf5"

    print_fields(filename)
