import torch
import numpy as np
import trimesh
import argparse
import time
import os
from scipy.spatial import cKDTree as KDTree

import data_processing.implicit_waterproofing as iw
import models.local_model as model
from models.generation import Generator

def get_parser():
    parser = argparse.ArgumentParser(description="Point-Cloud Completion")
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-res", type=int, default=128)
    parser.add_argument("-num_points", type=int, default=3000)
    args = parser.parse_args()
    return args

def read_pcd(path):
    pcd = trimesh.load(path).vertices
    pcd = np.array(pcd)
    return pcd

def scale(points):
    # taken from data_processing/convert_to_scaled_off.py    
    # bounds – Bounding box with [min, max] coordinates
    bound_min = points.min(axis=0)[np.newaxis, :]
    bound_max = points.max(axis=0)[np.newaxis, :] 
    bounds = np.concatenate([bound_min, bound_max], axis=0)
    total_size = (bounds[1] - bounds[0]).max()
    centers = (bounds[1] + bounds[0]) /2

    points -= centers
    points /= total_size

    points /= 1.0

    return points


def occupancy(points, res=128, num_points=3000):
    # taken from data_processing/voxelized_pointcloud_sampling.py
    bb_min = -0.5
    bb_max = 0.5
    # make the 3D voxelized grid, the returned gridpoints are flattened
    grid_points = iw.create_grid_points_from_bounds(bb_min, bb_max, res)
    kdtree = KDTree(grid_points)

    np.random.seed(314)
    random_indexces = np.random.choice(points.shape[0], size=num_points, replace=False)
    sampled_points = points[random_indexces]

    occupancies = np.zeros(len(grid_points), dtype=np.int8)

    _, idx = kdtree.query(sampled_points)
    occupancies[idx] = 1

    # reshape the flat occupancy into a 3D tensor
    input = np.reshape(occupancies, (res,)*3)

    return sampled_points, input


    



if __name__ == "__main__":
    args = get_parser()
    checkpoint_dict = {300: 12, 3000: 23}
    input_dir = os.path.dirname(args.input)
    mesh_name = os.path.basename(args.input).split(".")[0]
    pcd = read_pcd(args.input)
    pcd = scale(pcd)

    # calculate occupancy
    sampled_point, pcd_occupancy = occupancy(pcd, args.res, args.num_points)
    input_pcd = trimesh.PointCloud(sampled_point)
    input_pcd.export("{}/{}_scaled.xyz".format(input_dir, mesh_name))
    net = model.ShapeNetPoints()

    # prepare the model and load checkpoint
    checkpoint = checkpoint_dict[args.num_points]
    gen = Generator(net, 0.5, exp_name='ShapeNetPoints', checkpoint=checkpoint, resolution=args.res, batch_points=100000)

    # predict
    inputs = torch.tensor(pcd_occupancy, dtype=torch.float32)
    inputs = torch.unsqueeze(inputs, 0) # add batch dimesnion
    data = {'inputs': inputs}
    logits = gen.generate_mesh(data)

    # make the mesh from prediction
    mesh = gen.mesh_from_logits(logits)

    # save to file
    export_path = '{}/output/'.format(input_dir)
    os.makedirs(export_path, exist_ok=True)
    mesh.export("{}/{}_ifnet.obj".format(export_path, mesh_name))