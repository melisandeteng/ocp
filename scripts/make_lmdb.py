
from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.datasets import SinglePointLmdbDataset, TrajectoryLmdbDataset
import argparse
import ase.io
from ase.build import bulk
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS
import matplotlib.pyplot as plt
import lmdb
import pickle
from tqdm import tqdm
import torch
import os
import json

def read_trajectory_extract_features(a2g, traj_path):
    traj = ase.io.read(traj_path, ":")
    tags = traj[0].get_tags()
    images = [traj[0], traj[-1]]
    data_objects = a2g.convert_all(images, disable_tqdm=True)
    data_objects[0].tags = torch.LongTensor(tags)
    data_objects[1].tags = torch.LongTensor(tags)
    return data_objects

def make_lmdb(system_paths, db_name, ref_energies_file=None):
    """
    arguments:
        system_paths: list of paths to .traj files
        db_name: "db_name.lmdb" where to save the file 
    """
    substract_ref = False
    if ref_energies_file is not None:
        substract_ref = True
        with open(ref_energies_file, 'r') as f:
            ref_energies = json.load(f)
            
    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=6,
        r_energy=True,    # False for test data
        r_forces=True,
        r_distances=False,
        r_fixed=True,
    )

    db = lmdb.open(
        db_name,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
        )

    idx = 0
    for system in system_paths:
    # Extract Data object
        
        data_objects = read_trajectory_extract_features(a2g, system)
        initial_struc = data_objects[0] 
        relaxed_struc = data_objects[1]

        initial_struc.y_init = initial_struc.y  # subtract off reference energy, if applicable
        del initial_struc.y
        initial_struc.y_relaxed = relaxed_struc.y # subtract off reference energy, if applicable
        if substract_ref:
            print("susbtracting ref")
            object_id = os.path.basename(system).split(".")[0]
            ref_energy = ref_energies[object_id]
            initial_struc.y_init -= ref_energy
            initial_struc.y_relaxed -= ref_energy
            initial_struc.system = object_id
        initial_struc.pos_relaxed = relaxed_struc.pos

    # Filter data if necessary
    # OCP filters adsorption energies > |10| eV

        initial_struc.sid = idx  # arbitrary unique identifier

    # no neighbor edge case check
        if initial_struc.edge_index.shape[1] == 0:
            print("no neighbors", traj_path)
            continue

    # Write to LMDB
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(initial_struc, protocol=-1))
        txn.commit()
        db.sync()
        idx += 1

    db.close()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        help="Root dir of .traj files location",
    )
    parser.add_argument(
        "--paths-file",
        help="Path to txt file containing list of .traj file names",
    )
    parser.add_argument(
        "--dbname",
        help="db name",
    )
    
    parser.add_argument(
        "--refenergy",
        help="db name",
    )
    return parser


if __name__=="__main__":
    
    parser = get_parser()
    args = parser.parse_args()
    with open(args.paths_file, "r") as f:
        filenames = [line.rstrip() for line in f]
    
    system_paths = [os.path.join(args.root, fn) for fn in filenames]
    
    make_lmdb(system_paths, args.dbname, args.refenergy)
    print("saved file in " + args.dbname)