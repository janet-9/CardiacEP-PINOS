# Using the functions for igb reading and pts file reading from the openCARP-PINNs paper
# Reads the vm,wm simulation files and the mesh pts file and constructs a 2 channel tensor encoding the output 
# Option to save the output with varying spatial and temporal resolution abstraction. 

import os
import re
import struct
import numpy as np
import argparse
import torch

#Extract the info from the igb simulation outpt
def read_array_igb(igbfile):
    """
    Read a .igb file and return as a list of NumPy arrays (time x nodes)
    """
    data = []
    with open(igbfile, mode="rb") as file:
        header = file.read(1024)
        words = header.split()
        word = []
        for i in range(4):
            word.append(int([re.split(r"(\d+)", s.decode("utf-8")) for s in [words[i]]][0][1]))

        nnode = word[0] * word[1] * word[2]
        nframes = os.path.getsize(igbfile) // (4 * nnode)

        for _ in range(nframes):
            data.append(struct.unpack("f" * nnode, file.read(4 * nnode)))

    return np.array(data)  # shape: (T, nnode)

# Extract the nodal points from the mesh file
def read_pts(modname, n=3, vtx=False, item_type=float):
    """Read .pts (or .vtx) coordinate file"""
    with open(modname + (".vtx" if vtx else ".pts")) as file:
        count = int(file.readline().split()[0])
        if vtx:
            file.readline()

        pts = np.empty((count, n), item_type)
        for i in range(count):
            pts[i] = [item_type(val) for val in file.readline().split()[0:n]]

    return pts if n > 1 else pts.flat

# Embed the vm and wm data into a tensor structure using the mesh grid. 
# Note that the time interval saving is 1ms, the mesh is 10cm by 10cm with a resolution of 250um.
def generate_data(v_file_name, w_file_name, pt_file_name, t_limit=None):
    """Reconstruct structured voltage and recovery fields"""
    data_V = np.array(read_array_igb(v_file_name))  # (T, nnode)
    data_W = np.array(read_array_igb(w_file_name))  # (T, nnode)
    coordinates = np.array(read_pts(pt_file_name))  # (nnode, 3)

    # Apply time limit if provided
    if t_limit is not None and t_limit < data_V.shape[0]:
        data_V = data_V[:t_limit, :]
        data_W = data_W[:t_limit, :]
        print(f"Time limited to {t_limit} / {data_V.shape[0]} steps")
    else:
        print(f"Using all {data_V.shape[0]} timesteps")

    t = np.arange(0, data_V.shape[0]).reshape(-1, 1)

    # Normalize and select only x, y coordinates to cm from the um values - should give a range 0-10 
    coordinates = (coordinates - np.min(coordinates)) / 10000
    coordinates = coordinates[:, 0:2]

    x = np.unique(coordinates[:, 0]).reshape((1, -1))
    y = np.unique(coordinates[:, 1]).reshape((1, -1))
    len_x, len_y, len_t = x.shape[1], y.shape[1], t.shape[0]

    # Normalize and reshape voltage and recovery
    data_V = (data_V + 80) / 100  # scaling for AP data
    data_V = data_V.T.reshape(len_x, len_y, len_t)
    data_W = data_W.T.reshape(len_x, len_y, len_t)

    # Stack into a 2-channel tensor (C, X, Y, T)
    tensor = np.stack([data_V, data_W], axis=0)  # (2, X, Y, T)
    # Permute the tensor to t,x,y form
    tensor = np.transpose(tensor, (0, 3, 1, 2))  # (2, t, x, y)
    return tensor, coordinates, x, y, t


def downsample_tensor_space(tensor, factor):
    """Spatially downsample a (2, T, X, Y) tensor"""
    return tensor[:, :, ::factor, ::factor]


def downsample_tensor_time(tensor, factor):
    """Temporally downsample a (2, T, X, Y) tensor"""
    return tensor[:, ::factor, :, :]


def main():
    parser = argparse.ArgumentParser(
        description="Extract and downsample openCARP simulation data as PyTorch tensors."
    )
    parser.add_argument(
        "--sim_folder",
        type=str,
        required=True,
        help="Path to simulation output folder containing .igb and .pts files.",
    )
    parser.add_argument(
        "--space_factors",
        type=int,
        nargs="+",
        default=[10, 12],
        help="Spatial resolution downsampling factors to apply (1 = full resolution).",
    )

    parser.add_argument(
        "--time_factors",
        type=int,
        nargs="+",
        default=[5],
        help="Temporal resolution downsampling factors to apply (1 = full resolution).",
    )

    parser.add_argument(
        "--sim_type",
        type=str,
        required=True,
        help="Simulation type (stable, chaotic, planar, centrifugal)"
    )
    parser.add_argument(
        "--runtime",
        type=int,
        required=True,
        help="Runtime extracted from simulation to the tensor output (eg. 5000ms)"
    )
    parser.add_argument(
        "--conmul",
        type=float,
        default = 1.0,
        help="Conductivity Multipler for the tissue (from baseline). Default is %(default)s"
    )
    args = parser.parse_args()

    sim_dir = os.path.abspath(args.sim_folder)
    v_file = os.path.join(sim_dir, "vm.igb")
    w_file = os.path.join(sim_dir, "w.igb")
    pts_file = os.path.join(sim_dir, "2D_10cm_250um_i")

    print(f"Reading simulation data from: {sim_dir}")
    tensor, coords, x, y, t = generate_data(v_file, w_file, pts_file, t_limit=args.runtime)

    # Convert to torch tensor
    tensor_torch = torch.tensor(tensor, dtype=torch.float32)

    # Save full-resolution tensor
    save_name = f"2D_AP_{args.sim_type}_250um_{args.runtime}ms_{args.conmul}_x1_t1.pt"
    full_path = os.path.join(sim_dir, save_name)
    coords_path = os.path.join(sim_dir, "coords_cm.npy")
    torch.save(tensor_torch, full_path)
    
    np.save(coords_path, coords)

    print(f"Saved full-resolution tensor: {full_path}")
    print(f"Saved coordintaes (normalised to cm): {coords_path}")

    # Save downsampled versions (spatiotemporal)
    for s_factor in args.space_factors:
        for t_factor in args.time_factors:
            # Skip full-resolution (1x in both)
            if s_factor == 1 and t_factor == 1:
                continue

            # Apply temporal downsampling first (time axis)
            temp_down = downsample_tensor_time(tensor, t_factor)

            # Then apply spatial downsampling (x, y axes)
            downsampled = downsample_tensor_space(temp_down, s_factor)

            # Save
            tensor_down = torch.tensor(downsampled, dtype=torch.float32)
            path = os.path.join(
                sim_dir,
                f"2D_AP_{args.sim_type}_250um_{args.runtime}ms_{args.conmul}_x{s_factor}_t{t_factor}.pt"
            )
            torch.save(tensor_down, path)
            print(f"Saved downsampled tensor (space {s_factor}x, time {t_factor}x): {path}")

    print("All tensorised data saved successfully.")
    
    
    for f in args.space_factors:
        if f > 1:
            downsampled = downsample_tensor_space(tensor, f)
            tensor_down = torch.tensor(downsampled, dtype=torch.float32)
            path = os.path.join(sim_dir, f"{save_name}_x{f}.pt")
            torch.save(tensor_down, path)
            print(f"Saved spatially downsampled tensor (factor {f}): {path}")

    print("All tensorised data saved successfully.")


if __name__ == "__main__":
    main()
