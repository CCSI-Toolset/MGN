import torch
from pathlib import Path


def generate_non_ghost_nodes_dict(patch_dict_base_path):

    patch_dict_base_path = Path(patch_dict_base_path)

    patch_dict_paths = list(patch_dict_base_path.iterdir())
    patch_dict_paths.sort()

    non_ghost_nodes_dict = {}
    for patch_dict_file in patch_dict_paths:

        patch_dict = torch.load(patch_dict_file)
        non_ghost_nodes_dict[patch_dict_file.name] = {"global_idx": [], "local_idx": []}

        for global_idx in patch_dict["can_send"].tolist():
            local_idx = patch_dict["node_global_idx_to_subset_idx"][global_idx]
            non_ghost_nodes_dict[patch_dict_file.name]["global_idx"].append(global_idx)
            non_ghost_nodes_dict[patch_dict_file.name]["local_idx"].append(local_idx)

    return non_ghost_nodes_dict


def generate_ghost_nodes_info_dict(patch_dict_base_path):

    patch_dict_base_path = Path(patch_dict_base_path)

    patch_dict_paths = list(patch_dict_base_path.iterdir())
    patch_dict_paths.sort()

    # Make Global Node ID to Patch ID Dict

    glob_id_to_patch_dict = {}
    for patch_dict_file in patch_dict_paths:

        patch_dict = torch.load(patch_dict_file)

        for glob_id in patch_dict["can_send"].tolist():
            if glob_id in glob_id_to_patch_dict:
                print("Someone already has id: {0}".format(glob_id))
            glob_id_to_patch_dict[glob_id] = {
                "patch_id": patch_dict_file.name,
                "local_idx": patch_dict["node_global_idx_to_subset_idx"][glob_id],
            }

    # Find Patches Needed for Ghost Node Updates

    ghost_nodes_info_dict = {}

    for patch_dict_file in patch_dict_paths:
        patch_dict = torch.load(patch_dict_file)

        patches_needed = {}
        for x in patch_dict["will_need"].tolist():
            needed_patch_id = glob_id_to_patch_dict[x]["patch_id"]
            source_local_idx = glob_id_to_patch_dict[x]["local_idx"]
            dest_local_idx = patch_dict["node_global_idx_to_subset_idx"][x]
            if needed_patch_id not in patches_needed:
                patches_needed[needed_patch_id] = {
                    "global_id": [],
                    "source_local_idx": [],
                    "dest_local_idx": [],
                }
            patches_needed[needed_patch_id]["global_id"].append(x)
            patches_needed[needed_patch_id]["source_local_idx"].append(source_local_idx)
            patches_needed[needed_patch_id]["dest_local_idx"].append(dest_local_idx)

        if len(patches_needed) > 0:
            ghost_nodes_info_dict[patch_dict_file.name] = patches_needed

    return ghost_nodes_info_dict
