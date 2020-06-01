import torch
import torch_geometric as pyg
from tqdm import tqdm


def dil_test():
    edge_index = torch.tensor([[10, 9], [9, 10], [9, 6], [6, 9], [9, 7], [7, 9], [0, 7], [7, 0], [0, 8], [8, 0], [10, 0],
                               [0, 10], [10, 3], [3, 10], [3, 1], [1, 3], [10, 11], [11, 10], [11, 2], [2, 11], [11, 5], [5, 11],
                               [5, 4], [4, 5], [6, 4], [4, 6], [8, 12], [12, 8], [12, 13], [13, 12], [6, 14], [14, 6],
                               [7, 14], [14, 7], [2, 15], [15, 2], [5, 15], [15, 5], [2, 16], [16, 2], [1, 16], [16, 1],
                               [2, 3], [3, 2], [3, 0], [0, 3], [4, 17], [17, 4], [11, 17], [17, 11], [9, 17], [17, 9],
                               [8, 13], [13, 8]], dtype=torch.long).t()

    x = torch.tensor([[2, -1], [-2, -4], [-5, -1], [-2, -1.5], [0, 3], [-4, 3], [3, 3], [4, 3], [4, -3], [2, 2], [1, 1],
                      [-3, 1], [6, -2], [8, -3], [5, 4], [-7, 1], [-6, -4], [1, 2]])

    x = torch.cat([x, torch.zeros(x.shape[0], 1)], dim=1)
    norms = torch.zeros_like(x)
    norms[:, -1] = 1.0

    dilations = [2, 4, 6, 8]
    all_dilated_edge_indices_new = compute_all_node_dilated_edges(edge_index.numpy(), x.numpy(), norms.numpy(), dilation=dilations)
    for i, dilation in enumerate(dilations):
        print('d{}'.format(dilation), all_dilated_edge_indices_new[i].t())


def plane_projection(n, u):
    return u - n * torch.dot(u, n) / (torch.linalg.vector_norm(n) * torch.linalg.vector_norm(u))


def cosine_similarity(a, b):
    return torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))


def cosine_similarity_in_plane(n, a, b):
    proj_a = plane_projection(n, a)
    proj_b = plane_projection(n, b)
    return cosine_similarity(proj_a, proj_b)


def edge_indices_to_adj_list(edge_indices):
    adj_lists = [[] for _ in range(torch.max(edge_indices).item() + 1)]
    for i in range(edge_indices.shape[1]):
        edge = edge_indices[:, i].numpy()
        adj_lists[edge[0]].append(edge[1])

    return [torch.tensor(adj_list) for adj_list in adj_lists]


def compute_all_node_dilated_edges(edge_indices_in, poses_in, norms_in, dilation=1):
    with torch.no_grad():
        edge_indices = pyg.utils.coalesce(torch.from_numpy(edge_indices_in))
        poses = torch.from_numpy(poses_in)
        norms = torch.from_numpy(norms_in)

        adj_lists = edge_indices_to_adj_list(edge_indices)

        num_nodes = poses.shape[0]
        all_dilated_edge_indices = [[] for _ in range(len(dilation))]
        for center_idx in tqdm(range(num_nodes)):
            dilated_edge_indices = compute_dilated_edges(center_idx, adj_lists, poses, norms, dilation)
            for i in range(len(all_dilated_edge_indices)):
                all_dilated_edge_indices[i] += dilated_edge_indices[i]

        for i in range(len(all_dilated_edge_indices)):
            if len(all_dilated_edge_indices[i]) != 0:
                all_dilated_edge_indices[i] = torch.tensor(all_dilated_edge_indices[i]).t()
                all_dilated_edge_indices[i] = pyg.utils.coalesce(all_dilated_edge_indices[i])

                # Format row wise to be compatible with cropping
                all_dilated_edge_indices[i] = all_dilated_edge_indices[i].t().long()

    return all_dilated_edge_indices


def sort_nodes_by_dist(node_inds, center_idx, poses):
    one_hop_directions = poses[node_inds] - poses[center_idx]
    one_hop_dists = torch.linalg.vector_norm(one_hop_directions, dim=1)
    one_hop_dists, one_hop_sorted_indices = torch.sort(one_hop_dists)
    return node_inds[one_hop_sorted_indices], one_hop_directions[one_hop_sorted_indices]


def compute_dilated_edges(center_idx, adj_lists, poses, norms, dilations=2):
    # TODO: It could be enforced that all n-dilated nodes must also be n-hop nodes.
    #  Further, if two nodes select the same neighbor as their dilated node, it could
    #  be possible to require one of those nodes to select their next-best neighbor
    #  (currently that neighbor will be selected by both resulting in fewer n-dilated nodes).

    if isinstance(dilations, int):
        dilations = [dilations]

    dilated_edge_indices = [[] for _ in range(len(dilations))]
    one_hop_inds = adj_lists[center_idx]
    if one_hop_inds.numel() > 0:
        # Sort one-hop indices by smallest distance from center node first.
        #  This helps us guarantee the order in which nodes are processed.
        one_hop_inds, _ = sort_nodes_by_dist(one_hop_inds, center_idx, poses)

        for one_hop_idx in one_hop_inds:
            if one_hop_idx != center_idx:
                last_idx = center_idx
                current_idx = one_hop_idx
                current_norm = norms[current_idx]
                current_direction = poses[current_idx] - poses[last_idx]
                dilation_idx = 0
                for current_dilation in range(2, max(dilations) + 1):
                    # Max similarity starts at zero to avoid choosing nodes that are greater than
                    #  90 degrees from the current one-hop direction as that would be akin to
                    #  traversing closer to the central node.
                    max_idx, max_similarity = -1, 0.0
                    for neighbor_idx in adj_lists[current_idx]:
                        if neighbor_idx not in one_hop_inds and neighbor_idx != last_idx:
                            neighbor_direction = poses[neighbor_idx] - poses[current_idx]
                            similarity = cosine_similarity_in_plane(current_norm, current_direction,
                                                                    neighbor_direction)

                            if similarity >= max_similarity:
                                max_similarity = similarity.item()
                                max_idx = neighbor_idx.item()

                    # Only write out requested dilations. Also don't continue dilating
                    #  if no further path is found.
                    if max_idx == -1:
                        break
                    else:
                        if current_dilation in dilations:
                            # Dilated edges point from dilated nodes toward central
                            #  node (for message passing convention)
                            dilated_edge_indices[dilation_idx].append([max_idx, center_idx])
                            dilation_idx += 1
                        last_idx = current_idx
                        current_idx = max_idx
                        current_norm = norms[current_idx]
                        current_direction = plane_projection(current_norm, current_direction)
                        current_direction /= torch.linalg.vector_norm(current_direction)

    return dilated_edge_indices


if __name__ == '__main__':
    dil_test()
