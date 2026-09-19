# This file is part of Robust Weight Transfer for Blender.
#
# Portions of this code are based on:
#   RobustSkinWeightsTransferCode (https://github.com/rin-23/RobustSkinWeightsTransferCode/blob/main/src/utils.py)
#   by Rinat Abdrashitov, used under the MIT License (see below).
#
# Changes were made to make the code compatible with Blender's data structures
# and to improve performance and robustness.
#
# Copyright (C) 2025 sentfromspacevr
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Attribution: Developed by sentfromspacevr (https://github.com/sentfromspacevr)
#
# ---- Original MIT License Notice Follows ----
#
# The following portions of this file are based on work by Rinat Abdrashitov and are licensed under the MIT License:
#
# Copyright (c) 2024 Rinat Abdrashitov
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import warnings
from dataclasses import dataclass

import numpy as np
import scipy as sp
import robust_laplacian
from mathutils.bvhtree import BVHTree
from scipy.spatial import cKDTree


def build_surface_bvh(V, F):
    """Build Blender's native acceleration structure for a triangle mesh."""
    V = np.asarray(V)
    F = np.asarray(F)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError("Source vertices must be an N by 3 array")
    if F.ndim != 2 or F.shape[1] != 3 or len(F) == 0:
        raise ValueError("Source mesh has no triangles")
    if np.any(F < 0) or np.any(F >= len(V)):
        raise ValueError("Source mesh contains invalid triangle indices")

    tree = BVHTree.FromPolygons(V.tolist(), F.tolist(), all_triangles=True)
    if tree is None:
        raise ValueError("Could not build a surface search tree for the source mesh")
    return tree


def _barycentric_coordinates(points, triangle_vertices):
    """Calculate barycentric coordinates for corresponding points/triangles."""
    a = triangle_vertices[:, 0]
    ab = triangle_vertices[:, 1] - a
    ac = triangle_vertices[:, 2] - a
    ap = points - a

    d00 = np.einsum('ij,ij->i', ab, ab)
    d01 = np.einsum('ij,ij->i', ab, ac)
    d11 = np.einsum('ij,ij->i', ac, ac)
    d20 = np.einsum('ij,ij->i', ap, ab)
    d21 = np.einsum('ij,ij->i', ap, ac)
    denominator = d00 * d11 - d01 * d01

    barycentric = np.zeros((len(points), 3), dtype=np.float64)
    scale = np.maximum(d00 * d11, np.finfo(np.float64).tiny)
    valid = np.abs(denominator) > np.finfo(np.float64).eps * scale
    if np.any(valid):
        barycentric[valid, 1] = (
            d11[valid] * d20[valid] - d01[valid] * d21[valid]
        ) / denominator[valid]
        barycentric[valid, 2] = (
            d00[valid] * d21[valid] - d01[valid] * d20[valid]
        ) / denominator[valid]
        barycentric[valid, 0] = 1.0 - barycentric[valid, 1] - barycentric[valid, 2]

    if np.any(~valid):
        # Degenerate faces do not have unique barycentric coordinates. Assign
        # the result to the face vertex nearest the BVH hit point.
        degenerate_triangles = triangle_vertices[~valid]
        degenerate_points = points[~valid, np.newaxis, :]
        offsets = degenerate_triangles - degenerate_points
        nearest = np.argmin(
            np.einsum('ijk,ijk->ij', offsets, offsets),
            axis=1,
        )
        barycentric[np.flatnonzero(~valid), nearest] = 1.0

    return barycentric


def find_closest_point_on_surface(P, V, F, surface_bvh=None):
    """
    Given a number of points find their closest points on the surface of the V,F mesh

    Args:
        P: #P by 3, where every row is a point coordinate
        V: #V by 3 mesh vertices
        F: #F by 3 mesh triangles indices
    Returns:
        sqrD #P smallest squared distances
        I #P primitive indices corresponding to smallest distances
        C #P by 3 closest points
        B #P by 3 of the barycentric coordinates of the closest point
    """
    
    P = np.asarray(P)
    V = np.asarray(V)
    F = np.asarray(F)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("Target vertices must be an N by 3 array")
    if surface_bvh is None:
        surface_bvh = build_surface_bvh(V, F)

    sqrD = np.empty(len(P), dtype=np.float64)
    I = np.empty(len(P), dtype=np.int64)
    C = np.empty((len(P), 3), dtype=np.float64)
    for index, point in enumerate(P):
        closest, _normal, triangle_index, distance = surface_bvh.find_nearest(point)
        if closest is None or triangle_index is None or distance is None:
            raise ValueError("Could not find a closest point on the source mesh")
        sqrD[index] = distance * distance
        I[index] = triangle_index
        C[index] = closest

    B = _barycentric_coordinates(C, V[F[I]])

    return sqrD,I,C,B

def interpolate_attribute_from_bary(A,B,I,F):
    """
    Interpolate per-vertex attributes A via barycentric coordinates B of the F[I,:] vertices

    Args:
        A: #V by N per-vertex attributes
        B  #B by 3 array of the barycentric coordinates of some points
        I  #B primitive indices containing the closest point
        F: #F by 3 mesh triangle indices
    Returns:
        A_out #B interpolated attributes
    """
    F_closest = F[I,:]
    a1 = A[F_closest[:,0],:]
    a2 = A[F_closest[:,1],:]
    a3 = A[F_closest[:,2],:]

    b1 = B[:,0]
    b2 = B[:,1]
    b3 = B[:,2]

    b1 = b1.reshape(-1,1)
    b2 = b2.reshape(-1,1)
    b3 = b3.reshape(-1,1)
    
    A_out = a1*b1 + a2*b2 + a3*b3

    return A_out


def normalize_vec(v):
    return v/np.linalg.norm(v)


def find_matches_closest_surface(source_verts, source_triangles, source_normals, target_verts, target_normals, source_weights, dDISTANCE_THRESHOLD_SQRD, dANGLE_THRESHOLD_DEGREES, flip_vertex_normal, surface_bvh=None, return_distances=False):
    """
    For each vertex on the target mesh find a match on the source mesh.

    Args:
        V1: #V1 by 3 source mesh vertices
        F1: #F1 by 3 source mesh triangles indices
        N1: #V1 by 3 source mesh normals
        
        V2: #V2 by 3 target mesh vertices
        F2: #F2 by 3 target mesh triangles indices
        N2: #V2 by 3 target mesh normals
        
        W1: #V1 by num_bones source mesh skin weights

        dDISTANCE_THRESHOLD_SQRD: scalar distance threshold
        dANGLE_THRESHOLD_DEGREES: scalar normal threshold

    Returns:
        Matched: #V2 array of bools, where Matched[i] is True if we found a good match for vertex i on the source mesh
        W2: #V2 by num_bones, where W2[i,:] are skinning weights copied directly from source using closest point method
        sqrD: optional third result, squared world-space surface distances
    """
    sqrD,I,C,B = find_closest_point_on_surface(
        target_verts, source_verts, source_triangles, surface_bvh
    )
    
    # for each closest point on the source, interpolate its per-vertex attributes(skin weights and normals) 
    # using the barycentric coordinates
    W2 = interpolate_attribute_from_bary(source_weights,B,I,source_triangles)
    N1_match_interpolated = interpolate_attribute_from_bary(source_normals,B,I,source_triangles)
    
    norm_N1 = np.linalg.norm(N1_match_interpolated, axis=1, keepdims=True)
    norm_N2 = np.linalg.norm(target_normals, axis=1, keepdims=True)
    valid_normals = np.logical_and(
        norm_N1[:, 0] > np.finfo(np.float32).eps,
        norm_N2[:, 0] > np.finfo(np.float32).eps,
    )
    normalized_N1 = np.zeros_like(N1_match_interpolated)
    normalized_N2 = np.zeros_like(target_normals)
    np.divide(N1_match_interpolated, norm_N1, out=normalized_N1, where=norm_N1 > 0)
    np.divide(target_normals, norm_N2, out=normalized_N2, where=norm_N2 > 0)

    dot_product = np.einsum('ij,ij->i', normalized_N1, normalized_N2)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure the dot product is in the valid range for arccos
    rad_angles = np.arccos(dot_product)
    deg_angles = np.degrees(rad_angles)
    is_distance_threshold = sqrD <= dDISTANCE_THRESHOLD_SQRD
    angle_thresholds = np.full(deg_angles.shape, dANGLE_THRESHOLD_DEGREES)

    is_deg_threshold = np.logical_and(valid_normals, deg_angles <= angle_thresholds)
    if flip_vertex_normal:
        deg_angles_mirror = 180 - deg_angles
        is_deg_threshold = np.logical_or(is_deg_threshold, deg_angles_mirror <= angle_thresholds)

    Matched = np.logical_and(is_distance_threshold, is_deg_threshold)    
    return (Matched, W2, sqrD) if return_distances else (Matched, W2)


def _union_find_components(vertex_count, edges):
    """Return connected-component labels without scipy.sparse.csgraph."""
    parent = np.arange(vertex_count, dtype=np.int64)
    rank = np.zeros(vertex_count, dtype=np.uint8)

    def find(vertex):
        root = vertex
        while parent[root] != root:
            root = parent[root]
        while parent[vertex] != vertex:
            next_vertex = parent[vertex]
            parent[vertex] = root
            vertex = next_vertex
        return root

    for left, right in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        left_root = find(int(left))
        right_root = find(int(right))
        if left_root == right_root:
            continue
        if rank[left_root] < rank[right_root]:
            parent[left_root] = right_root
        elif rank[left_root] > rank[right_root]:
            parent[right_root] = left_root
        else:
            parent[right_root] = left_root
            rank[left_root] += 1

    roots = np.fromiter((find(vertex) for vertex in range(vertex_count)),
                        dtype=np.int64, count=vertex_count)
    _roots, labels = np.unique(roots, return_inverse=True)
    return len(_roots), labels.astype(np.int64, copy=False)


def _connected_components(graph):
    """Use SciPy when available, with a dependency-light fallback."""
    try:
        return sp.sparse.csgraph.connected_components(graph, directed=False)
    except (ImportError, ModuleNotFoundError):
        coo = graph.tocoo()
        edges = np.column_stack((coo.row, coo.col))
        return _union_find_components(graph.shape[0], edges)


def find_vertex_merge_map(vertices, distance):
    """Map nearby vertices to non-destructive, transitively merged clusters."""
    vertices = np.asarray(vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
        raise ValueError("Cannot virtually merge invalid vertex coordinates")
    if not np.isfinite(distance) or distance < 0:
        raise ValueError("Virtual merge distance must be finite and non-negative")
    vertex_count = len(vertices)
    identity = np.arange(vertex_count, dtype=np.int64)
    if vertex_count == 0 or distance <= 0:
        return identity
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Cannot virtually merge vertices with non-finite coordinates")

    pairs = cKDTree(vertices).query_pairs(float(distance), output_type='ndarray')
    if pairs.size == 0:
        return identity

    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    rows = np.hstack((pairs[:, 0], pairs[:, 1]))
    cols = np.hstack((pairs[:, 1], pairs[:, 0]))
    graph = sp.sparse.coo_matrix(
        (np.ones(rows.size, dtype=np.uint8), (rows, cols)),
        shape=(vertex_count, vertex_count),
    ).tocsr()
    _component_count, labels = _connected_components(graph)
    return labels.astype(np.int64, copy=False)


def _collapse_vertices_for_inpainting(V2, F2, W2, Matched, merge_map):
    """Create the temporary welded arrays used only by the weight solver."""
    cluster_count = int(merge_map.max()) + 1 if len(merge_map) else 0
    if cluster_count == len(V2):
        return V2, F2, W2, Matched

    counts = np.bincount(merge_map, minlength=cluster_count).astype(np.float64)
    merged_vertices = np.zeros((cluster_count, 3), dtype=np.float64)
    np.add.at(merged_vertices, merge_map, V2)
    merged_vertices /= counts[:, np.newaxis]

    matched_counts = np.bincount(
        merge_map, weights=Matched.astype(np.float64), minlength=cluster_count
    )
    merged_matched = matched_counts > 0
    merged_weights = np.zeros((cluster_count, W2.shape[1]), dtype=np.float64)
    np.add.at(merged_weights, merge_map[Matched], W2[Matched])
    merged_weights[merged_matched] /= matched_counts[merged_matched, np.newaxis]

    merged_faces = merge_map[F2]
    nondegenerate = np.logical_and.reduce((
        merged_faces[:, 0] != merged_faces[:, 1],
        merged_faces[:, 1] != merged_faces[:, 2],
        merged_faces[:, 2] != merged_faces[:, 0],
    ))
    merged_faces = merged_faces[nondegenerate]
    if len(merged_faces):
        # Overlapping loose parts can produce identical faces after the virtual
        # weld. Count each geometric face only once in the Laplacian.
        canonical_faces = np.sort(merged_faces, axis=1)
        _unique, unique_indices = np.unique(
            canonical_faces, axis=0, return_index=True
        )
        merged_faces = merged_faces[np.sort(unique_indices)]

    return merged_vertices, merged_faces, merged_weights, merged_matched


@dataclass
class InpaintingDomain:
    weights: np.ndarray
    matched: np.ndarray
    merge_map: np.ndarray
    laplacian: object
    mass: np.ndarray
    rejected: np.ndarray


class InpaintingError(ValueError):
    def __init__(self, message, rejected=None):
        super().__init__(message)
        self.rejected = rejected


def prepare_inpainting(V2, F2, W2, Matched, point_cloud, virtual_merge_distance=0.0):
    """Build the actual solver graph, shared by solving and loose-part selection."""
    V = np.asarray(V2, dtype=np.float64)
    F = np.asarray(F2)
    W = np.asarray(W2, dtype=np.float64)
    matched = np.asarray(Matched, dtype=bool)
    if (V.ndim != 2 or V.shape[1] != 3 or not len(V)
            or not np.all(np.isfinite(V))):
        raise InpaintingError("Target vertices must be a nonempty finite N by 3 array")
    if (F.ndim != 2 or F.shape[1] != 3 or not np.issubdtype(F.dtype, np.integer)
            or np.any(F < 0) or np.any(F >= len(V))):
        raise InpaintingError("Target contains invalid triangle indices")
    if (W.ndim != 2 or W.shape[0] != len(V) or not W.shape[1]
            or not np.all(np.isfinite(W)) or matched.shape != (len(V),)):
        raise InpaintingError("Target weights or match mask are invalid")
    if not np.isfinite(virtual_merge_distance) or virtual_merge_distance < 0:
        raise InpaintingError("Virtual merge distance must be finite and non-negative")
    # Inpainting does not change fully matched input. Seam synchronization is
    # deliberately separate and also operates on fully matched meshes.
    if np.all(matched):
        return InpaintingDomain(W.copy(), matched, np.arange(len(V)), None,
                                np.zeros(len(V)), np.zeros(len(V), dtype=bool))
    merge_map = find_vertex_merge_map(V, virtual_merge_distance)
    V, F, W, matched = _collapse_vertices_for_inpainting(V, F, W, matched, merge_map)
    n = len(V)
    if np.all(matched) or not np.any(matched):
        return InpaintingDomain(W, matched, merge_map, None, np.zeros(n),
                                (~matched)[merge_map])
    use_points = point_cloud and n == len(V2)
    try:
        if use_points and n >= 3:
            L, M = robust_laplacian.point_cloud_laplacian(V, n_neighbors=min(30, n - 1))
            mass = M.diagonal()
        else:
            # Removed faces can leave isolated vertices. Keep their constraints,
            # but do not feed unused vertices to the native surface builder.
            F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])]
            if len(F):
                _, keep = np.unique(np.sort(F, axis=1), axis=0, return_index=True)
                F = F[np.sort(keep)]
            active = np.unique(F)
            L = sp.sparse.csr_matrix((n, n), dtype=np.float64)
            mass = np.zeros(n, dtype=np.float64)
            if len(active):
                remap = np.full(n, -1, dtype=np.int64)
                remap[active] = np.arange(len(active))
                local_L, local_M = robust_laplacian.mesh_laplacian(V[active], remap[F])
                coo = local_L.tocoo()
                L = sp.sparse.csr_matrix((coo.data, (active[coo.row], active[coo.col])), shape=(n, n))
                mass[active] = local_M.diagonal()
        L = L.astype(np.float64).tocsr()
        if np.any(~np.isfinite(L.data)) or np.any(~np.isfinite(mass)) or np.any(mass < 0):
            raise InpaintingError("Inpainting produced an invalid Laplacian or mass matrix")
        graph = L.copy()
        graph.setdiag(0)
        graph.eliminate_zeros()
        count, labels = _connected_components(graph)
        anchored = np.zeros(count, dtype=bool)
        anchored[labels[matched]] = True
        rejected = ~anchored[labels]
        rejected |= (~matched) & (mass <= 0)
        return InpaintingDomain(W, matched, merge_map, L, mass, rejected[merge_map])
    except (RuntimeError, ValueError) as error:
        raise InpaintingError(str(error)) from error


def solve_inpainting(domain, skip_rejected=False):
    """Solve constrained vertices; optionally leave unsupported output unused.

    Partial transfers discard rejected rows at write time. Those rows must not
    supply constraints or participate in the supported linear system.
    """
    if np.any(domain.rejected) and not skip_rejected:
        raise InpaintingError("Loose parts without a matched vertex remain in the solver graph",
                              domain.rejected)
    W = domain.weights.copy()
    rejected = np.zeros(len(W), dtype=bool)
    rejected[domain.merge_map[domain.rejected]] = True
    active = np.flatnonzero(~rejected)
    unknown = np.flatnonzero(~domain.matched[active])
    if len(unknown):
        L = domain.laplacian[active][:, active]
        mass = domain.mass[active]
        inv_mass = np.zeros_like(mass)
        np.divide(1.0, mass, out=inv_mass, where=mass > 0)
        Q = L + L @ sp.sparse.diags(inv_mass) @ L
        known = np.flatnonzero(domain.matched[active])
        rhs = -(Q[unknown][:, known] @ W[active[known]])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', sp.sparse.linalg.MatrixRankWarning)
                solution = sp.sparse.linalg.spsolve(Q[unknown][:, unknown].tocsc(), rhs)
            W[active[unknown]] = np.asarray(solution).reshape(len(unknown), W.shape[1])
        except (RuntimeError, ValueError, sp.sparse.linalg.MatrixRankWarning) as error:
            raise InpaintingError("Constrained weight solve failed: " + str(error)) from error
    with np.errstate(over='ignore'):
        result = W[domain.merge_map].astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise InpaintingError("Inpainting produced non-finite weights")
    return result


def inpaint(V2, F2, W2, Matched, point_cloud, virtual_merge_distance=0.0):
    """
    Inpaint weights for all the vertices on the target mesh for which  we didnt 
    find a good match on the source (i.e. Matched[i] == False).

    Args:
        V2: #V2 by 3 target mesh vertices
        F2: #F2 by 3 target mesh triangles indices
        W2: #V2 by num_bones, where W2[i,:] are skinning weights copied directly from source using closest point method
        Matched: #V2 array of bools, where Matched[i] is True if we found a good match for vertex i on the source mesh
        virtual_merge_distance: vertices within this world-space distance are
            temporarily welded for the solve; the Blender mesh is not changed

    Returns:
        W_inpainted: #V2 by num_bones, final skinning weights where we inpainted weights for all vertices i where Matched[i] == False
    """
    
    try:
        domain = prepare_inpainting(V2, F2, W2, Matched, point_cloud, virtual_merge_distance)
        return True, solve_inpainting(domain)
    except (RuntimeError, ValueError, TypeError):
        return False, np.asarray(W2, dtype=np.float32)



def limit_mask(weights, adjacency_matrix, dilation_repeat=5, limit_num=4):
    if weights.shape[1] <= limit_num: return np.zeros_like(weights)
    
    count = np.count_nonzero(weights, axis=1)
    to_limit = count > limit_num
    k = weights.shape[1] - limit_num
    weights_inds = np.argpartition(weights, kth=k, axis=1)[:, :k]
    row_indices = np.arange(weights.shape[0])[:, None]
    erode_mask = np.zeros_like(weights, dtype=bool)
    erode_mask[row_indices, weights_inds] = True
    erode_mask = np.logical_and(erode_mask, to_limit[:, np.newaxis])
    erode_mask = sp.sparse.csr_array(erode_mask).astype(np.float32)
    adj_mat = adjacency_matrix
    degrees = adj_mat.sum(axis=1)
    smooth_mat = (1/degrees[:, np.newaxis]) * adj_mat
    for _ in range(dilation_repeat):
        avg_weights = smooth_mat @ erode_mask
        erode_mask = erode_mask.maximum(avg_weights)
    
    return erode_mask.toarray()


def smooth_weigths(verts, weights, matched, adjacency_matrix, adjacency_list, num_smooth_iter_steps, smooth_alpha, distance_threshold):
    not_matched = ~matched
    VIDs_to_smooth = np.zeros(verts.shape[0], dtype=bool)

    def get_points_within_distance(V, VID, distance=distance_threshold):
        """
        Get all neighbours of vertex VID within dDISTANCE_THRESHOLD
        """
        queue = []
        queue.append(VID)
        while len(queue) != 0:
            vv = queue.pop()
            if vv < len(adjacency_list):
                neigh = adjacency_list[vv]
                for nn in neigh:
                    if ~VIDs_to_smooth[nn] and np.linalg.norm(V[VID,:]-V[nn]) < distance:
                        VIDs_to_smooth[nn] = True
                        if nn not in queue:
                            queue.append(nn)

    for i in range(verts.shape[0]):
        if not_matched[i]:
            get_points_within_distance(verts, i, distance_threshold)
            
    adj_mat = adjacency_matrix.astype(np.float32)
    degrees = adj_mat.sum(axis=1)
    
    smooth_mat = sp.sparse.diags(1/degrees) @ adj_mat
    weights_smoothed = sp.sparse.csr_array(weights)
    for _ in range(num_smooth_iter_steps):
        weights_smoothed = (1 - smooth_alpha) * weights_smoothed + smooth_alpha * (smooth_mat @ weights_smoothed)
        weights_smoothed[~VIDs_to_smooth] = weights[~VIDs_to_smooth]
    return weights_smoothed.todense()
            
