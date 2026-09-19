# Robust Weight Transfer for Blender

A Blender addon for getting smooth weight transfer results with just one click! 

Robust Weight Transfer can successfully transfer weights from the body, without needing additional weight painting work like smoothing in problem areas (between legs, between chest, armpits).

Weight transfer code is based on https://github.com/rin-23/RobustSkinWeightsTransferCode/tree/main

You can find a tutorial on the Jinxxy page of the addon https://jinxxy.com/SentFromSpaceVR/robust-weight-transfer

## Development

### Installing Dependencies

Blender 5.2 users can normally press **Install Dependencies** in the add-on
panel. Dependencies are installed into the add-on's own `deps` directory from
precompiled wheels; Blender's bundled Python installation is not modified.

#### Option 1
When using the same Python version that Blender uses, you can install the dependencies with:
```
python -m pip install --only-binary=:all: --no-deps --target deps -r requirements.txt
```
#### Option 2
With this `pip` command you can control which platform and for which Python version wheels should be downloaded:
```
python -m pip download --platform win_amd64 --python-version 313 --only-binary=:all: --no-deps -d whl -r requirements.txt
```
The dependency wheels get downloaded into the `whl` directory. From here you can unzip the content of the wheels into the `deps` directory

### Changes to the original addon/repository:
- Allow flipped normals 
    - Handles solid meshes when vertex normals show into opposite direction of the vertex normals on the body
- Remesh using Robust Laplacian's point cloud Laplacian (During Point mode)
    - The Blender 5.2 build uses SciPy for constrained inpainting and component detection, avoiding the unavailable Windows `libigl` wheel
- Update the add-on metadata for Blender 5.2 and publish this build as version 1.2 (`bl_info['version'] == (1, 2, 0)`).
- Install pinned, binary-only dependencies into the add-on's local `deps` directory, without modifying Blender's bundled Python environment.
- Replace the original `libigl` closest-surface, connected-components, and constrained-inpainting calls with Blender's BVH tree and SciPy implementations.
- Add validation for empty or invalid meshes, missing transferable groups, degenerate faces, zero-length normals, singular solves, and disconnected targets.
- Add non-destructive **Virtual Merge by Distance** for loose parts; only temporary solver arrays are merged, so the original mesh topology and attributes are preserved.
- Reload cached helper modules during in-place ZIP upgrades so Blender does not mix files from different add-on versions.

### Blender 5.2 performance

The add-on uses Blender's built-in BVH tree for closest-surface matching.
The tree is built once per source mesh and reused across all selected targets,
avoiding the exhaustive target-vertex by source-triangle search.

### Partial reweighting

Enable **Settings → Vertex Mapping → Partial Reweight** to replace weights only
near the source. **Max Distance** serves both as the direct-match tolerance and
the outer limit of the partial transfer. Distance is measured in world space to
the nearest point on the source's triangle surface, respecting **Use Deformed
Source** and **Use Deformed Target**. Normal matching rules still determine which
vertices supply direct constraints for inpainting.

**Falloff Width** is the percentage of Max Distance over which the transfer fades
smoothly into existing weights. Its default of **20%** fully replaces weights
through the inner 80% of the range, then fades to no change at the outer limit.
Set it to 100% to fade across the whole range, or 0% for a hard cutoff that includes
the boundary. A Max Distance of zero transfers only at exact surface hits.

Vertices beyond the range retain their exact existing weights and group
memberships, including tiny weights and unweighted vertices. In-range parts that
cannot reach a usable constraint in the Point/Surface solver graph are preserved
and reported per object. An entirely out-of-range target is a successful no-op.
Virtual Merge and compatible cross-object solves can still rescue unsupported
parts. Select Rejected Loose Parts and rejected-weight coloring exclude vertices
with zero distance strength.

Distance strength multiplies the existing Transfer Mask on single targets and
also works with **Apply to all Selected Objects** (which continues to ignore
per-object transfer masks). Locked and non-transferred groups stay protected.
Smoothing and influence limiting operate on supported transferred candidates
before blending. As with transfer masks, the final blend can contain more groups
than the configured limit; it is not pruned again after blending. Seam clusters
containing preserved or partially blended vertices are skipped to protect the
falloff.

Partial Reweight defaults to off and Reset to Defaults restores the 20% falloff.
Standalone **Inpaint** remains independent of these distance controls. Invalid
input or numerical failures still cancel the entire transfer before weights are
written.

### Loose parts without modifying the mesh

**Virtual Merge by Distance** is available under Settings. When enabled,
nearby vertices are welded only in the temporary mesh used by the inpainting
solver, then the computed weights are mapped back to the original vertices.
The actual mesh topology, split normals, UV seams, shape keys, and modifiers
remain unchanged. Use the smallest distance that connects the intended seams;
an unnecessarily large value can connect unrelated nearby surfaces.

Virtual Merge can rescue a rejected loose part only when its temporary solver
component reaches at least one matched vertex. It cannot infer weights for an
entirely unconstrained component. The solver now checks this explicitly instead
of relying on a numerical singular-matrix warning, which could previously let
unmatched parts pass with all-zero weights. **Select Rejected Loose Parts** uses
the same solver graph, including Point mode's generated connections.

When a virtual weld actually occurs, both Point and Surface modes use the
temporary welded surface. Fully matched meshes bypass inpainting. Virtual Merge
alone does not guarantee equal final seam weights after smoothing or limiting.

### Keep seams together under bone scaling

Enable **Settings → Synchronize Seam Weights** to give nearby open borders of
different loose parts identical final deform weights. This is independent of
Enable Smoothing and Virtual Merge, and works even when every vertex matched.

- **Seam Distance** defaults to `0.0001` world-space units; zero matches exact
  duplicates. Detection uses the original, undeformed mesh positions, so a seam
  can still be found when the current pose has pulled it apart.
- By default, synchronization stays within each target mesh. Enable **Apply to
  all Selected Objects** and **Across Selected Objects** to include seams between
  compatible selected targets. Groups are aligned by name, not group index.
- With **Virtual Merge** also enabled, compatible targets share a temporary
  inpainting solve. A matched object can then supply weights to a neighboring
  object with no matches of its own. Virtual Merge Distance controls solver
  connections; Seam Distance independently controls final seam synchronization.

Seam reconciliation runs after smoothing, influence limiting, and transfer-mask
blending. It averages deform weights, preserves fixed contributions, chooses a
common set of influences when limiting is enabled, and normalizes the remaining
weight budget. Vertices away from detected seams receive no additional smoothing.

Locked groups and non-transferred weights remain protected. Clusters with
different protected deform weights, partial transfer masks, or an infeasible
weight budget/group limit are skipped and reported. Mask-zero vertices are not
written. As before, transfer masks apply to single-target transfers; the existing
Apply to all Selected Objects mode does not use per-object transfer masks.
Standalone **Inpaint** respects its binary mask and inversion, and skips seam
clusters containing vertices outside that mask.

Cross-object sharing supports targets bound to the same armature with compatible
armature settings. Unbound targets use the source's deform-group names and can
share with its standard armature setup. Different armatures, differing Preserve
Volume settings, and envelope/masked/multiple-armature setups do not share seam
connections. The add-on does not create or change armature modifiers. Mesh data
must be single-user when seam synchronization is enabled.

The feature does not join meshes, change topology, UVs, normals, shape keys, or
modifiers. It addresses separation caused by differing skin weights; it does not
close existing geometric gaps, stitch mismatched boundary sampling, or compensate
for different shape keys or modifier deformation. Closed overlapping surfaces
and cuts whose borders remain in the same connected part are not detected as
seams. Use a small tolerance to avoid matching unrelated nearby open borders;
proximity clusters are transitive.

### Balance and normalize transferred weights

**Balance L/R Weight Groups** balances the aggregate weight totals of each
transferred deform-bone pair on every target object. It recognizes Blender-style
`.L`/`.R` names and FFXIV-style `_l`/`_r` names. It does not mirror vertices or
move geometry; instead, it redistributes each writable pair's existing per-vertex
pair mass, preserving every affected vertex's total.

**Normalize Weights After Transfer** scales all deform-bone groups on vertices
touched by the transfer so their total is one. It includes pre-existing
target-only deform groups, but never changes non-deform groups. Both options
leave mask-zero and out-of-range Partial Reweight vertices untouched, and neither
changes locked groups. If locked or protected values leave no feasible way to
normalize or balance a pair, the add-on preserves those values and reports a
warning. These options apply to **Transfer Weights** only; Utilities → Inpaint
remains unchanged.
