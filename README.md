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

### VS Code Blender Extension

I recommend using https://github.com/JacquesLucke/blender_vscode during development.

## Academic Work

This Blender addon is based on "Robust Skin Weights Transfer via Weight Inpainting" by Rinat Abdrashitov, Kim Raichstat, Jared Monsen and David Hill published at SIGGRAPH ASIA 2023

You can find the project page here https://www.dgp.toronto.edu/~rinat/projects/RobustSkinWeightsTransfer/index.html

When using the addon for academic work please cite them!

Changes to the original work:
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

### Loose parts without modifying the mesh

**Virtual Merge by Distance** is available under Settings. When enabled,
nearby vertices are welded only in the temporary mesh used by the inpainting
solver, then the computed weights are mapped back to the original vertices.
The actual mesh topology, split normals, UV seams, shape keys, and modifiers
remain unchanged. Use the smallest distance that connects the intended seams;
an unnecessarily large value can connect unrelated nearby surfaces.

The add-on also supports installing an update over an already loaded older
version without requiring Blender to discard cached add-on modules first.
