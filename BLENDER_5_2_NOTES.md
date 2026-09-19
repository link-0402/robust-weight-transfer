# Blender 5.2 compatibility notes

- Add-on metadata now targets Blender 5.2.0.
- Dependency installation uses `pip --target deps` and precompiled wheels only.
- Removed `--user`, `PYTHONUSERBASE`, `--break-system-packages`, and the
  generated constraints file.
- Removed the `libigl` dependency because compatible Windows wheels are not
  published for Blender 5.2's Python. Closest-surface matching, connected
  components, and constrained inpainting now use NumPy/SciPy implementations.
- Updated debug colors from legacy `Mesh.vertex_colors` to
  `Mesh.color_attributes`.
- Fixed registration order, dependency-missing startup, active-object polling,
  and per-target mask settings.

## 1.2.1 corrections

- Replaced the exhaustive NumPy closest-surface search with Blender's native
  BVH tree and reuse one tree for every target in a transfer operation.
- Added stable barycentric interpolation for BVH hits and safe handling for
  degenerate faces and zero-length normals.
- Added clear validation for empty meshes and missing transferable groups.
- Avoided singular inpainting solves when every or no target vertex is matched.
- Corrected topology-modifier detection for Blender 5.2, including Geometry
  Nodes, and ignore modifiers disabled in the viewport.
- Fixed duplicated deformation controls and unrelated armature warnings in the
  sidebar, and restricted Inpaint to supported object modes.

## 1.2.2 virtual merge

- Added an optional **Virtual Merge by Distance** inpainting mode for models
  whose visually connected pieces use separate vertices.
- Nearby vertices are clustered only in temporary solver arrays. The Blender
  mesh, vertex count, normals, UV seams, shape keys, and modifiers are never
  changed.
- When vertices are virtually joined, both Point and Surface modes solve on
  the welded temporary surface for stability and Merge-by-Distance parity.
- The loose-part diagnostic now treats virtual merge clusters as connections.
- Fixed `Select Rejected Loose Parts` passing NumPy booleans to Blender's
  BMesh API, which Blender 5.2 rejects.
- Singular disconnected solves now fail cleanly without emitting SciPy matrix
  rank warnings to Blender's console.

## 1.2.3 installation fix

- Freshly import cached add-on submodules during an in-place ZIP upgrade. Blender's
  legacy installer can otherwise combine the newly copied `__init__.py` with
  an older `weighttransfer.py` module still held in memory, causing missing
  import errors until Blender is restarted.

## Loose-part solver corrections and seam synchronization

- Explicitly reject solver components with unknown vertices and no matched
  constraint. Previously, singular solves could return finite zero weights and
  incorrectly report success, depending on mesh scale.
- Share actual solver-graph preparation with Select Rejected Loose Parts,
  including Point mode and virtual cross-object connections.
- Bound Point mode's neighbor count for small meshes and handle isolated
  clusters left by collapsed faces without discarding known constraints.
- Add opt-in Synchronize Seam Weights, Seam Distance, and Across Selected Objects.
  Final seam weights are reconciled after smoothing, limiting, and mask blending.
- Support shared virtual inpainting solves for compatible selected targets,
  retaining each object's topology and matching group names across objects.
- Preserve locks, non-deform groups, untouched contributions, and transfer masks;
  report seam clusters that cannot be synchronized within those constraints.
- Clear stale weights in writable transferred groups even when an entire output
  column is zero. Honor standalone Inpaint's mask inversion and protected vertices.
- Reload the new helper modules on in-place upgrades, alongside existing helpers.
- Add headless Blender numerical and operator regression tests, including actual
  armature deformation and topology/attribute preservation checks.

## 1.2.1 final-weight controls

- Add opt-in locked-safe deform-weight normalization for transfer-touched vertices.
- Add opt-in aggregate L/R balancing for transferred `.L`/`.R` and `_l`/`_r`
  deform-bone pairs.
