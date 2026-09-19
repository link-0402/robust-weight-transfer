"""Stage transfer results before synchronization and Blender weight writes."""
import numpy as np

from . import util, seams, weighttransfer

WRITE_THRESHOLD = 1e-5
WEIGHT_TOLERANCE = 1e-7


def armature_key(obj, fallback=None):
    modifiers = [m for m in obj.modifiers if m.type == 'ARMATURE' and m.show_viewport]
    if not modifiers:
        return ('RIG', fallback.as_pointer(), False) if fallback else ('UNBOUND',)
    if len(modifiers) == 1 and modifiers[0].object:
        mod = modifiers[0]
        if (not mod.use_vertex_groups or mod.use_bone_envelopes or mod.vertex_group
                or mod.use_multi_modifier):
            # These settings can produce different motion for identical weights.
            return ('OBJECT', obj.as_pointer())
        return ('RIG', mod.object.as_pointer(), mod.use_deform_preserve_volume)
    # Ambiguous deformation setups may be processed, but never shared.
    return ('OBJECT', obj.as_pointer())


def source_armature(obj):
    mods = [m for m in obj.modifiers if m.type == 'ARMATURE' and m.object and m.show_viewport]
    return mods[0].object if len(mods) == 1 else None


def make_target(obj, depsgraph, settings):
    if settings.use_deformed_target and util.has_modifier(obj, *util.TOPOLOGY_MODS):
        raise ValueError(f'{obj.name}: disable topology-changing modifiers or Use Deformed Target')
    if settings.seam_sync and obj.data.users > 1:
        raise ValueError(f'{obj.name}: make the mesh single-user before synchronizing seams')
    evaluated = obj.evaluated_get(depsgraph) if settings.use_deformed_target else obj
    vertices, triangles, normals = util.get_obj_arrs_world(evaluated)
    if len(vertices) != len(obj.data.vertices):
        raise ValueError(f'{obj.name}: evaluated topology differs from the original mesh')
    if not len(vertices) or not len(triangles):
        raise ValueError(f'{obj.name}: target must contain vertices and faces')
    return dict(obj=obj, vertices=vertices, triangles=triangles, normals=normals)


def inpaint_mask(obj):
    settings = obj.robust_weight_transfer_settings
    if not util.is_group_valid(obj.vertex_groups, settings.inpaint_group):
        return np.zeros(len(obj.data.vertices), dtype=bool)
    mask = util.get_group_arr(obj, settings.inpaint_group) > settings.inpaint_threshold
    return ~mask if settings.inpaint_group_invert else mask


def distance_strength(squared_distances, radius, falloff_percent):
    """Smoothly fade over the outer fraction of a world-space surface radius."""
    distances = np.asarray(squared_distances, dtype=np.float64)
    if (not np.all(np.isfinite(distances)) or np.any(distances < 0)
            or not np.isfinite(radius) or radius < 0
            or not np.isfinite(falloff_percent) or not 0 <= falloff_percent <= 100):
        raise ValueError('Partial reweight distances and falloff must be finite and non-negative')
    if radius == 0 or falloff_percent == 0:
        return (distances <= radius ** 2).astype(np.float64)
    width = radius * falloff_percent / 100
    t = np.clip((np.sqrt(distances) - (radius - width)) / width, 0, 1)
    return 1 - t * t * (3 - 2 * t)


def match_target(target, source_vertices, source_triangles, source_normals,
                 source_weights, settings, surface_bvh):
    matched, weights, distances = weighttransfer.find_matches_closest_surface(
        source_vertices, source_triangles, source_normals,
        target['vertices'], target['normals'], source_weights,
        settings.max_distance ** 2, np.degrees(settings.max_normal_angle_difference),
        settings.flip_vertex_normal, surface_bvh, return_distances=True)
    if not settings.apply_to_selected:
        matched &= ~inpaint_mask(target['obj'])
    target.update(matched=matched, weights=weights)
    if settings.partial_reweight:
        target['strength'] = distance_strength(
            distances, settings.max_distance, settings.partial_reweight_falloff)


def batches(targets, across, fallback=None):
    result = {}
    for index, target in enumerate(targets):
        key = armature_key(target['obj'], fallback) if across else ('OBJECT', index)
        result.setdefault(key, []).append(target)
    return list(result.values())


def prepare_batch(batch, settings):
    offsets = np.cumsum([0] + [len(t['vertices']) for t in batch])
    domain = weighttransfer.prepare_inpainting(
        np.concatenate([t['vertices'] for t in batch]),
        np.concatenate([t['triangles'] + offset for t, offset in zip(batch, offsets)]),
        np.concatenate([t['weights'] for t in batch]),
        np.concatenate([t['matched'] for t in batch]),
        settings.inpaint_mode == 'POINT',
        settings.virtual_merge_distance if settings.virtual_merge else 0.0,
    )
    return domain, offsets


def solve_targets(targets, settings, fallback=None, partial=False):
    across = (settings.seam_sync and settings.seam_sync_across_objects
              and settings.apply_to_selected and settings.virtual_merge)
    for batch in batches(targets, across, fallback):
        if partial and not any(np.any(t['strength'] > 0) for t in batch):
            for target in batch:
                target['supported'] = np.zeros(len(target['vertices']), dtype=bool)
            continue
        try:
            domain, offsets = prepare_batch(batch, settings)
            if np.any(domain.rejected) and not partial:
                affected = [t['obj'].name for t, start, end in zip(batch, offsets, offsets[1:])
                            if np.any(domain.rejected[start:end])]
                raise ValueError('No matched constraint for loose parts in ' + ', '.join(affected)
                                 + '. Adjust Virtual Merge Distance or use Select Rejected Loose Parts.')
            solved = weighttransfer.solve_inpainting(domain, skip_rejected=partial)
        except (RuntimeError, ValueError) as error:
            names = ', '.join(t['obj'].name for t in batch)
            raise ValueError(f'Inpainting failed on {names}: {error}') from error
        for target, start, end in zip(batch, offsets, offsets[1:]):
            target['weights'] = solved[start:end].copy()
            if partial:
                target['supported'] = ~domain.rejected[start:end]


def process_weights(target, settings):
    """Smooth/limit candidates on the supported mesh before mask blending."""
    obj = target['obj']
    active = np.flatnonzero(target.get('supported', np.ones(len(target['vertices']), bool)))
    if not len(active):
        return
    weights = target['weights'][active].copy()
    adjacency = util.get_mesh_adjacency_matrix_sparse(obj.data, include_self=True)[active][:, active]
    if settings.smoothing_enable:
        neighbors = [[int(j) for j in adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i + 1]]
                      if j != i]
                     for i in range(len(active))]
        weights = np.asarray(weighttransfer.smooth_weigths(
            target['vertices'][active], weights, target['matched'][active], adjacency,
            neighbors, settings.smoothing_repeat, settings.smoothing_factor, settings.max_distance))
    if settings.enforce_four_bone_limit:
        weights[weights <= 0.0001] = 0
        weights *= 1 - weighttransfer.limit_mask(weights, adjacency, limit_num=settings.num_limit_groups)
        weights[weights <= 0.0001] = 0
    target['weights'][active] = weights


def stage_weights(target, names, included, apply_mask=False):
    """Reconcile by name, including zero columns, without writing to Blender."""
    obj = target['obj']
    current = {g.name: w.copy() for g, w in zip(obj.vertex_groups, util.get_groups_arr(obj).T)}
    final = {name: w.copy() for name, w in current.items()}
    mask = np.ones(len(obj.data.vertices))
    object_settings = obj.robust_weight_transfer_settings
    if apply_mask and util.is_group_valid(obj.vertex_groups, object_settings.vertex_group):
        mask = util.get_group_arr(obj, object_settings.vertex_group).astype(np.float64)
        if object_settings.vertex_group_invert:
            mask = 1 - mask
    if 'strength' in target:
        mask *= target['strength'] * target['supported']
    writable = set()
    for index, name in enumerate(names):
        group = obj.vertex_groups.get(name)
        if not included[index] or (group and group.lock_weight):
            continue
        old = current.get(name, np.zeros(len(mask)))
        weights = np.asarray(target['weights'][:, index], dtype=np.float64)
        if not np.all(np.isfinite(weights)):
            raise ValueError(f'{obj.name}: non-finite output weights')
        final[name] = np.clip((1 - mask) * old + mask * weights, 0, 1).astype(np.float32)
        final[name][(mask > 0) & (final[name] < 1e-5)] = 0
        writable.add(name)
    target.update(final=final, writable=writable, protected=mask < 1,
                  write_vertices=mask > 0)


def synchronize_targets(targets, settings, deform_names, fallback=None):
    """Synchronize complete deform vectors, retaining target-only contributions."""
    if not settings.seam_sync:
        return 0, 0, 0
    names = set(deform_names)
    for target in targets:
        obj = target['obj']
        target['deform_names'] = set(deform_names) | {
            g.name for g in obj.vertex_groups if util.is_vertex_group_deform_bone(obj, g.name)}
        names.update(target['deform_names'])
    names = sorted(names)
    if not names:
        return 0, 0, 0
    across = settings.seam_sync_across_objects and settings.apply_to_selected
    compatibility_keys = {}
    positions, boundaries, components, compatibility = [], [], [], []
    weights, writable, protected, confidence = [], [], [], []
    component_offset = 0
    offsets = [0]
    for index, target in enumerate(targets):
        obj = target['obj']
        # Base-mesh coordinates deliberately ignore pose and shape-key mix.
        verts, triangles, _ = util.get_obj_arrs_world(obj)
        edges = np.empty((len(obj.data.edges), 2), dtype=np.int64)
        obj.data.edges.foreach_get('vertices', edges.ravel())
        boundary, labels = seams.boundary_components(len(verts), triangles, edges)
        key = armature_key(obj, fallback) if across else ('OBJECT', index)
        key_id = compatibility_keys.setdefault(key, len(compatibility_keys))
        positions.append(verts)
        boundaries.append(boundary)
        components.append(labels + component_offset)
        compatibility.append(np.full(len(verts), key_id))
        component_offset += int(labels.max()) + 1
        weights.append(np.column_stack([
            target['final'].get(name, np.zeros(len(verts)))
            if name in target['deform_names'] else np.zeros(len(verts)) for name in names]))
        writable.append(np.tile([name in target['writable'] and name in target['deform_names']
                                 for name in names], (len(verts), 1)))
        protected.append(target['protected'])
        confidence.append(target.get('matched', np.ones(len(verts), dtype=bool)).astype(np.float64))
        offsets.append(offsets[-1] + len(verts))
    clusters, excluded = seams.find_seam_clusters(
        np.concatenate(positions), np.concatenate(boundaries), np.concatenate(components),
        np.concatenate(compatibility), settings.seam_distance)
    final, synced, skipped = seams.synchronize_weights(
        np.concatenate(weights), np.concatenate(writable), np.concatenate(protected), clusters,
        settings.num_limit_groups if settings.enforce_four_bone_limit else None,
        np.concatenate(confidence))
    # Only copy columns in the original transfer scope. Protected groups remain
    # untouched even if another object transfers that same group name.
    for target, start, end in zip(targets, offsets, offsets[1:]):
        for column, name in enumerate(names):
            if name in target['writable'] and name in target['deform_names']:
                target['final'][name] = final[start:end, column]
    return synced, skipped, excluded if across else 0


def target_deform_names(target, transferred_deform_names=()):
    """Return target deform names, including newly staged source groups."""
    obj = target['obj']
    names = [group.name for group in obj.vertex_groups]
    names.extend(name for name in transferred_deform_names if name not in names)
    return [name for name in names if util.is_vertex_group_deform_bone(obj, name)]


def lr_group_pairs(names):
    """Find complete Blender and FFXIV-style left/right name pairs."""
    available = set(names)
    pairs = []
    for name in sorted(available):
        if name.endswith('.L'):
            other = name[:-2] + '.R'
        elif name.endswith('_l'):
            other = name[:-2] + '_r'
        else:
            continue
        if other in available:
            pairs.append((name, other))
    return pairs


def _set_normalized_values(values, movable, fixed):
    """Return a locked-safe normalized deform vector, or None if impossible."""
    result = np.asarray(values, dtype=np.float64).copy()
    fixed_total = result[fixed].sum()
    if fixed_total > 1 + WEIGHT_TOLERANCE:
        return None
    remaining = max(0.0, 1.0 - fixed_total)
    candidate = result[movable]
    total = candidate.sum()
    if total <= WEIGHT_TOLERANCE:
        if remaining > WEIGHT_TOLERANCE:
            return None
        result[movable] = 0
        return result
    candidate *= remaining / total
    candidate[candidate < WRITE_THRESHOLD] = 0
    total = candidate.sum()
    if total <= WEIGHT_TOLERANCE:
        if remaining > WEIGHT_TOLERANCE:
            return None
        result[movable] = 0
        return result
    candidate *= remaining / total
    # Keep the persisted total stable despite floating-point rounding.
    largest = int(np.argmax(candidate))
    candidate[largest] += remaining - candidate.sum()
    result[movable] = candidate
    return result


def normalize_target_deform_weights(target, transferred_deform_names):
    """Normalize writable deform weights on vertices touched by the transfer."""
    names = target_deform_names(target, transferred_deform_names)
    if not names:
        return 0
    obj = target['obj']
    vertex_count = len(obj.data.vertices)
    for name in names:
        target['final'].setdefault(name, np.zeros(vertex_count, dtype=np.float32))
    locked = np.array([bool((group := obj.vertex_groups.get(name)) and group.lock_weight)
                       for name in names])
    movable = ~locked
    target['writable'].update(name for name, can_write in zip(names, movable) if can_write)
    values = np.column_stack([target['final'][name] for name in names])
    failed = 0
    for index in np.flatnonzero(target['write_vertices']):
        normalized = _set_normalized_values(values[index], movable, locked)
        if normalized is None:
            failed += 1
            continue
        values[index] = normalized
    for column, name in enumerate(names):
        target['final'][name] = values[:, column].astype(np.float32)
    return failed


def balance_target_lr_weights(target, transferred_deform_names):
    """Move writable L/R pair mass toward equal group totals without changing vertex sums."""
    deform_names = set(target_deform_names(target, transferred_deform_names))
    transferred = set(transferred_deform_names) & deform_names
    limited = 0
    for left, right in lr_group_pairs(transferred):
        left_values = target['final'].get(left)
        right_values = target['final'].get(right)
        if left_values is None or right_values is None:
            continue
        difference = float(left_values.sum(dtype=np.float64)
                           - right_values.sum(dtype=np.float64))
        if abs(difference) <= WEIGHT_TOLERANCE:
            continue
        # Both columns must be writable: moving mass between only one side would
        # break the per-vertex normalization and could change a protected group.
        if left not in target['writable'] or right not in target['writable']:
            limited += 1
            continue
        if difference > 0:
            donor, receiver = left_values, right_values
        else:
            donor, receiver = right_values, left_values
        eligible = target['write_vertices'] & (donor > 0)
        capacity = float(donor[eligible].sum(dtype=np.float64))
        amount = min(abs(difference) / 2, capacity)
        if amount <= WEIGHT_TOLERANCE:
            limited += 1
            continue
        indices = np.flatnonzero(eligible)
        changes = donor[indices].astype(np.float64) * (amount / capacity)
        donor[indices] -= changes.astype(donor.dtype)
        receiver[indices] += changes.astype(receiver.dtype)
        # A pair cannot become exactly balanced when its protected contribution
        # exceeds every writable donor value available to offset it.
        if amount + WEIGHT_TOLERANCE < abs(difference) / 2:
            limited += 1
    return limited


def postprocess_transfer_targets(targets, settings, transferred_deform_names):
    """Apply optional final transfer corrections before Blender writes weights."""
    counts = {'normalization_failed': 0, 'balance_limited': 0}
    for target in targets:
        if settings.normalize_weights_after_transfer:
            counts['normalization_failed'] += normalize_target_deform_weights(
                target, transferred_deform_names)
        if settings.balance_lr_weight_groups:
            counts['balance_limited'] += balance_target_lr_weights(
                target, transferred_deform_names)
        # Balancing preserves each pair's total, but a final normalization pass
        # accounts for any values that fall below the writer's removal cutoff.
        if settings.normalize_weights_after_transfer and settings.balance_lr_weight_groups:
            normalize_target_deform_weights(target, transferred_deform_names)
    return counts


def write_targets(targets):
    for target in targets:
        obj = target['obj']
        vertices = target['write_vertices']
        for name in sorted(target['writable']):
            w = target['final'][name]
            group = obj.vertex_groups.get(name)
            if group is None:
                if not np.any(w[vertices] >= WRITE_THRESHOLD):
                    continue
                group = obj.vertex_groups.new(name=name)
            if group.lock_weight:
                continue
            for index in np.flatnonzero(vertices & (w >= WRITE_THRESHOLD)):
                group.add([int(index)], float(w[index]), 'REPLACE')
            remove = np.flatnonzero(vertices & (w < WRITE_THRESHOLD)).tolist()
            if remove:
                group.remove(remove)


def report_seams(operator, counts):
    synced, skipped, excluded = counts
    if synced:
        operator.report({'INFO'}, f'Synchronized {synced} seam clusters')
    if skipped or excluded:
        operator.report({'WARNING'}, f'Skipped {skipped} protected or infeasible seam clusters; '
                        f'excluded {excluded} border pairs with incompatible deformation setups')


def report_partial(operator, targets):
    for target in targets:
        if 'strength' in target:
            skipped = np.count_nonzero((target['strength'] > 0) & ~target['supported'])
            if skipped:
                operator.report({'WARNING'}, f"{target['obj'].name}: preserved {skipped} in-range "
                                'vertices without a usable weight constraint')


def report_postprocess(operator, counts):
    if counts['normalization_failed']:
        operator.report({'WARNING'}, 'Could not normalize '
                        f"{counts['normalization_failed']} vertex/vertices because locked weights "
                        'leave no valid deform-weight budget')
    if counts['balance_limited']:
        operator.report({'WARNING'}, 'Could not fully balance '
                        f"{counts['balance_limited']} L/R weight-group pair(s) because their "
                        'writable weights cannot offset protected contributions')
