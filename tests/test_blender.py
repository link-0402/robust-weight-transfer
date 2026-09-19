"""Run with Blender --background --factory-startup --python this_file -- [deps_dir]."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
if '--' in sys.argv:
    sys.path.insert(0, sys.argv[sys.argv.index('--') + 1])

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rwt_test_addon', ROOT / '__init__.py',
                                            submodule_search_locations=[str(ROOT)])
addon = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = addon
spec.loader.exec_module(addon)
addon.register()
wt, seams, transfer, util = addon.weighttransfer, addon.transfer.seams, addon.transfer, addon.util


def split_surface():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                  [1, 0, 0], [2, 0, 0], [1, 1, 0], [2, 1, 0]], dtype=float)
    f = np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6], [5, 7, 6]])
    w = np.tile([0.25, 0.75], (8, 1))
    matched = np.array([True] * 4 + [False] * 4)
    return v, f, w, matched


class NumericalTests(unittest.TestCase):
    def test_partial_distance_strength(self):
        distance = np.array([0, 0.5, 0.8, 0.85, 0.9, 0.95, 1, 2])
        np.testing.assert_allclose(transfer.distance_strength(distance ** 2, 1, 20),
                                   [1, 1, 1, 0.84375, 0.5, 0.15625, 0, 0], atol=1e-12)
        np.testing.assert_array_equal(transfer.distance_strength(distance ** 2, 1, 0),
                                      [1, 1, 1, 1, 1, 1, 1, 0])
        np.testing.assert_allclose(transfer.distance_strength(np.array([0, .25, 1, 4]), 1, 100),
                                   [1, .5, 0, 0])
        for falloff in [0, 20, 100]:
            np.testing.assert_array_equal(transfer.distance_strength(distance ** 2, 0, falloff),
                                          [1, 0, 0, 0, 0, 0, 0, 0])
        for radius, falloff in [(-1, 20), (np.nan, 20), (1, 101)]:
            with self.assertRaises(ValueError):
                transfer.distance_strength(distance ** 2, radius, falloff)

    def test_partial_solve_does_not_use_rejected_constraints(self):
        v, f, w, matched = split_surface()
        matched[1:4] = False
        w[1:4] = [1, 0]
        w[4:] = [0, 1]
        domain = wt.prepare_inpainting(v, f, w, matched, False)
        with self.assertRaises(wt.InpaintingError):
            wt.solve_inpainting(domain)
        result = wt.solve_inpainting(domain, skip_rejected=True)
        np.testing.assert_allclose(result[:4], np.tile([.25, .75], (4, 1)), atol=1e-6)
        np.testing.assert_array_equal(domain.rejected, [False] * 4 + [True] * 4)
        for point in [False, True]:
            empty = wt.prepare_inpainting(v, f, w, np.zeros(8, bool), point)
            self.assertTrue(np.all(np.isfinite(wt.solve_inpainting(empty, skip_rejected=True))))

    def test_disconnected_rejected_at_every_scale(self):
        v, f, w, matched = split_surface()
        for scale in [0.01, 0.1, 1, 3, 10, 100]:
            with self.subTest(scale=scale):
                domain = wt.prepare_inpainting(v * scale, f, w, matched, False)
                np.testing.assert_array_equal(domain.rejected, ~matched)
                self.assertFalse(wt.inpaint(v * scale, f, w, matched, False)[0])

    def test_virtual_merge_rescues_both_modes(self):
        v, f, w, matched = split_surface()
        for point in [False, True]:
            domain = wt.prepare_inpainting(v, f, w, matched, point, 0.0001)
            self.assertFalse(domain.rejected.any())
            np.testing.assert_allclose(wt.solve_inpainting(domain), w, atol=1e-6)

    def test_all_and_no_matches(self):
        v, f, w, matched = split_surface()
        w[4:] = [0.9, 0.1]
        ok, out = wt.inpaint(v, f, w, np.ones(8, bool), False, 0.0001)
        self.assertTrue(ok)
        np.testing.assert_allclose(out, w)
        domain = wt.prepare_inpainting(v, f, w, np.zeros(8, bool), True, 0.0001)
        self.assertTrue(domain.rejected.all())

    def test_small_point_graph_and_diagnostic(self):
        v, f, w, matched = split_surface()
        v, f, w = v[:4], f[:2], w[:4]
        matched = np.array([True, False, False, False])
        domain = wt.prepare_inpainting(v, f, w, matched, True)
        self.assertFalse(domain.rejected.any())
        np.testing.assert_allclose(wt.solve_inpainting(domain), w, atol=1e-6)

    def test_point_graph_can_connect_surface_components(self):
        v, f, w, matched = split_surface()
        v[4:, 0] += 0.2
        surface = wt.prepare_inpainting(v, f, w, matched, False)
        point = wt.prepare_inpainting(v, f, w, matched, True)
        self.assertTrue(surface.rejected[4:].all())
        self.assertFalse(point.rejected.any())
        np.testing.assert_allclose(wt.solve_inpainting(point), w, atol=1e-6)

    def test_final_sync_repairs_smoothing_regression(self):
        v, f, w, matched = split_surface()
        w[:4] = [[1, 0], [0, 1], [0.8, 0.2], [0.2, 0.8]]
        ok, solved = wt.inpaint(v, f, w, matched, False, 0.0001)
        self.assertTrue(ok)
        edges = np.unique(np.sort(np.vstack([f[:, :2], f[:, 1:], f[:, [2, 0]]]), axis=1), axis=0)
        adjacency = wt.sp.sparse.csr_array((np.ones(len(edges) * 2),
                                           (edges.ravel(), edges[:, ::-1].ravel())), shape=(8, 8))
        adjacency.setdiag(1)
        lists = [adjacency[[i]].indices.tolist() for i in range(8)]
        smoothed = np.asarray(wt.smooth_weigths(v, solved, matched, adjacency, lists, 4, 0.2, 5))
        self.assertGreater(np.max(np.abs(smoothed[[1, 3]] - smoothed[[4, 6]])), 0.1)
        final, synced, skipped = seams.synchronize_weights(
            smoothed, np.ones_like(smoothed, bool), np.zeros(8, bool),
            [np.array([1, 4]), np.array([3, 6])], 1)
        np.testing.assert_array_equal(final[[1, 3]], final[[4, 6]])
        np.testing.assert_array_equal(final[[0, 2, 5, 7]], smoothed[[0, 2, 5, 7]])
        self.assertEqual((synced, skipped), (2, 0))

    def test_duplicate_and_collapsed_faces(self):
        v, f, w, matched = split_surface()
        v[4:] = v[:4]
        ok, out = wt.inpaint(v, np.vstack([f, f[:, ::-1]]), w, matched, False, 0.0001)
        self.assertTrue(ok)
        np.testing.assert_allclose(out, w, atol=1e-6)
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                      [2, 0, 0], [2.00001, 0, 0], [2, 0.00001, 0]])
        f = np.array([[0, 1, 2], [3, 4, 5]])
        w = np.tile([0.25, 0.75], (6, 1))
        matched = np.array([True, False, False, True, True, True])
        self.assertTrue(wt.inpaint(v, f, w, matched, False, 0.0001)[0])
        matched[3:] = False
        self.assertFalse(wt.inpaint(v, f, w, matched, False, 0.0001)[0])

    def test_invalid_data(self):
        v, f, w, matched = split_surface()
        for value in [np.nan, np.inf]:
            broken = v.copy(); broken[0, 0] = value
            self.assertFalse(wt.inpaint(broken, f, w, matched, False)[0])
            broken = w.copy(); broken[0, 0] = value
            self.assertFalse(wt.inpaint(v, f, broken, matched, False)[0])
        self.assertFalse(wt.inpaint(v, f + 100, w, matched, False)[0])

    def test_seam_detection_boundaries_scope_and_zero(self):
        v, f, w, matched = split_surface()
        edges = np.unique(np.sort(np.vstack([f[:, :2], f[:, 1:], f[:, [2, 0]]]), axis=1), axis=0)
        boundary, labels = seams.boundary_components(len(v), f, edges)
        clusters, excluded = seams.find_seam_clusters(v, boundary, labels, np.zeros(8), 0)
        self.assertEqual([c.tolist() for c in clusters], [[1, 4], [3, 6]])
        self.assertEqual(excluded, 0)
        clusters, excluded = seams.find_seam_clusters(v, boundary, labels, labels, 0)
        self.assertEqual(clusters, [])
        self.assertEqual(excluded, 2)
        # Closed tetrahedra have no open borders, even with duplicate geometry.
        tetra = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
        boundary, _ = seams.boundary_components(4, tetra, np.array([[0, 1], [1, 2], [2, 3]]))
        self.assertFalse(boundary.any())

    def test_seam_clusters_avoid_density_chaining(self):
        # A seam sampled far more densely on one side puts a single sparse-side
        # vertex within range of several dense-side vertices. Without a
        # reciprocal-nearest check, connected components chains the whole run
        # into one oversized cluster (the entire seam, in this example)
        # instead of many local pairs, smearing weights across positions that
        # were never actually close together.
        front_y = np.linspace(0, 1, 100)
        back_y = np.linspace(0, 1, 10)
        vertices = np.vstack([
            np.column_stack([np.zeros(100), front_y, np.zeros(100)]),
            np.column_stack([np.full(10, 0.001), back_y, np.zeros(10)]),
        ])
        boundary = np.ones(110, dtype=bool)
        components = np.array([0] * 100 + [1] * 10)
        clusters, excluded = seams.find_seam_clusters(
            vertices, boundary, components, np.zeros(110), 0.07)
        self.assertEqual(excluded, 0)
        self.assertEqual(len(clusters), 10)
        self.assertTrue(all(len(cluster) == 2 for cluster in clusters))

    def test_seam_limit_protections_and_stable_ties(self):
        w = np.eye(6)[:2]
        w[0] = [0.25, 0.25, 0.25, 0.25, 0, 0]
        w[1] = [0, 0, 0.25, 0.25, 0.25, 0.25]
        editable = np.ones_like(w, bool)
        out, synced, skipped = seams.synchronize_weights(w, editable, np.zeros(2, bool), [np.arange(2)], 4)
        np.testing.assert_array_equal(out[0], out[1])
        self.assertLessEqual(np.count_nonzero(out[0]), 4)
        self.assertAlmostEqual(out[0].sum(), 1)
        self.assertEqual((synced, skipped), (1, 0))
        editable[:, 0] = False
        out, synced, skipped = seams.synchronize_weights(w, editable, np.zeros(2, bool), [np.arange(2)], 4)
        np.testing.assert_array_equal(out, w)
        self.assertEqual(skipped, 1)
        out, _, skipped = seams.synchronize_weights(w, np.ones_like(w, bool), np.array([True, False]), [np.arange(2)])
        np.testing.assert_array_equal(out, w)
        self.assertEqual(skipped, 1)

    def test_confidence_prevents_matched_weight_dilution(self):
        # A double-sided mesh duplicates every vertex as a front- and a
        # back-facing copy. The front copy commonly gets a clean direct
        # surface match; its back-facing twin, unmatched, only reaches a
        # (weaker) weight through inpainting. Averaging the two blindly drags
        # the well-matched side down at every seam vertex around the
        # silhouette. Confidence-weighting the merge toward the matched side
        # avoids that dilution.
        w = np.array([[1.0, 0.0], [0.3, 0.7]])
        editable = np.ones_like(w, bool)
        out, synced, skipped = seams.synchronize_weights(
            w, editable, np.zeros(2, bool), [np.arange(2)], confidence=np.array([1.0, 0.0]))
        np.testing.assert_allclose(out, [[1.0, 0.0], [1.0, 0.0]])
        self.assertEqual((synced, skipped), (1, 0))
        # With no confidence signal, the merge falls back to a plain mean.
        out, synced, skipped = seams.synchronize_weights(w, editable, np.zeros(2, bool), [np.arange(2)])
        np.testing.assert_allclose(out, [[0.65, 0.35], [0.65, 0.35]])
        # All-zero confidence within a cluster also falls back to a plain mean.
        out, synced, skipped = seams.synchronize_weights(
            w, editable, np.zeros(2, bool), [np.arange(2)], confidence=np.zeros(2))
        np.testing.assert_allclose(out, [[0.65, 0.35], [0.65, 0.35]])

    def test_fixed_contributions_count_toward_limit(self):
        w = np.array([[0.2, 0.6, 0.2], [0.2, 0.1, 0.7]])
        editable = np.array([[False, True, True]] * 2)
        out, synced, skipped = seams.synchronize_weights(w, editable, np.zeros(2, bool), [np.arange(2)], 2)
        np.testing.assert_array_equal(out[0], out[1])
        np.testing.assert_array_equal(out[:, 0], w[:, 0])
        self.assertEqual(np.count_nonzero(out[0]), 2)
        self.assertEqual((synced, skipped), (1, 0))

    def test_infeasible_fixed_weights_and_limits_are_skipped(self):
        for fixed, limit in [(1.1, 4), (0.2, 1)]:
            w = np.array([[fixed, 0.5, 0.3], [fixed, 0.1, 0.7]])
            out, synced, skipped = seams.synchronize_weights(
                w, np.array([[False, True, True]] * 2), np.zeros(2, bool), [np.arange(2)], limit)
            np.testing.assert_array_equal(out, w)
            self.assertEqual((synced, skipped), (0, 1))


class BlenderTests(unittest.TestCase):
    def setUp(self):
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        self.settings = bpy.context.scene.robust_weight_transfer_settings
        bpy.ops.object.rbt_reset_scene_settings()
        self.settings.use_deformed_source = False
        self.settings.use_deformed_target = False
        self.settings.inpaint_mode = 'SURFACE'
        self.settings.max_distance = 0.1
        self.rig = self.make_rig('Rig')
        self.v, self.f, _, _ = split_surface()

    def make_rig(self, name):
        data = bpy.data.armatures.new(name)
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        for index, name in enumerate([
                'A', 'B', 'C', 'D', 'E', 'F', 'Arm.L', 'Arm.R', 'Leg_l', 'Leg_r', 'Spine']):
            bone = data.edit_bones.new(name)
            bone.head = (index * 0.1, 0, -1)
            bone.tail = (index * 0.1, 0, 1)
        bpy.ops.object.mode_set(mode='OBJECT')
        obj.select_set(False)
        return obj

    def mesh(self, name, v, f, rig=True):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(np.asarray(v).tolist(), [], np.asarray(f).tolist())
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        if rig:
            mod = obj.modifiers.new('Armature', 'ARMATURE')
            mod.object = self.rig
        return obj

    def weights(self, obj, names, values):
        for name, column in zip(names, np.asarray(values).T):
            group = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
            for index, value in enumerate(column):
                if value > 0:
                    group.add([index], float(value), 'REPLACE')

    def select(self, active, *others):
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        for obj in (active,) + others:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = active

    def source(self):
        obj = self.mesh('Source', self.v, self.f)
        # Split source has an intentional weight jump at its duplicate border.
        values = np.array([[0.9, 0.1]] * 4 + [[0.1, 0.9]] * 4)
        self.weights(obj, ['A', 'B'], values)
        self.settings.source_object = obj
        return obj

    def read(self, obj, names=('A', 'B')):
        return np.column_stack([util.get_group_arr(obj, name) if obj.vertex_groups.get(name)
                                else np.zeros(len(obj.data.vertices)) for name in names])

    def snapshot(self, obj):
        return ([(tuple(v.co), tuple(v.normal)) for v in obj.data.vertices],
                [tuple(e.vertices) for e in obj.data.edges],
                [tuple(p.vertices) for p in obj.data.polygons],
                [[tuple(d.uv) for d in uv.data] for uv in obj.data.uv_layers],
                [[tuple(p.co) for p in key.data] for key in obj.data.shape_keys.key_blocks]
                if obj.data.shape_keys else [],
                [(m.name, m.type) for m in obj.modifiers])

    def partial_scene(self, heights=(0, .75, 1, 2)):
        source = self.mesh('Source', [[-10, -10, 0], [10, -10, 0], [0, 10, 0]], [[0, 1, 2]])
        self.weights(source, ['A', 'B'], np.tile([.9, .1], (3, 1)))
        vertices = [[x, y, z] for z in heights for x, y in [(0, 0), (1, 0), (0, 1)]]
        faces = [[i, i + 1, i + 2] for i in range(0, len(vertices), 3)]
        target = self.mesh('Target', vertices, faces)
        self.weights(target, ['A', 'B'], np.tile([.2, .8], (len(vertices), 1)))
        self.settings.source_object = source
        self.settings.partial_reweight = True
        self.settings.partial_reweight_falloff = 50
        self.settings.max_distance = 1
        self.select(target)
        return source, target

    def memberships(self, obj):
        return [tuple((obj.vertex_groups[g.group].name, g.weight) for g in v.groups)
                for v in obj.data.vertices]

    def test_partial_operator_falloff_preserves_outside_memberships(self):
        _, target = self.partial_scene()
        target.vertex_groups['A'].add([9], 1e-8, 'REPLACE')
        target.vertex_groups['B'].remove([10])
        target.vertex_groups['A'].remove([10])
        before = self.memberships(target)
        geometry = self.snapshot(target)
        self.settings.smoothing_enable = True
        self.settings.seam_sync = True
        self.assertEqual(bpy.ops.object.skin_weight_transfer(), {'FINISHED'})
        np.testing.assert_allclose(self.read(target)[:6],
                                   [[.9, .1]] * 3 + [[.55, .45]] * 3, atol=1e-6)
        self.assertEqual(self.memberships(target)[6:], before[6:])
        self.assertEqual(self.snapshot(target), geometry)

    def test_partial_masks_locks_and_missing_groups(self):
        source, target = self.partial_scene()
        target.vertex_groups.remove(target.vertex_groups['B'])
        self.weights(source, ['C'], np.ones((3, 1)))
        self.weights(target, ['C', 'TargetOnly', 'Mask'], np.tile([.3, .4, .25], (12, 1)))
        target.vertex_groups['C'].lock_weight = True
        target.robust_weight_transfer_settings.vertex_group = 'Mask'
        target.robust_weight_transfer_settings.vertex_group_invert = True
        before = self.memberships(target)
        self.assertEqual(bpy.ops.object.skin_weight_transfer(), {'FINISHED'})
        np.testing.assert_allclose(self.read(target)[:6],
                                   [[.725, .075]] * 3 + [[.4625, .0375]] * 3, atol=1e-6)
        np.testing.assert_allclose(self.read(target, ['C', 'TargetOnly', 'Mask']),
                                   np.tile([.3, .4, .25], (12, 1)))
        self.assertEqual(self.memberships(target)[6:], before[6:])

    def test_partial_zero_radius_and_hard_cutoff(self):
        source, target = self.partial_scene()
        self.settings.partial_reweight_falloff = 0
        self.assertEqual(bpy.ops.object.skin_weight_transfer(), {'FINISHED'})
        np.testing.assert_allclose(self.read(target)[:9], np.tile([.9, .1], (9, 1)), atol=1e-6)
        self.weights(target, ['A', 'B'], np.tile([.2, .8], (12, 1)))
        # Use exact BVH hits for the zero-radius case; triangle-interior
        # projections can have a nonzero floating-point residual.
        for vertex, original in zip(target.data.vertices[:3], source.data.vertices):
            vertex.co = original.co
        target.data.update()
        self.settings.max_distance = 0
        self.settings.partial_reweight_falloff = 100
        self.assertEqual(bpy.ops.object.skin_weight_transfer(), {'FINISHED'})
        np.testing.assert_allclose(self.read(target), [[.9, .1]] * 3 + [[.2, .8]] * 9, atol=1e-6)

    def test_partial_no_matches_and_selected_noop(self):
        _, target = self.partial_scene((2,))
        target.vertex_groups.remove(target.vertex_groups['B'])
        far = self.mesh('Far', self.v + [0, 0, 3], self.f)
        before = self.memberships(target)
        for point in ['POINT', 'SURFACE']:
            self.settings.inpaint_mode = point
            self.settings.apply_to_selected = True
            self.select(target, far)
            self.assertEqual(bpy.ops.object.skin_weight_transfer(), {'FINISHED'})
            self.assertEqual(self.memberships(target), before)
            self.assertNotIn('B', target.vertex_groups)
            self.assertEqual(len(far.vertex_groups), 0)

    def test_partial_unsupported_normals_and_diagnostics(self):
        _, target = self.partial_scene((0, .5, 2))
        # Reject the middle part's normals, while the last part is outside.
        for polygon in target.data.polygons:
            if polygon.index == 1:
                polygon.flip()
        target.data.update()
        self.settings.flip_vertex_normal = False
        self.settings.draw_matched = True
        before = self.memberships(target)
        with patch.object(transfer, 'report_partial', wraps=transfer.report_partial) as report:
            self.assertEqual(bpy.ops.object.skin_weight_transfer(), {'FINISHED'})
        staged = report.call_args.args[1][0]
        self.assertEqual(np.count_nonzero((staged['strength'] > 0) & ~staged['supported']), 3)
        self.assertEqual(self.memberships(target)[3:], before[3:])
        colors = target.data.color_attributes['RBT Matched'].data
        self.assertLess(colors[3].color[1], .01)
        self.assertGreater(colors[6].color[1], .99)
        bpy.ops.object.mode_set(mode='EDIT')
        self.assertEqual(bpy.ops.object.select_non_matched(), {'FINISHED'})
        bpy.ops.object.mode_set(mode='OBJECT')
        self.assertEqual([v.select for v in target.data.vertices], [False] * 3 + [True] * 3 + [False] * 3)

    def test_partial_no_usable_normals_in_both_modes(self):
        _, target = self.partial_scene((.5,))
        target.data.polygons[0].flip()
        target.data.update()
        self.settings.flip_vertex_normal = False
        before = self.memberships(target)
        for mode in ['SURFACE', 'POINT']:
            self.settings.inpaint_mode = mode
            self.assertEqual(bpy.ops.object.skin_weight_transfer(), {'FINISHED'})
            self.assertEqual(self.memberships(target), before)

    def test_partial_surface_distance_world_and_evaluated_shapes(self):
        source, target = self.partial_scene((.5,))
        # Local .5 becomes world distance 1, halfway through a radius-2 falloff.
        for obj in [source, target]:
            obj.location = (4, -3, 2)
            obj.scale = (2, 3, 2)
        self.settings.max_distance = 2
        self.settings.partial_reweight_falloff = 100
        bpy.context.view_layer.update()
        self.assertEqual(bpy.ops.object.skin_weight_transfer(), {'FINISHED'})
        np.testing.assert_allclose(self.read(target), np.tile([.55, .45], (3, 1)), atol=1e-6)
        source.shape_key_add(name='Basis')
        key = source.shape_key_add(name='Raised')
        for vertex in key.data:
            vertex.co.z = .5
        key.value = 1
        self.settings.use_deformed_source = True
        bpy.context.view_layer.update()
        self.assertEqual(bpy.ops.object.skin_weight_transfer(), {'FINISHED'})
        np.testing.assert_allclose(self.read(target), np.tile([.9, .1], (3, 1)), atol=1e-6)
        target.shape_key_add(name='Basis')
        key = target.shape_key_add(name='Far')
        for vertex in key.data:
            vertex.co.z += 2
        key.value = 1
        self.settings.use_deformed_target = True
        bpy.context.view_layer.update()
        before = self.memberships(target)
        self.assertEqual(bpy.ops.object.skin_weight_transfer(), {'FINISHED'})
        self.assertEqual(self.memberships(target), before)

    def test_partial_virtual_cross_object_recovery(self):
        source = self.mesh('Source', self.v[:4], self.f[:2])
        self.weights(source, ['A', 'B'], np.tile([.9, .1], (4, 1)))
        self.settings.source_object = source
        left = self.mesh('Left', self.v[:4], self.f[:2])
        right = self.mesh('Right', self.v[4:], (self.f[2:] - 4)[:, ::-1])
        self.weights(right, ['B', 'A'], np.tile([.8, .2], (4, 1)))
        self.settings.partial_reweight = True
        self.settings.partial_reweight_falloff = 0
        self.settings.max_distance = 2
        self.settings.flip_vertex_normal = False
        self.settings.apply_to_selected = True
        self.settings.seam_sync = True
        self.settings.seam_sync_across_objects = True
        self.settings.virtual_merge = True
        # Batch mode retains the existing convention of ignoring object masks.
        self.weights(right, ['Mask'], np.zeros((4, 1)))
        right.robust_weight_transfer_settings.vertex_group = 'Mask'
        for mode in ['SURFACE', 'POINT']:
            self.settings.inpaint_mode = mode
            self.select(left, right)
            self.assertEqual(bpy.ops.object.skin_weight_transfer(), {'FINISHED'})
            np.testing.assert_allclose(self.read(right), np.tile([.9, .1], (4, 1)), atol=1e-6)

    def test_partial_seams_skip_falloff_and_preserved_members(self):
        self.settings.seam_sync = True
        for strength in [0, .5]:
            targets = self.staged_pair()
            for target in targets:
                target['strength'] = np.full(4, strength)
                target['supported'] = np.ones(4, bool)
                transfer.stage_weights(target, ['A', 'B'], [True, True])
            before = [dict((k, v.copy()) for k, v in t['final'].items()) for t in targets]
            self.settings.apply_to_selected = True
            self.settings.seam_sync_across_objects = True
            self.assertEqual(transfer.synchronize_targets(targets, self.settings, ['A', 'B']), (0, 2, 0))
            for target, previous in zip(targets, before):
                for name, values in previous.items():
                    np.testing.assert_array_equal(target['final'][name], values)

    def test_partial_processing_excludes_unsupported_neighbors(self):
        # Connected triangles, but support stops at vertex 2. Unsupported weights
        # must not enter smoothing averages or influence-limit dilation.
        obj = self.mesh('Target', [[0, 0, 0], [.1, 0, 0], [0, .1, 0], [.1, .1, 0]],
                        [[0, 1, 2], [1, 3, 2]])
        self.settings.smoothing_enable = True
        self.settings.max_distance = 1
        self.settings.num_limit_groups = 2
        outputs = []
        for unsupported in [[1, 0, 0, 0, 0], [.2, .2, .2, .2, .2]]:
            target = transfer.make_target(obj, bpy.context.evaluated_depsgraph_get(), self.settings)
            target.update(weights=np.array([[.7, .3, 0, 0, 0]] * 3 + [unsupported]),
                          matched=np.array([True, False, False, False]),
                          supported=np.array([True, True, True, False]))
            transfer.process_weights(target, self.settings)
            outputs.append(target['weights'][:3].copy())
        np.testing.assert_allclose(outputs[0], np.tile([.7, .3, 0, 0, 0], (3, 1)))
        np.testing.assert_array_equal(*outputs)

    def test_partial_numerical_failure_cancels_without_writes(self):
        _, target = self.partial_scene((0, .75))
        self.settings.draw_matched = True
        before = self.memberships(target)
        with patch.object(wt, 'solve_inpainting', side_effect=wt.InpaintingError('test failure')):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                bpy.ops.object.skin_weight_transfer()
        self.assertEqual(self.memberships(target), before)
        self.assertNotIn('RBT Matched', target.data.color_attributes)

    def test_partial_defaults_reset_and_standalone_inpaint(self):
        self.assertFalse(self.settings.partial_reweight)
        self.assertEqual(self.settings.partial_reweight_falloff, 20)
        self.settings.partial_reweight = True
        self.settings.partial_reweight_falloff = 70
        bpy.ops.object.rbt_reset_scene_settings()
        self.assertFalse(self.settings.partial_reweight)
        self.assertEqual(self.settings.partial_reweight_falloff, 20)
        self.settings.partial_reweight = True
        self.settings.max_distance = 0
        self.settings.inpaint_mode = 'SURFACE'
        self.test_standalone_inpaint_inversion_and_diagnostic()

    def test_operator_final_seams_and_geometry_and_bone_scaling(self):
        self.source()
        target = self.mesh('Target', self.v, self.f)
        target.data.uv_layers.new(name='UVMap')
        target.shape_key_add(name='Basis')
        key = target.shape_key_add(name='Shape')
        key.data[0].co.z = 0.1
        self.weights(target, ['MaskData', 'C'], np.tile([0.33, 0.7], (8, 1)))
        # Source C exists but is zero everywhere: stale destination C must clear.
        self.settings.source_object.vertex_groups.new(name='C')
        before = self.snapshot(target)
        self.settings.seam_sync = True
        self.settings.smoothing_enable = True
        self.settings.num_limit_groups = 1
        self.select(target)
        self.assertEqual(bpy.ops.object.skin_weight_transfer(), {'FINISHED'})
        out = self.read(target)
        np.testing.assert_array_equal(out[[1, 3]], out[[4, 6]])
        self.assertTrue(np.all(np.count_nonzero(out[[1, 3]], axis=1) <= 1))
        np.testing.assert_allclose(out[[1, 3]].sum(axis=1), 1)
        np.testing.assert_array_equal(self.read(target, ['C']), 0)
        np.testing.assert_allclose(self.read(target, ['MaskData']), 0.33)
        self.assertEqual(before, self.snapshot(target))
        for name, scale, location, rotation in [('A', (1.7, 0.6, 1.3), (0.3, 0, 0.1), (0, 0.2, 0.3)),
                                                 ('B', (0.7, 1.6, 0.8), (0, 0.2, 0), (0.2, 0, 0))]:
            bone = self.rig.pose.bones[name]
            bone.scale = scale
            bone.location = location
            bone.rotation_mode = 'XYZ'
            bone.rotation_euler = rotation
        bpy.context.view_layer.update()
        evaluated = target.evaluated_get(bpy.context.evaluated_depsgraph_get())
        coords = np.array([v.co[:] for v in evaluated.data.vertices])
        np.testing.assert_allclose(coords[[1, 3]], coords[[4, 6]], atol=1e-6)

    def test_cross_object_defaults_names_and_shared_solve(self):
        source = self.mesh('Source', self.v[:4], self.f[:2])
        self.weights(source, ['A', 'B'], np.tile([0.25, 0.75], (4, 1)))
        self.settings.source_object = source
        left = self.mesh('Left', self.v[:4], self.f[:2])
        right = self.mesh('Right', self.v[4:], self.f[2:] - 4)
        # Different group ordering must not change the result.
        self.weights(right, ['B', 'A'], np.tile([0.1, 0.9], (4, 1)))
        self.settings.apply_to_selected = True
        self.settings.seam_sync = True
        self.settings.virtual_merge = True
        # Reject all right vertices by normals, including its duplicate seam.
        for polygon in right.data.polygons:
            polygon.flip()
        right.data.update()
        self.settings.flip_vertex_normal = False
        self.select(left, right)
        original_right = self.read(right).copy()
        self.settings.draw_matched = True
        with self.assertRaises(RuntimeError):
            bpy.ops.object.skin_weight_transfer()
        np.testing.assert_array_equal(self.read(right), original_right)
        self.assertEqual(len(left.vertex_groups), 0)
        self.assertEqual(len(left.data.color_attributes), 0)
        self.settings.seam_sync_across_objects = True
        self.select(right, left)
        bpy.ops.object.mode_set(mode='EDIT')
        self.assertEqual(bpy.ops.object.select_non_matched(), {'FINISHED'})
        bpy.ops.object.mode_set(mode='OBJECT')
        self.assertFalse(any(vertex.select for vertex in right.data.vertices))
        self.assertEqual(bpy.ops.object.skin_weight_transfer(), {'FINISHED'})
        np.testing.assert_allclose(self.read(left), np.tile([0.25, 0.75], (4, 1)), atol=1e-6)
        np.testing.assert_allclose(self.read(right), self.read(left), atol=1e-6)

    def test_protected_seams_masks_and_locked_groups(self):
        self.source()
        target = self.mesh('Target', self.v, self.f)
        self.weights(target, ['A', 'B'], np.array([[0.8, 0.2]] * 4 + [[0.3, 0.7]] * 4))
        target.vertex_groups['A'].lock_weight = True
        self.settings.seam_sync = True
        self.select(target)
        before_a = self.read(target, ['A']).copy()
        self.assertEqual(bpy.ops.object.skin_weight_transfer(), {'FINISHED'})
        np.testing.assert_array_equal(self.read(target, ['A']), before_a)
        self.assertNotEqual(self.read(target)[1, 0], self.read(target)[4, 0])
        target.vertex_groups['A'].lock_weight = False
        self.weights(target, ['TransferMask'], np.array([[0.0]] * 4 + [[0.5]] * 4))
        target.robust_weight_transfer_settings.vertex_group = 'TransferMask'
        baseline = self.read(target).copy()
        self.assertEqual(bpy.ops.object.skin_weight_transfer(), {'FINISHED'})
        np.testing.assert_array_equal(self.read(target)[:4], baseline[:4])
        np.testing.assert_allclose(self.read(target)[5], 0.5 * baseline[5] + 0.5 * np.array([0.1, 0.9]), atol=1e-6)

    def test_standalone_inpaint_inversion_and_diagnostic(self):
        self.source()
        target = self.mesh('Target', self.v, self.f)
        self.weights(target, ['A', 'B'], np.array([[0.25, 0.75]] * 4 + [[0.9, 0.1]] * 4))
        self.weights(target, ['Inpaint'], np.array([[1.0]] * 4 + [[0.0]] * 4))
        target.robust_weight_transfer_settings.inpaint_group = 'Inpaint'
        target.robust_weight_transfer_settings.inpaint_group_invert = True
        self.settings.virtual_merge = True
        self.settings.seam_sync = True
        self.select(target)
        self.assertEqual(bpy.ops.object.rwt_inpaint(), {'FINISHED'})
        np.testing.assert_allclose(self.read(target), np.tile([0.25, 0.75], (8, 1)), atol=1e-6)
        bpy.ops.object.mode_set(mode='EDIT')
        self.assertEqual(bpy.ops.object.select_non_matched(), {'FINISHED'})
        bpy.ops.object.mode_set(mode='OBJECT')
        self.assertFalse(any(v.select for v in target.data.vertices))
        self.settings.virtual_merge = False
        bpy.ops.object.mode_set(mode='EDIT')
        self.assertEqual(bpy.ops.object.select_non_matched(), {'FINISHED'})
        bpy.ops.object.mode_set(mode='OBJECT')
        self.assertEqual([v.select for v in target.data.vertices], [False] * 4 + [True] * 4)

    def staged_pair(self, right_rig=True):
        left = self.mesh('Left', self.v[:4], self.f[:2])
        right = self.mesh('Right', self.v[4:], self.f[2:] - 4, rig=right_rig)
        self.weights(right, ['B', 'A'], np.tile([0.9, 0.1], (4, 1)))
        targets = []
        for obj, values in [(left, [0.8, 0.2]), (right, [0.1, 0.9])]:
            target = transfer.make_target(obj, bpy.context.evaluated_depsgraph_get(), self.settings)
            target['weights'] = np.tile(values, (4, 1))
            transfer.stage_weights(target, ['A', 'B'], [True, True])
            targets.append(target)
        return targets

    def test_cross_scope_and_unbound_palette(self):
        self.settings.seam_sync = True
        self.settings.apply_to_selected = True
        targets = self.staged_pair(right_rig=False)
        self.assertEqual(transfer.synchronize_targets(targets, self.settings, ['A', 'B'], self.rig), (0, 0, 0))
        self.settings.seam_sync_across_objects = True
        self.assertEqual(transfer.synchronize_targets(targets, self.settings, ['A', 'B'], self.rig), (2, 0, 0))
        transfer.write_targets(targets)
        np.testing.assert_array_equal(self.read(targets[0]['obj'])[[1, 3]], self.read(targets[1]['obj'])[[0, 2]])

    def test_incompatible_armatures_and_target_only_weights(self):
        self.settings.seam_sync = self.settings.seam_sync_across_objects = True
        self.settings.apply_to_selected = True
        targets = self.staged_pair()
        other = self.make_rig('OtherRig')
        targets[1]['obj'].modifiers[0].object = other
        self.assertEqual(transfer.synchronize_targets(targets, self.settings, ['A', 'B'], self.rig), (0, 0, 2))
        targets[1]['obj'].modifiers[0].object = self.rig
        self.weights(targets[1]['obj'], ['C'], np.tile([0.2], (4, 1)))
        transfer.stage_weights(targets[1], ['A', 'B'], [True, True])
        self.assertEqual(transfer.synchronize_targets(targets, self.settings, ['A', 'B'], self.rig), (0, 2, 0))
        transfer.write_targets(targets)
        np.testing.assert_allclose(self.read(targets[1]['obj'], ['C']), 0.2)

    def test_seams_use_rest_positions_and_world_transforms(self):
        self.settings.seam_sync = self.settings.seam_sync_across_objects = True
        self.settings.apply_to_selected = True
        targets = self.staged_pair()
        right = targets[1]['obj']
        # Different local coordinates and object transforms, same world seam.
        for vertex in right.data.vertices:
            vertex.co.x -= 3
        right.location.x = 3
        bpy.context.view_layer.update()
        transfer.write_targets(targets)
        self.rig.pose.bones['A'].scale = (2, 1, 1)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        before = [util.get_obj_arrs_world(t['obj'].evaluated_get(depsgraph))[0] for t in targets]
        self.assertGreater(np.linalg.norm(before[0][1] - before[1][0]), 0.1)
        self.assertEqual(transfer.synchronize_targets(targets, self.settings, ['A', 'B'], self.rig), (2, 0, 0))
        transfer.write_targets(targets)
        bpy.context.view_layer.update()
        after = [util.get_obj_arrs_world(t['obj'].evaluated_get(bpy.context.evaluated_depsgraph_get()))[0]
                 for t in targets]
        np.testing.assert_allclose(after[0][[1, 3]], after[1][[0, 2]], atol=1e-6)

    def test_settings_defaults_and_reset(self):
        self.assertFalse(self.settings.seam_sync)
        self.assertFalse(self.settings.seam_sync_across_objects)
        self.assertAlmostEqual(self.settings.seam_distance, 0.0001)
        self.assertFalse(self.settings.balance_lr_weight_groups)
        self.assertFalse(self.settings.normalize_weights_after_transfer)
        self.settings.seam_sync = True
        self.settings.seam_sync_across_objects = True
        self.settings.seam_distance = 0.01
        self.settings.balance_lr_weight_groups = True
        self.settings.normalize_weights_after_transfer = True
        bpy.ops.object.rbt_reset_scene_settings()
        self.assertFalse(self.settings.seam_sync)
        self.assertFalse(self.settings.seam_sync_across_objects)
        self.assertAlmostEqual(self.settings.seam_distance, 0.0001)
        self.assertFalse(self.settings.balance_lr_weight_groups)
        self.assertFalse(self.settings.normalize_weights_after_transfer)

    def test_normalize_and_balance_final_transfer_weights(self):
        obj = self.mesh('Target', self.v[:4], self.f[:2])
        self.weights(obj, ['Arm.L', 'Arm.R', 'Spine', 'Mask'],
                     np.tile([.2, .1, .2, .8], (4, 1)))
        target = transfer.make_target(obj, bpy.context.evaluated_depsgraph_get(), self.settings)
        target['weights'] = np.tile([.7, .1], (4, 1))
        transfer.stage_weights(target, ['Arm.L', 'Arm.R'], [True, True])
        self.settings.normalize_weights_after_transfer = True
        self.settings.balance_lr_weight_groups = True
        counts = transfer.postprocess_transfer_targets([target], self.settings, ['Arm.L', 'Arm.R'])
        self.assertEqual(counts, {'normalization_failed': 0, 'balance_limited': 0})
        transfer.write_targets([target])
        deform = self.read(obj, ['Arm.L', 'Arm.R', 'Spine'])
        np.testing.assert_allclose(deform.sum(axis=1), 1, atol=1e-6)
        self.assertAlmostEqual(deform[:, 0].sum(), deform[:, 1].sum(), places=6)
        np.testing.assert_allclose(self.read(obj, ['Mask']), .8, atol=1e-6)

    def test_balance_recognizes_ffxiv_names_and_respects_touched_vertices(self):
        obj = self.mesh('Target', self.v[:4], self.f[:2])
        self.weights(obj, ['Leg_l', 'Leg_r'], np.tile([.2, .8], (4, 1)))
        target = transfer.make_target(obj, bpy.context.evaluated_depsgraph_get(), self.settings)
        target['weights'] = np.tile([.9, .1], (4, 1))
        transfer.stage_weights(target, ['Leg_l', 'Leg_r'], [True, True])
        before = target['final']['Leg_l'][:2].copy(), target['final']['Leg_r'][:2].copy()
        target['write_vertices'][:2] = False
        self.settings.balance_lr_weight_groups = True
        counts = transfer.postprocess_transfer_targets([target], self.settings, ['Leg_l', 'Leg_r'])
        self.assertEqual(counts['balance_limited'], 0)
        np.testing.assert_array_equal(target['final']['Leg_l'][:2], before[0])
        np.testing.assert_array_equal(target['final']['Leg_r'][:2], before[1])
        self.assertAlmostEqual(target['final']['Leg_l'].sum(), target['final']['Leg_r'].sum(), places=6)

    def test_normalize_preserves_locked_values_and_reports_infeasible_vertices(self):
        obj = self.mesh('Target', self.v[:4], self.f[:2])
        self.weights(obj, ['Arm.L', 'Arm.R', 'Spine'], np.tile([.2, .2, .3], (4, 1)))
        for name in ['Arm.L', 'Arm.R', 'Spine']:
            obj.vertex_groups[name].lock_weight = True
        target = transfer.make_target(obj, bpy.context.evaluated_depsgraph_get(), self.settings)
        target['weights'] = np.tile([.9, .1], (4, 1))
        transfer.stage_weights(target, ['Arm.L', 'Arm.R'], [True, True])
        before = {name: values.copy() for name, values in target['final'].items()}
        self.settings.normalize_weights_after_transfer = True
        counts = transfer.postprocess_transfer_targets([target], self.settings, ['Arm.L', 'Arm.R'])
        self.assertEqual(counts['normalization_failed'], 4)
        for name, values in before.items():
            np.testing.assert_array_equal(target['final'][name], values)

    def test_normalize_repairs_writer_cutoff_residual(self):
        obj = self.mesh('Target', self.v[:4], self.f[:2])
        self.weights(obj, ['Arm.L', 'Arm.R', 'Spine'],
                     np.tile([.2, .2, 1e-8], (4, 1)))
        target = transfer.make_target(obj, bpy.context.evaluated_depsgraph_get(), self.settings)
        target['weights'] = np.tile([.6, .4], (4, 1))
        transfer.stage_weights(target, ['Arm.L', 'Arm.R'], [True, True])
        self.settings.normalize_weights_after_transfer = True
        counts = transfer.postprocess_transfer_targets([target], self.settings, ['Arm.L', 'Arm.R'])
        self.assertEqual(counts['normalization_failed'], 0)
        transfer.write_targets([target])
        deform = self.read(obj, ['Arm.L', 'Arm.R', 'Spine'])
        np.testing.assert_allclose(deform.sum(axis=1), 1, atol=1e-6)
        np.testing.assert_array_equal(deform[:, 2], 0)

    def test_shared_mesh_is_rejected_without_mutation(self):
        self.source()
        target = self.mesh('Target', self.v, self.f)
        linked = bpy.data.objects.new('Linked', target.data)
        bpy.context.collection.objects.link(linked)
        self.settings.seam_sync = True
        self.select(target)
        with self.assertRaises(RuntimeError):
            bpy.ops.object.skin_weight_transfer()
        self.assertEqual(len(target.vertex_groups), 0)
        self.assertEqual(target.data, linked.data)

    def test_non_deform_group_is_not_used_from_other_rig(self):
        # A group can be a bone on one target but an unrelated mask on another.
        self.settings.seam_sync = True
        first = self.mesh('First', self.v, self.f)
        second = self.mesh('Second', self.v + [0, 0, 3], self.f)
        other = self.make_rig('Other')
        other.data.bones['C'].use_deform = False
        second.modifiers[0].object = other
        targets = []
        for obj in [first, second]:
            self.weights(obj, ['C'], np.tile([0.4], (8, 1)))
            target = transfer.make_target(obj, bpy.context.evaluated_depsgraph_get(), self.settings)
            target['weights'] = np.tile([0.8, 0.2, 0.4], (8, 1))
            transfer.stage_weights(target, ['A', 'B', 'C'], [True] * 3)
            targets.append(target)
        transfer.synchronize_targets(targets, self.settings, ['A', 'B'], self.rig)
        transfer.write_targets(targets)
        np.testing.assert_allclose(self.read(second, ['C']), 0.4)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    addon.unregister()
    if not result.wasSuccessful():
        sys.exit(1)
