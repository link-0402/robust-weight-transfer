# This file is part of Robust Weight Transfer for Blender.
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

bl_info = {
    "name": "Robust Weight Transfer",
    "author": "sentfromspacevr",
    "version": (1, 2, 2),
    "blender": (5, 2, 0),
    "doc_url": "https://jinxxy.com/SentFromSpaceVR/robust-weight-transfer",
    "location": "View3D > Sidebar > SENT Tab",
    "category": "Object",
}

import sys
import os

import bpy
import bmesh
import numpy as np
import webbrowser
import math
import importlib
import subprocess
import bpy.utils.previews

libs_path = os.path.join(os.path.dirname(__file__), "deps")
# Dependencies are installed directly into this add-on-owned directory.  This
# avoids Blender's read-only Python installation and the fragile --user /
# PYTHONUSERBASE combination used by older releases of the add-on.
# Keep this directory ahead of Blender's global site-packages and any other
# add-on dependency directories. `site.addsitedir` appends it, which can make
# Python reuse an incomplete SciPy copy from a previous add-on installation.
if libs_path in sys.path:
    sys.path.remove(libs_path)
sys.path.insert(0, libs_path)

DEPENDENCIES = ["robust_laplacian", "scipy"]
DEPENDENCY_PACKAGES = {
    "robust_laplacian": "robust-laplacian==1.0.0",
    "scipy": "scipy>=1.16.2,<2",
}
missing_deps = []
for module in DEPENDENCIES:
    try:
        importlib.import_module(module)
    except ImportError:
        missing_deps.append(DEPENDENCY_PACKAGES[module])

installed_deps = False

print(missing_deps)

if not missing_deps:
    import scipy as sp
    weighttransfer_module_name = f"{__name__}.weighttransfer"
    util_module_name = f"{__name__}.util"
    # Blender's legacy ZIP installer replaces files and immediately enables the
    # add-on in the same Python process. Evict cached helpers so an in-place
    # upgrade cannot combine a new __init__.py with old module implementations.
    sys.modules.pop(weighttransfer_module_name, None)
    sys.modules.pop(util_module_name, None)
    sys.modules.pop(f"{__name__}.seams", None)
    sys.modules.pop(f"{__name__}.transfer", None)
    importlib.invalidate_caches()
    weighttransfer = importlib.import_module('.weighttransfer', __name__)
    util = importlib.import_module('.util', __name__)
    seams = importlib.import_module('.seams', __name__)
    transfer = importlib.import_module('.transfer', __name__)
    from .weighttransfer import build_surface_bvh, find_matches_closest_surface, limit_mask, smooth_weigths


class RobustWeightTransfer(bpy.types.Operator):
    """Transfer Skin Weights Robust"""
    bl_idname = "object.skin_weight_transfer"
    bl_label = "Robust Weight Transfer"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        if context.mode != 'OBJECT' and context.mode != 'PAINT_WEIGHT': return False
        
        scene_settings: SceneSettingsGroup = context.scene.robust_weight_transfer_settings
        if not scene_settings.source_object: return False
        if not scene_settings.apply_to_selected and scene_settings.source_object == context.active_object: return False
        if scene_settings.group_selection == 'DEFORM_POSE_BONES':
            armature_mods = [mod for mod in scene_settings.source_object.modifiers if mod.type == "ARMATURE"]
            if len(armature_mods) != 1: return False # no armature modifier or more than one
            if not armature_mods[0].object: return False

            
        objs = lambda x: [obj for obj in x if obj != scene_settings.source_object and isinstance(obj.data, bpy.types.Mesh)]
        if scene_settings.apply_to_selected:
            target_objs = objs(context.selected_objects)
        else:
            if not context.object: return False
            target_objs = objs([context.object])
            
        if len(target_objs) == 0: return False
        if scene_settings.use_deformed_target and any(util.has_modifier(obj, *util.TOPOLOGY_MODS) for obj in target_objs): return False
        
        if not scene_settings.apply_to_selected:
            obj = target_objs[0]
            object_settings: ObjectSettingsGroup = obj.robust_weight_transfer_settings
            mask = object_settings.vertex_group
            if len(mask) > 0 and mask not in obj.vertex_groups: return False
            inpaint = object_settings.inpaint_group
            if len(inpaint) > 0 and  inpaint not in obj.vertex_groups: return False
        return True


    def execute(self, context: bpy.types.Context):
        settings = context.scene.robust_weight_transfer_settings
        source_original = settings.source_object
        target_objs = ([obj for obj in context.selected_objects
                        if obj != source_original and obj.type == 'MESH']
                       if settings.apply_to_selected else [context.object])
        depsgraph = context.evaluated_depsgraph_get()
        source = source_original.evaluated_get(depsgraph) if settings.use_deformed_source else source_original
        targets = []
        try:
            source_verts, source_triangles, source_normals = util.get_obj_arrs_world(source)
            surface_bvh = build_surface_bvh(source_verts, source_triangles)
            names = [g.name for g in source.vertex_groups]
            deform = [util.is_vertex_group_deform_bone(source, name) for name in names]
            included = deform if settings.group_selection == 'DEFORM_POSE_BONES' else [True] * len(names)
            if not any(included):
                raise ValueError(f'Source object {source.name} has no transferable vertex groups')
            source_weights = util.get_groups_arr(source, included)
            for obj in target_objs:
                target = transfer.make_target(obj, depsgraph, settings)
                transfer.match_target(target, source_verts, source_triangles, source_normals,
                                      source_weights, settings, surface_bvh)
                targets.append(target)
            fallback = transfer.source_armature(source_original)
            transfer.solve_targets(targets, settings, fallback, partial=settings.partial_reweight)
            for target in targets:
                transfer.process_weights(target, settings)
                transfer.stage_weights(target, names, included, apply_mask=not settings.apply_to_selected)
            counts = transfer.synchronize_targets(
                targets, settings, [name for name, use in zip(names, deform) if use], fallback)
            post_counts = transfer.postprocess_transfer_targets(
                targets, settings, [name for name, use in zip(names, deform) if use])
        except (ValueError, RuntimeError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}

        # All solves and final seam reconciliation must succeed before any weights
        # or diagnostic colors are committed to a target.
        transfer.write_targets(targets)
        if settings.draw_matched:
            for target in targets:
                matched = target['matched']
                if settings.partial_reweight:
                    matched = matched | (target['strength'] == 0)
                util.draw_debug_vertex_colors(target['obj'], matched)
            if isinstance(context.space_data, bpy.types.SpaceView3D):
                context.space_data.shading.type = 'SOLID'
                context.space_data.shading.color_type = 'VERTEX'
        transfer.report_seams(self, counts)
        transfer.report_partial(self, targets)
        transfer.report_postprocess(self, post_counts)
        self.report({'INFO'}, f'Weights transferred from {source.name} to {len(targets)} object(s)')
        return {'FINISHED'}


class SelectNonMatched(bpy.types.Operator):
    """Select Rejected Loose Parts"""
    bl_idname = "object.select_non_matched"
    bl_label = "Select Rejected Loose Parts"
    bl_description = "Select vertices in Edit Mode, that make Weight Inpainting fail"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        scene_settings: SceneSettingsGroup = context.scene.robust_weight_transfer_settings
        if context.mode != 'EDIT_MESH': return False
        if not context.active_object: return False
        if not scene_settings.source_object: return False
        if scene_settings.source_object == context.active_object: return False
        if scene_settings.use_deformed_target and util.has_modifier(context.active_object, *util.TOPOLOGY_MODS): return False
        return True

    def execute(self, context):
        active = context.active_object
        settings = context.scene.robust_weight_transfer_settings
        bpy.ops.object.mode_set(mode='OBJECT')
        try:
            depsgraph = context.evaluated_depsgraph_get()
            source_original = settings.source_object
            source = source_original.evaluated_get(depsgraph) if settings.use_deformed_source else source_original
            vertices, triangles, normals = util.get_obj_arrs_world(source)
            surface_bvh = build_surface_bvh(vertices, triangles)
            deform = [util.is_vertex_group_deform_bone(source, g.name) for g in source.vertex_groups]
            included = deform if settings.group_selection == 'DEFORM_POSE_BONES' else [True] * len(deform)
            if not any(included):
                raise ValueError(f'Source object {source.name} has no transferable vertex groups')
            source_weights = util.get_groups_arr(source, included)
            across = (settings.apply_to_selected and settings.seam_sync
                      and settings.seam_sync_across_objects and settings.virtual_merge)
            objects = ([obj for obj in context.selected_objects
                        if obj != source_original and obj.type == 'MESH'] if across else [active])
            targets = []
            for obj in objects:
                target = transfer.make_target(obj, depsgraph, settings)
                transfer.match_target(target, vertices, triangles, normals,
                                      source_weights, settings, surface_bvh)
                targets.append(target)
            batch = next(batch for batch in transfer.batches(
                targets, across, transfer.source_armature(source_original))
                if any(t['obj'] == active for t in batch))
            domain, offsets = transfer.prepare_batch(batch, settings)
            index = next(i for i, t in enumerate(batch) if t['obj'] == active)
            selects = domain.rejected[offsets[index]:offsets[index + 1]]
            if settings.partial_reweight:
                selects &= batch[index]['strength'] > 0
        except (ValueError, RuntimeError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        finally:
            bpy.ops.object.mode_set(mode='EDIT')
        mesh = bmesh.from_edit_mesh(active.data)
        mesh.verts.ensure_lookup_table()
        for vertex, selected in zip(mesh.verts, selects):
            vertex.select_set(bool(selected))
        mesh.select_flush(True)
        mesh.select_flush(False)
        bmesh.update_edit_mesh(active.data, destructive=False)
        self.report({'INFO'}, f'Selected {np.count_nonzero(selects)} out of {len(selects)} vertices.')
        return {'FINISHED'}


class Inpaint(bpy.types.Operator):
    """Inpaint"""
    bl_idname = "object.rwt_inpaint"
    bl_label = "Inpaint"
    bl_description = "Inpaint active object using the inpaint mask"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        if not context.active_object: return False
        if context.active_object.type != 'MESH': return False
        if context.mode != 'OBJECT' and context.mode != 'PAINT_WEIGHT': return False
        scene_settings: SceneSettingsGroup = context.scene.robust_weight_transfer_settings
        object_settings: ObjectSettingsGroup = context.active_object.robust_weight_transfer_settings
        if len(object_settings.inpaint_group) == 0 or object_settings.inpaint_group not in context.active_object.vertex_groups: return False
        
        if scene_settings.use_deformed_target and util.has_modifier(context.active_object, *util.TOPOLOGY_MODS): return False
        return True

    def execute(self, context):
        settings = context.scene.robust_weight_transfer_settings
        obj = context.active_object
        try:
            target = transfer.make_target(obj, context.evaluated_depsgraph_get(), settings)
            names = [g.name for g in obj.vertex_groups]
            deform = [util.is_vertex_group_deform_bone(obj, name) for name in names]
            if not any(deform):
                raise ValueError(f'{obj.name} has no deform weights to inpaint')
            mask = transfer.inpaint_mask(obj)
            weights = util.get_groups_arr(obj, deform)
            target.update(weights=weights, matched=~mask)
            transfer.solve_targets([target], settings)
            target['weights'][~mask] = weights[~mask]
            transfer.stage_weights(target, names, deform)
            target['protected'] = ~mask
            target['write_vertices'] = mask
            counts = transfer.synchronize_targets([target], settings,
                [name for name, use in zip(names, deform) if use])
        except (ValueError, RuntimeError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        transfer.write_targets([target])
        transfer.report_seams(self, counts)
        self.report({'INFO'}, 'Weights inpainted.')
        return {'FINISHED'}


class ObjectSettingsGroup(bpy.types.PropertyGroup):
    vertex_group: bpy.props.StringProperty(name='Mask Vertex Group')
    vertex_group_invert: bpy.props.BoolProperty(name='Invert')
    inpaint_group: bpy.props.StringProperty(name='Inpaint Vertex Group')
    inpaint_group_invert: bpy.props.BoolProperty(name='Invert Inpaint')
    inpaint_threshold: bpy.props.FloatProperty(name='Inpaint Binary Threshold', default=0.5, min=0, max=1)
    
def update_enforce_four_bone_limit(self, context):
    """Ensure the correct group selection and enforce constraints."""
    if self.enforce_four_bone_limit:
        self.group_selection = 'DEFORM_POSE_BONES'
    
class SceneSettingsGroup(bpy.types.PropertyGroup):
    source_object: bpy.props.PointerProperty(name='Source', type=bpy.types.Object, poll=lambda self, obj: obj.type == 'MESH')
    shape_key_mix: bpy.props.BoolProperty(name='Use Shape Key Mix', description='Uses the Shape of the Shape Key Mix to transfer the weights', default=True)
    max_distance: bpy.props.FloatProperty(
        name='Max Distance',
        description='Maximum world-space distance to the source surface for direct matches; also the outer limit when Partial Reweight is enabled',
        default=0.05,
        min=0,
        unit='LENGTH',
        subtype='DISTANCE')
    partial_reweight: bpy.props.BoolProperty(
        name='Partial Reweight',
        description='Only transfer within Max Distance of the source surface, preserving existing weights outside the range and on unsupported parts',
        default=False)
    partial_reweight_falloff: bpy.props.FloatProperty(
        name='Falloff Width',
        description='Percentage of Max Distance used to smoothly fade into existing weights at the outer edge; zero gives a hard cutoff',
        default=20, min=0, max=100, subtype='PERCENTAGE')
    max_normal_angle_difference: bpy.props.FloatProperty(
        name='Max Normal Difference',
        description='Maximum allowed vertex normal difference between source and destination vertex',
        default=math.radians(30),
        min=0,
        max=math.pi,
        precision=3,
        step=100,
        unit='ROTATION',
        subtype='ANGLE')
    flip_vertex_normal: bpy.props.BoolProperty(
        name='Flip Vertex Normal',
        description='Allow vertex normal flipped at 180° between source and destination vertex',
        default=True)
    smoothing_factor: bpy.props.FloatProperty(
        name='Smoothing factor',
        description='Smoothing factor used in the smoothing pass.',
        default=0.2,
        min=0,
        max=1,
        step=10)
    smoothing_repeat: bpy.props.IntProperty(
        name='Smoothing repeat',
        description='Amount of iterations of smoothing used in the smoothing pass',
        default=4,
        min=0)
    apply_to_selected: bpy.props.BoolProperty(
        name='Apply to all Selected Objects',
        description='Weight transfers the from the source object to all selected objects')
    use_modifier: bpy.props.BoolProperty(name='Use Modifier', description='Uses the Shape resulting from the source objects modifier stack', default=True)
    use_deformed_source: bpy.props.BoolProperty(name='Use Deformed Source', description='Uses the Shape resulting from the source object\'s modifier stack and shape keys', default=True)
    use_deformed_target: bpy.props.BoolProperty(name='Use Deformed Target', description='Uses the Shape resulting from the target object\'s modifier stack and shape keys', default=True)
    draw_matched: bpy.props.BoolProperty(
        name='Visualize Rejected Weights',
        description='Draws rejected weights as a pink to the vertex color layer "RBT Matched". After each transfer it will set the vertex color layer to active and change the Viewport Shading to Solid, with Color set to Attribute')
    enforce_four_bone_limit: bpy.props.BoolProperty(
        name='Limit Groups per Vertex',
        description='Limit a vertex to being influenced to a specific amount of groups. This is useful when a mesh will be exported to game engines like Unity, that normally only support 4 bones per vertex',
        default=True,
        update=update_enforce_four_bone_limit)
    group_selection: bpy.props.EnumProperty(
        name='Group Type',
        description='Select what subset of Vertex Group\'s should be transferred',
        items=[
            ('ALL_GROUPS', 'All Groups', 'Transfer all groups'),
            ('DEFORM_POSE_BONES', 'Deform Pose Bones', 'Only transfer deform pose bones, used by the Armature')
        ],
        default='DEFORM_POSE_BONES')
    dilation_repeat: bpy.props.IntProperty(
        name='Dilation repeat',
        description='Amount of iterations used to smooth the weight remove mask, that is used to limit the bone influence per vertex to 4',
        default=4,
        min=0)
    inpaint_mode: bpy.props.EnumProperty(
        name='Mode',
        description='Choose the Inpaint Mode',
        items=[
            ('POINT', 'Point', 'Object is remeshed internally. Weights can "flow" outside a mesh/loose part and more robust' ),
            ('SURFACE', 'Surface', 'Mesh is used as is. Weights "flow" only inside a mesh/loose part. More likely to fail compared to "Point"')
        ],
        default='POINT')
    virtual_merge: bpy.props.BoolProperty(
        name='Virtual Merge by Distance',
        description='Temporarily weld nearby vertices only for weight inpainting. The mesh topology, normals, UVs, shape keys, and modifiers are not changed',
        default=False)
    virtual_merge_distance: bpy.props.FloatProperty(
        name='Virtual Merge Distance',
        description='World-space distance used to connect loose parts during weight inpainting',
        default=0.0001,
        min=0,
        soft_max=0.01,
        precision=5,
        unit='LENGTH',
        subtype='DISTANCE')
    seam_sync: bpy.props.BoolProperty(
        name='Synchronize Seam Weights',
        description='Give nearby open borders of separate loose parts identical final deform weights; protected or incompatible seams are skipped',
        default=False)
    seam_distance: bpy.props.FloatProperty(
        name='Seam Distance',
        description='World-space tolerance between original, undeformed mesh borders; zero matches exact duplicates',
        default=0.0001, min=0, soft_max=0.01, precision=5,
        unit='LENGTH', subtype='DISTANCE')
    seam_sync_across_objects: bpy.props.BoolProperty(
        name='Across Selected Objects',
        description='Also synchronize compatible selected targets; with Virtual Merge enabled, share their temporary inpainting solve',
        default=False)
    balance_lr_weight_groups: bpy.props.BoolProperty(
        name='Balance L/R Weight Groups',
        description='Balance total weights for transferred .L/.R and _l/_r deform-bone pairs while preserving locked and masked weights',
        default=False)
    normalize_weights_after_transfer: bpy.props.BoolProperty(
        name='Normalize Weights After Transfer',
        description='Normalize all deform weights to one on transfer-touched vertices without changing locked groups',
        default=False)
    smoothing_enable: bpy.props.BoolProperty(
        name='Enable Smoothing',
        description='Smooths weights in the area where weights got inpainted',
        default=False)
    smooth_limit_debug: bpy.props.BoolProperty(
        name='Limited vertices to Vertex Group',
        description='Visualize the vertices that got limited by writing to the "Limited" vertex group',
        default=False)
    num_limit_groups: bpy.props.IntProperty(
        name="Max groups per vertex",
        description="Amount of groups a vertex should be limited to. For VRChat/Unity keep it at 4.",
        min=1,
        default=4
    )


class RobustWeightTransferPanel(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Robust Weight Transfer"
    bl_idname = "OBJECT_PT_robust_weight_transfer_panel"
    bl_space_type = 'VIEW_3D'   # Defines the space type where the panel is located
    bl_region_type = 'UI'       # Specifies that the panel is drawn in the UI region
    bl_category = 'SENT'      # The name of the tab the panel will be in
    bl_options = set()

    def draw(self, context): 
        layout = self.layout

        if missing_deps:
            box = layout.box()
            col = box.column()
            global installed_deps
            if installed_deps:
                col.label(text="Dependencies installed!", icon='INFO')
                col.label(text="Restart Blender!", icon='ERROR')
                installed_deps = True
                return
            
            col.label(text="Blender will be unreactive while installing")
            col.operator("wm.install_rwt_dependencies", icon='IMPORT')
            col.label(text="This might take a few minutes", icon='INFO')
            return

        active_obj = context.object
        if not context.object:
            layout.label(text='No active object selected.')
            return
        
        props = active_obj.robust_weight_transfer_settings
        settings = context.scene.robust_weight_transfer_settings
        
        # Object field for source
        row = layout.row(align=True)
        row.prop(settings, "source_object")
        row.prop(settings, "use_deformed_source", toggle=True, text="", icon='MODIFIER')
        
        # Vertex group field
        row = layout.row()
        row.label(text='Transfer Mask')
        row = row.row(align=True)
        row.prop_search(props, "vertex_group", active_obj, "vertex_groups", text='')
        row.prop (props , "vertex_group_invert",text="", toggle=True, icon='ARROW_LEFTRIGHT')
        row.enabled = not settings.apply_to_selected
        
        objs = lambda x: [obj for obj in x if obj != settings.source_object and isinstance(obj.data, bpy.types.Mesh)]
        if settings.apply_to_selected:
            target_objs = objs(context.selected_objects)
        else:
            target_objs = objs([context.object])
        if (len(target_objs) > 0
                and settings.use_deformed_target
                and any(util.has_modifier(obj, *util.TOPOLOGY_MODS) for obj in target_objs)):
            objs_str = ', '.join(obj.name for obj in target_objs)
            col = layout.column(align=True)
            col.label(text=f'Error: {objs_str}', icon='ERROR')
            col.label(text='  Topology altering Modifier!', icon='SHAPEKEY_DATA')
            col.label(text='  Deactivate Use Deformed Target or apply/delete modifier.', icon='MODIFIER')
            
        source_obj = settings.source_object
        if source_obj and settings.group_selection == 'DEFORM_POSE_BONES':
            armature_mods = [mod for mod in source_obj.modifiers if mod.type == "ARMATURE"]
            if len(armature_mods) == 0:
                col = layout.column(align=True)
                col.label(text=f'Subset is set to Deform Pose Bones,', icon='ERROR')
                col.label(text=f'but {source_obj.name} has no Armature Modifier')
            elif len(armature_mods) == 1:
                if not armature_mods[0].object:
                    col = layout.column(align=True)
                    col.label(text=f'Subset is set to Deform Pose Bones,', icon='ERROR')
                    col.label(text=f'but {source_obj.name} has an empty Armature Modifier object')
            else:
                col = layout.column(align=True)
                col.label(text=f'Subset is set to Deform Pose Bones,', icon='ERROR')
                col.label(text=f'but {source_obj.name} has multiple Armature Modifiers')
            
        row = layout.row(align=True)
        row.prop(settings, 'apply_to_selected', text='', icon='RESTRICT_SELECT_OFF')
        row.operator("object.skin_weight_transfer", text="Transfer Weights")
        row.prop(settings, "use_deformed_target", toggle=True, text="", icon='MODIFIER')
        
        layout.separator(factor=1)
        
        col = layout.column()
        col.label(text='Inpaint Mask')
        row = col.row(align=True)
        row.prop_search(props, "inpaint_group", active_obj, 'vertex_groups', text='')
        row.prop(props , "inpaint_group_invert",text="", toggle=True, icon='ARROW_LEFTRIGHT')
        row.prop(props, 'inpaint_threshold', text="")
        row.enabled = not settings.apply_to_selected

        
class SettingsPanel(bpy.types.Panel):
    bl_idname = 'OBJECT_PT_robust_weight_transfer_settings_panel'
    bl_label = 'Settings'
    bl_space_type = 'VIEW_3D'   # Defines the space type where the panel is located
    bl_region_type = 'UI'       # Specifies that the panel is drawn in the UI region
    bl_category = 'SENT'      # The name of the tab the panel will be in
    bl_parent_id = 'OBJECT_PT_robust_weight_transfer_panel'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.robust_weight_transfer_settings
        layout.operator('object.rbt_reset_scene_settings', icon='LOOP_BACK', text='Reset to Defaults')
        layout.prop(settings, 'inpaint_mode')
        layout.prop(settings, 'virtual_merge')
        row = layout.row()
        row.enabled = settings.virtual_merge
        row.prop(settings, 'virtual_merge_distance')
        layout.prop(settings, 'seam_sync')
        row = layout.row()
        row.enabled = settings.seam_sync
        row.prop(settings, 'seam_distance')
        row = layout.row()
        row.enabled = settings.seam_sync and settings.apply_to_selected
        row.prop(settings, 'seam_sync_across_objects')
        layout.prop(settings, 'balance_lr_weight_groups')
        layout.prop(settings, 'normalize_weights_after_transfer')
        layout.prop(settings, 'draw_matched')
        row = layout.row()
        row.enabled = not settings.enforce_four_bone_limit
        row.prop(settings, 'group_selection', text='Subset')


class VertexMappingPanel(bpy.types.Panel):
    bl_label = "Vertex Mapping"
    bl_idname = "OBJECT_PT_vertex_mapping"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SENT'
    bl_parent_id = 'OBJECT_PT_robust_weight_transfer_settings_panel'

    def draw(self, context):
        layout = self.layout
        settings = context.scene.robust_weight_transfer_settings
        layout.prop(settings, "max_distance")
        layout.prop(settings, "partial_reweight")
        row = layout.row()
        row.enabled = settings.partial_reweight
        row.prop(settings, "partial_reweight_falloff")
        layout.prop(settings, "max_normal_angle_difference")
        layout.prop(settings, "flip_vertex_normal", text='Allow Flipped Vertex Normals')


class SmoothingPanel(bpy.types.Panel):
    bl_label = "Smoothing"
    bl_idname = "OBJECT_PT_smoothing"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SENT'
    bl_parent_id = 'OBJECT_PT_robust_weight_transfer_settings_panel'

    def draw(self, context):
        layout = self.layout
        settings = context.scene.robust_weight_transfer_settings
        layout.enabled = settings.smoothing_enable
        layout.prop(settings, 'smoothing_repeat')
        layout.prop(settings, 'smoothing_factor')
        
    def draw_header(self, context: bpy.types.Context):
        settings = context.scene.robust_weight_transfer_settings
        col = self.layout.column(align=True)
        col.prop(settings, 'smoothing_enable', text='')

class LimitGroupsPanel(bpy.types.Panel):
    bl_label = "Limit Vertex Groups"
    bl_idname = "OBJECT_PT_limit_groups"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SENT'
    bl_parent_id = 'OBJECT_PT_robust_weight_transfer_settings_panel'

    def draw(self, context):
        layout = self.layout
        settings = context.scene.robust_weight_transfer_settings
        layout.enabled = settings.enforce_four_bone_limit
        layout.prop(settings, 'num_limit_groups')
        
    def draw_header(self, context: bpy.types.Context):
        settings = context.scene.robust_weight_transfer_settings
        col = self.layout.column(align=True)
        col.prop(settings, 'enforce_four_bone_limit', text='')


class ResetSceneSettings(bpy.types.Operator):
    """Reset all settings to their default values"""
    bl_idname = "object.rbt_reset_scene_settings"
    bl_label = "Reset Robust Weight Transfer to Default Settings"

    def execute(self, context):
        settings = context.scene.robust_weight_transfer_settings
        for prop_name, prop in settings.bl_rna.properties.items():
            if prop.is_readonly or prop_name in {'rna_type', 'name'}:
                continue
            if hasattr(prop, 'default'):
                setattr(settings, prop_name, prop.default)
            else:
                setattr(settings, prop_name, None)
        return {'FINISHED'}


class UtilitiesPanel(bpy.types.Panel):
    bl_idname = 'OBJECT_PT_robust_weight_utilities_settings_panel'
    bl_label = 'Utilities'
    bl_space_type = 'VIEW_3D'   # Defines the space type where the panel is located
    bl_region_type = 'UI'       # Specifies that the panel is drawn in the UI region
    bl_category = 'SENT'      # The name of the tab the panel will be in
    bl_parent_id = 'OBJECT_PT_robust_weight_transfer_panel'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.robust_weight_transfer_settings
        layout.operator('object.select_non_matched')
        row = layout.row(align=True)
        row.operator('object.smooth_limit_weights')
        row.prop(settings, 'smooth_limit_debug', text='', icon='GROUP_VERTEX')
        layout.operator('object.rwt_inpaint')
    

class SmoothLimit(bpy.types.Operator):
    """Limit weights of to a specific amount"""
    bl_idname = "object.smooth_limit_weights"
    bl_label = "Smoothed Limit Vertex Groups"
    bl_description = "Limits the amount weights per vertex of the active object"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        if not context.active_object: return False
        if context.mode != 'OBJECT' and context.mode != 'PAINT_WEIGHT': return False
        if context.active_object.type != 'MESH': return False
        
        return True
    
    def execute(self, context):
        scene_settings = context.scene.robust_weight_transfer_settings
        obj = context.active_object
        is_deform = [util.is_vertex_group_deform_bone(obj, g.name) for g in obj.vertex_groups]
        if not np.any(is_deform):
            self.report({'ERROR'}, f'{obj.name} has no deform-bone vertex groups to limit')
            return {'CANCELLED'}
        W = util.get_groups_arr(obj, is_deform)
        adj_mat = util.get_mesh_adjacency_matrix_sparse(obj.data, True)
        mask = limit_mask(W, adj_mat, limit_num=scene_settings.num_limit_groups)
        W = (1 - mask) * W
        util.write_weights(obj, W[:, is_deform], [group.name for i, group in enumerate(obj.vertex_groups) if is_deform[i]], 0.0001)
        if scene_settings.smooth_limit_debug:
            limited = np.max(mask, axis=1)
            util.write_weights(obj, limited[:, np.newaxis], ['Limited'])
        return {'FINISHED'}

class InstallDependencies(bpy.types.Operator):
    """Install missing Python dependencies"""
    bl_idname = "wm.install_rwt_dependencies"
    bl_label = "Install Dependencies"
    
    def execute(self, context):
        python_exe = sys.executable
        os.makedirs(libs_path, exist_ok=True)
        command = [
            python_exe, "-m", "pip", "install",
            "--target", libs_path,
            "--upgrade",
            "--only-binary=:all:",
            "--no-deps",
            *missing_deps,
        ]
        print("Robust Weight Transfer dependency command:", command)
        try:
            result = subprocess.run(
                command,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            print(result.stdout)
            self.report({'INFO'}, "Installation successful! Please restart Blender.")
            global installed_deps
            installed_deps = True
            return {'FINISHED'}
        except subprocess.CalledProcessError as e:
            details = (e.stdout or str(e)).strip()
            print("Robust Weight Transfer dependency installation failed:\n", details)
            last_line = details.splitlines()[-1] if details else str(e)
            self.report({'ERROR'}, f"Dependency installation failed: {last_line}. See System Console.")
            return {'CANCELLED'}
        except OSError as e:
            print("Robust Weight Transfer could not start pip:", e)
            self.report({'ERROR'}, f"Could not start Blender Python/pip: {e}")
            return {'CANCELLED'}

class SentFromSpacePanel(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Sent From Space (@sentfromspacevr)"
    bl_idname = "VIEW3D_PT_sent_from_space_panel"
    bl_space_type = 'VIEW_3D'   # Defines the space type where the panel is located
    bl_region_type = 'UI'       # Specifies that the panel is drawn in the UI region
    bl_category = 'SENT'      # The name of the tab the panel will be in
    bl_options = set()
    bl_order = 1000
    version = 0
    registered_panel  = False
    pcoll = None
    
    class OpenThirdOperator(bpy.types.Operator):
        """Open Third in the web browser"""
        bl_idname = "wm.open_third"
        bl_label = "third3d.com"

        def execute(self, context):
            webbrowser.open("https://third3d.com/")
            return {'FINISHED'}

    # Operator to open Jinxxy
    class OpenJinxxyOperator(bpy.types.Operator):
        """Open Jinxxy in the web browser"""
        bl_idname = "wm.open_gumroad"
        bl_label = "Jinxxy"

        def execute(self, context):
            webbrowser.open("https://jinxxy.com/SentFromSpaceVR/robust-weight-transfer")
            return {'FINISHED'}
        
    # Operator to open Discord
    class OpenDiscordOperator(bpy.types.Operator):
        """Open Discord in the web browser"""
        bl_idname = "wm.open_discord"
        bl_label = "Discord"

        def execute(self, context):
            webbrowser.open("https://discord.gg/Fdy5RpunY4")
            return {'FINISHED'}

    # Operator to open Twitter
    class OpenTwitterOperator(bpy.types.Operator):
        """Open Twitter in the web browser"""
        bl_idname = "wm.open_twitter"
        bl_label = "Twitter"

        def execute(self, context):
            webbrowser.open("https://twitter.com/sentfromspacevr")
            return {'FINISHED'}

    # Operator to open GitHub
    class OpenGitHubOperator(bpy.types.Operator):
        """Open GitHub in the web browser"""
        bl_idname = "wm.open_github"
        bl_label = "GitHub"

        def execute(self, context):
            webbrowser.open("https://github.com/sentfromspacevr")
            return {'FINISHED'}
        
    @classmethod
    def _register(cls):
        cls.registered_panel = True
        bpy.utils.register_class(cls.OpenThirdOperator)
        bpy.utils.register_class(cls.OpenJinxxyOperator)
        bpy.utils.register_class(cls.OpenDiscordOperator)
        bpy.utils.register_class(cls.OpenTwitterOperator)
        bpy.utils.register_class(cls.OpenGitHubOperator)
        bpy.utils.register_class(cls)

        logo_path = os.path.join(os.path.dirname(__file__), "third-logo-icon.png")
        
        pcoll = bpy.utils.previews.new()
        pcoll.load("third_logo", logo_path, "IMAGE")
        cls.pcoll = pcoll
        
    
    @classmethod
    def _unregister(cls):
        if cls.registered_panel:
            cls.registered_panel = False
            bpy.utils.unregister_class(cls.OpenThirdOperator)
            bpy.utils.unregister_class(cls.OpenJinxxyOperator)
            bpy.utils.unregister_class(cls.OpenDiscordOperator)
            bpy.utils.unregister_class(cls.OpenTwitterOperator)
            bpy.utils.unregister_class(cls.OpenGitHubOperator)
            bpy.utils.unregister_class(cls)
            bpy.utils.previews.remove(cls.pcoll)
    
        
    def draw(self, context):
        layout = self.layout
        layout.operator("wm.open_third", text="third3d.com   ", icon_value=self.__class__.pcoll["third_logo"].icon_id)
        col = layout.column(align=True)
        row = col.row(align=True)
        row.operator("wm.open_discord", text="Discord")
        row.operator("wm.open_gumroad", text="Jinxxy")
        row = col.row(align=True)
        row.operator("wm.open_twitter", text="Twitter")
        row.operator("wm.open_github", text="GitHub")
    

def register():
    # bpy.types.VIEW3D_MT_make_links.append(menu_func)
    bpy.utils.register_class(RobustWeightTransferPanel)
    if missing_deps:
        bpy.utils.register_class(InstallDependencies)
    else:
        bpy.utils.register_class(ObjectSettingsGroup)
        bpy.utils.register_class(SceneSettingsGroup)
        bpy.types.Object.robust_weight_transfer_settings = bpy.props.PointerProperty(type=ObjectSettingsGroup)
        bpy.types.Scene.robust_weight_transfer_settings = bpy.props.PointerProperty(type=SceneSettingsGroup)
        bpy.utils.register_class(RobustWeightTransfer)
        bpy.utils.register_class(SettingsPanel)
        bpy.utils.register_class(VertexMappingPanel)
        bpy.utils.register_class(LimitGroupsPanel)
        bpy.utils.register_class(SmoothingPanel)
        bpy.utils.register_class(SelectNonMatched)
        bpy.utils.register_class(ResetSceneSettings)
        bpy.utils.register_class(UtilitiesPanel)
        bpy.utils.register_class(SmoothLimit)
        bpy.utils.register_class(Inpaint)
        
    if 'VIEW3D_PT_sent_from_space_panel' in dir(bpy.types):
        if SentFromSpacePanel.version > bpy.types.VIEW3D_PT_sent_from_space_panel.version:
            bpy.types.VIEW3D_PT_sent_from_space_panel.unregister()
            SentFromSpacePanel._register()
    else:
        SentFromSpacePanel._register()
    # bpy.utils.register_class(SentFromSpacePanel)
    
    
def unregister():
    # bpy.types.VIEW3D_MT_make_links.remove(menu_func)
    SentFromSpacePanel._unregister()
    if missing_deps:
        bpy.utils.unregister_class(InstallDependencies)
    else:
        bpy.utils.unregister_class(Inpaint)
        bpy.utils.unregister_class(SmoothLimit)
        bpy.utils.unregister_class(UtilitiesPanel)
        bpy.utils.unregister_class(ResetSceneSettings)
        bpy.utils.unregister_class(SelectNonMatched)
        bpy.utils.unregister_class(SmoothingPanel)
        bpy.utils.unregister_class(LimitGroupsPanel)
        bpy.utils.unregister_class(VertexMappingPanel)
        bpy.utils.unregister_class(SettingsPanel)
        bpy.utils.unregister_class(RobustWeightTransfer)
        del bpy.types.Object.robust_weight_transfer_settings
        del bpy.types.Scene.robust_weight_transfer_settings
        bpy.utils.unregister_class(SceneSettingsGroup)
        bpy.utils.unregister_class(ObjectSettingsGroup)
    bpy.utils.unregister_class(RobustWeightTransferPanel)
    

if __name__ == "__main__":
    register()
