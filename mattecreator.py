#--------------------------------------------------------------
# Workflow:

'''

- Install Addon
- Install Torch
- Setup Output Folder
- Download and Path to Checkpoint
- Open Compositor -> N Panel
- Plug In Video and Clean Plate
- Hit "Create Matte" or whatever
	- Need options for whether to create a composite with alpha, or just the black and white matte
- NN does its job, creates our matte and saves it in the destination folder
- Plug into Compositor

'''




#--------------------------------------------------------------


#--------------------------------------------------------------
# Meta Dictionary
#--------------------------------------------------------------

bl_info = {
	'name' : 'MatteCreator',
	'author' : 'SceneFiller',
	'version' : (1, 0, 0),
	'blender' : (3, 3, 0),
	'location' : 'View3d > Tool',
	'warning' : '',
	'wiki_url' : '',
	'category' : '3D View',
}

#--------------------------------------------------------------
# Import
#--------------------------------------------------------------

import os
import bpy
import bpy_extras
import math 
from mathutils import Vector
from bpy_extras.image_utils import load_image
from bpy_extras import view3d_utils
from bpy_extras.io_utils import ImportHelper

#--------------------------------------------------------------
# Miscellaneous Functions
#--------------------------------------------------------------

def MATTECREATOR_FN_contextOverride(area_to_check):
	return [area for area in bpy.context.screen.areas if area.type == area_to_check][0]

#--------------------------------------------------------------
# Camera Projection Tools
#--------------------------------------------------------------		

# Functions ---------------------- 

def MATTECREATOR_FN_helloWorld(self, context):
	print('Hello world!')

# Classes ---------------------- 

class MATTECREATOR_OT_helloWorld(bpy.types.Operator):
	# Hello world!
	bl_idname = 'mattecreator.hello_world'
	bl_label = ''
	bl_options = {'REGISTER', 'UNDO'}
	bl_description = 'Hello world!'

	def execute(self, context):
		MATTECREATOR_FN_helloWorld(self, context)
		return {'FINISHED'}	

#--------------------------------------------------------------
# Interface
#--------------------------------------------------------------

# Classes ---------------------- 

class MATTECREATOR_PT_panelMain(bpy.types.Panel):
	bl_label = 'MatteCreator'
	bl_idname = 'MATTECREATOR_PT_panelMain'
	bl_space_type = 'VIEW_3D'
	bl_region_type = 'UI'
	bl_category = 'MatteCreator'

	def draw(self, context):
		layout = self.layout		

class MATTECREATOR_PT_panelHelloWorld(bpy.types.Panel):
	bl_label = 'Hello World!'
	bl_idname = 'MATTECREATOR_PT_panelHelloWorld'
	bl_space_type = 'VIEW_3D'
	bl_region_type = 'UI'
	bl_category = 'MatteCreator'
	bl_parent_id = 'MATTECREATOR_PT_panelMain'

	def draw(self, context):
		layout = self.layout
		row = layout.row()

#--------------------------------------------------------------
# Register 
#--------------------------------------------------------------

classes = ()

classes_interface = (MATTECREATOR_PT_panelMain, MATTECREATOR_PT_panelHelloWorld)

def register():

	# Register Classes
	for c in classes_interface:
		bpy.utils.register_class(c)

def unregister():

	# Unregister
	for c in reversed(classes_interface):
		bpy.utils.unregister_class(c)

if __name__ == '__main__':
	register()