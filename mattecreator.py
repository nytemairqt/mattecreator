# MIT License

# Copyright (c) 2020 University of Washington

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Original work by Shanchuan Lin, Andrey Ryabtsev, Soumyadip Sengupta, Brian Curless, Steve Seitz, Ira Kemelmacher-Shlizerman
# https://grail.cs.washington.edu/projects/background-matting-v2/#/

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

MATTECREATOR_MISSING_DEPENDENCIES = True

import os
import bpy
import sys
import subprocess
import platform
from time import sleep

import bpy_extras
import math 
import importlib 

from mathutils import Vector
from bpy_extras import view3d_utils
from bpy_extras.io_utils import ImportHelper
from bpy_extras.image_utils import load_image

try:
	import torch 
	from torch import nn 
	from torch.utils.data import DataLoader

	import torchvision 
	from torchvision import transforms as T 
	from torchvision.transforms import functional as F 

	import numpy as np
	import PIL
	import cv2

	MATTECREATOR_MISSING_DEPENDENCIES = False
except: 
	MATTECREATOR_MISSING_DEPENDENCIES = True

from threading import Thread

# Functions ---------------------- 

def MATTECREATOR_FN_getOS():
	if os.name == 'nt':
		return 'windows'
	elif os.name == 'posix' and platform.system() == "Darwin":
		return 'macOS'
	elif os.name == 'posix' and platform.system() == "Linux":
		return 'linux'

def MATTECREATOR_FN_pythonExecutable(OS):
	if OS == 'windows':
		return os.path.join(sys.prefix, 'bin', 'python.exe')
	elif OS == 'macOS':
		return os.path.abspath(sys.executable)
	elif OS == 'linux':
		return os.path.join(sys.prefix, 'sys.prefix/bin', 'python')
	else:
		return None

def MATTECREATOR_FN_installPythonPackage(package, executable, OS):	
	# Update PIP
	try:
		subprocess.call([executable, '-m', 'ensurepip'])
		subprocess.call([executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
	except:
		print('There was an issue with Pip, please try again or install packages manually.')
		print(f'Python.exe location: {executable}')
	# Install Torch with CUDA Support	
	if package == 'torch':
		if OS == 'windows':
			try:		
				subprocess.call([executable, '-m', 'pip', 'install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu117'])
			except:
				print('There was an issue installing the Library.')
		elif OS == 'macOS':
			try:		
				subprocess.cal([executable, 'pip3', 'install', 'torch', 'torchvision'])
			except:
				print('There was an issue installing the Library.')
		elif OS == 'linux':
			try:		
				subprocess.call([executable, 'pip3', 'install', 'torch', 'torchvision'])
			except:
				print('There was an issue installing the Library.')
	else:
		try:
			subprocess.call([executable, '-m', 'pip', 'install', package])
		except:
			print('There was an issue installing the Library, please try again or install packages manually.')
			print(f'Python.exe location: {executable}')
		
MATTECREATOR_OPERATING_SYSTEM = MATTECREATOR_FN_getOS()
MATTECREATOR_PYTHON_EXECUTABLE = MATTECREATOR_FN_pythonExecutable(MATTECREATOR_OPERATING_SYSTEM)
MATTECREATOR_PYTHON_DEPENDENCIES = ['opencv-python', 'pillow', 'torch', 'numpy']

def MATTECREATOR_FN_loadVideo(path):
	cap = cv2.VideoCapture(path)

	w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	frame_rate = cap.get(cv2.CAP_PROP_FPS)
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	return cap

def MATTECREATOR_FN_loadImage(path):
	bgr = PIL.Image.open(path).convert('RGB')
	bgr = np.array(bgr)
	return [bgr]

def MATTECREATOR_FN_zipDataSet(datasets, transforms=None, assert_equal_length=False):
	if assert_equal_length:
		for i in range(1, len(datasets)):
			assert len(datasets[i]) == len(datasets[i - 1]), 'Datasets are not equal in length.'

def MATTECREATOR_FN_extractMatte(self, context):
	VIDEO_RESIZE = None
	PREPROCESS_ALIGNMENT = None

	model_path = 'D:/Documents/Scenefiller/mattecreator/model/torchscript_resnet50_fp16.pth'
	output_dir = 'D:/Documents/Scenefiller/mattecreator/output/'

	# Reference SRC & BGR Files	
	try:
		video_path = bpy.path.abspath(bpy.context.scene.MATTECREATOR_VAR_videoSource.filepath)
	except:
		self.report({'WARNING'}, 'Please select a valid foreground Video File.')
		return{'CANCELLED'}
	try:
		image_path = bpy.path.abspath(bpy.context.scene.MATTECREATOR_VAR_cleanPlate.filepath)	
	except:
		self.report({'WARNING'}, 'Please select a valid background Image Plate.')
		return{'CANCELLED'}
		

	# Instantiate Model
	if bpy.context.scene.MATTECREATOR_HYPERPARAM_device == 'cpu':
		device = torch.device('cpu')
	else:
		device = torch.device('cuda')

	precision = torch.float16
	model = torch.jit.load(model_path)
	model.backbone_scale = 0.25
	model.refine_mode = 'sampling'
	model.refine_sample_pixels = 80000
	model = model.to(device).eval()

	# Load Data
	vid = MATTECREATOR_CLASS_videoDataset(video_path)
	bgr = MATTECREATOR_FN_loadImage(image_path)

	dataset = MATTECREATOR_CLASS_zipDataset([vid, bgr], transforms=MATTECREATOR_CLASS_pairCompose([
    MATTECREATOR_CLASS_pairApply(T.Resize(VIDEO_RESIZE[::-1]) if VIDEO_RESIZE else nn.Identity()),
    MATTECREATOR_CLASS_homographicAlignment() if PREPROCESS_ALIGNMENT else MATTECREATOR_CLASS_pairApply(nn.Identity()),
    MATTECREATOR_CLASS_pairApply(T.ToTensor())]))

	w = vid.width
	h = vid.height

	# Instantiate Writers
	fgr_writer = MATTECREATOR_CLASS_videoWriter(os.path.join(output_dir, 'fgr.mp4'), vid.frame_rate, w, h)
	com_writer = MATTECREATOR_CLASS_videoWriter(os.path.join(output_dir, 'com.mp4'), vid.frame_rate, w, h)	

	bpy.ops.wm.console_toggle()

	with torch.no_grad():
		for idx, input_batch in enumerate(DataLoader(dataset, batch_size=1, pin_memory=True)):
			src, bgr = input_batch
			tgt_bgr = torch.tensor([120/255, 255/255, 155/255], device=device).view(1, 3, 1, 1)

			src = src.to(precision).to(device, non_blocking=True)
			bgr = bgr.to(precision).to(device, non_blocking=True)

			if context.scene.MATTECREATOR_HYPERPARAM_modelType == 'mattingbase':
				pha, fgr, err, _ = model(src, bgr)
			elif context.scene.MATTECREATOR_HYPERPARAM_modelType == 'mattingrefine':
				pha, fgr, _, _, err, ref = model(src, bgr)
			elif context.scene.MATTECREATOR_HYPERPARAM_modelType == 'OP3':
				pha, fgr = model(src, bgr)

			com = fgr * pha + tgt_bgr * (1 - pha)

			fgr_writer.add_batch(fgr)
			com_writer.add_batch(com)
			print(f'Writing frame... {idx} / {vid.frame_count}')	

	print('Finished writing.')
	bpy.ops.wm.console_toggle()	

# Classes ---------------------- 

class MATTECREATOR_OT_installPackages(bpy.types.Operator):
	bl_idname = 'mattecreator.install_packages'
	bl_label = ''
	bl_options = {'REGISTER', 'UNDO'}
	bl_description = 'Install necessary packages.'

	def execute(self, context):
		bpy.ops.wm.console_toggle()
		for dependency in MATTECREATOR_PYTHON_DEPENDENCIES:
			MATTECREATOR_FN_installPythonPackage(dependency, MATTECREATOR_PYTHON_EXECUTABLE, MATTECREATOR_OPERATING_SYSTEM)
		print('-----------------------------------------------------------------------------')
		print('All dependencies installed successfully, please restart Blender.')
		return{'FINISHED'}

class MATTECREATOR_OT_setModelCheckpointPath(bpy.types.Operator, ImportHelper):
	bl_idname = 'mattecreator.set_model_checkpoint_path'
	bl_label = ''
	bl_options = {'REGISTER', 'UNDO'}
	bl_description = 'Select a checkpoint file to load into the Model.'

	filter_glob: bpy.props.StringProperty(
			default='*.pth;',
			options={'HIDDEN'}
		)

	def execute(self, context):
		bpy.context.scene.MATTECREATOR_VAR_modelPath = self.properties.filepath 

class MATTECREATOR_OT_setOutputDirPath(bpy.types.Operator, ImportHelper):
	bl_idname = 'mattecreator.set_output_dir_path'
	bl_label = ''
	bl_options = {'REGISTER', 'UNDO'}
	bl_description = 'Select a folder to export generated matte.'

	def execute(self, context):
		bpy.context.scene.MATTECREATOR_VAR_outputDir = self.properties.filepath 	


class MATTECREATOR_OT_extractMatte(bpy.types.Operator):
	# Hello world!
	bl_idname = 'mattecreator.extract_matte'
	bl_label = ''
	bl_options = {'REGISTER', 'UNDO'}
	bl_description = 'Run Neural forward pass to extract and save Matte.'

	def execute(self, context):		
		MATTECREATOR_FN_extractMatte(self, context)		
		return {'FINISHED'}	


class MATTECREATOR_CLASS_videoWriter:
	def __init__(self, path, frame_rate, width, height):
		self.out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

	def add_batch(self, frames):
		frames = frames.mul(255).byte()
		frames = frames.cpu().permute(0, 2, 3, 1).numpy()
		for i in range(frames.shape[0]):
			frame = frames[i]
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
			self.out.write(frame)

class MATTECREATOR_CLASS_imageSequenceWriter:
	def __init__(self, path, extension):
		self.path = path
		self.extension = extension
		self.index = 0
		os.makedirs(path)

	def add_batch(self, frames):
		Thread(target=self._add_batch, args=(frames, self.index)).start()
		self.index += frames.shape[0]

	def _add_batch(self, frames, index):
		frames = frames.cpu()
		for i in range(frames.shape[0]):
			frame = frames[i]
			frame = to_pil_image(frame) # NEED TO INSTANTIATE
			frame.save(os.path.join(self.path, str(index + i).zfill(5) + '.' + self.extension))	


if not MATTECREATOR_MISSING_DEPENDENCIES:
	class MATTECREATOR_CLASS_videoDataset(torch.utils.data.Dataset):
		def __init__(self, path: str, transforms: any = None):
			self.cap = cv2.VideoCapture(path)
			self.transforms = transforms

			self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
			self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

		def __len__(self):
			return self.frame_count

		def __getitem__(self, idx):
			if isinstance(idx, slice):
				return [self[i] for i in range(*idx.indices(len(self)))]

			if self.cap.get(cv2.CAP_PROP_POS_FRAMES) != idx:
				self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
			ret, img = self.cap.read()
			if not ret:
				raise IndexError(f'Idx: {idx} out of length: {len(self)}')
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			#img = PIL.Image.fromarray(img) #DO I NEED THIS????
			if self.transforms:
				img = self.transforms(img)
			return img

		def __enter__(self):
			return self

		def __exit__(self, exc_type, exc_value, exc_traceback):
			self.cap.release()	
				
	class MATTECREATOR_CLASS_zipDataset(torch.utils.data.Dataset):
		def __init__(self, datasets, transforms=None, assert_equal_length=False):
			self.datasets = datasets
			self.transforms = transforms
	        
			if assert_equal_length:
				for i in range(1, len(datasets)):
					assert len(datasets[i]) == len(datasets[i - 1]), 'Datasets are not equal in length.'
	    
		def __len__(self):
			return max(len(d) for d in self.datasets)
	    
		def __getitem__(self, idx):
			x = tuple(d[idx % len(d)] for d in self.datasets)
			if self.transforms:
				x = self.transforms(*x)
			return x    	
	

#--------------------------------------------------------------
# Transforms
#--------------------------------------------------------------

if not MATTECREATOR_MISSING_DEPENDENCIES:
	class MATTECREATOR_CLASS_pairCompose(torchvision.transforms.Compose):
		def __call__(self, *x):
			for transform in self.transforms:
				x = transform(*x)
			return x

class MATTECREATOR_CLASS_pairApply:
	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, *x):
		return [self.transforms(xi) for xi in x]

class MATTECREATOR_CLASS_homographicAlignment:    
	def __init__(self):
		self.detector = cv2.ORB_create()
		self.matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)

	def __call__(self, src, bgr):
		src = np.asarray(src)
		bgr = np.asarray(bgr)

		keypoints_src, descriptors_src = self.detector.detectAndCompute(src, None)
		keypoints_bgr, descriptors_bgr = self.detector.detectAndCompute(bgr, None)

		matches = self.matcher.match(descriptors_bgr, descriptors_src, None)
		matches.sort(key=lambda x: x.distance, reverse=False)
		num_good_matches = int(len(matches) * 0.15)
		matches = matches[:num_good_matches]

		points_src = np.zeros((len(matches), 2), dtype=np.float32)
		points_bgr = np.zeros((len(matches), 2), dtype=np.float32)
		for i, match in enumerate(matches):
			points_src[i, :] = keypoints_src[match.trainIdx].pt
			points_bgr[i, :] = keypoints_bgr[match.queryIdx].pt

		H, _ = cv2.findHomography(points_bgr, points_src, cv2.RANSAC)

		h, w = src.shape[:2]
		bgr = cv2.warpPerspective(bgr, H, (w, h))
		msk = cv2.warpPerspective(np.ones((h, w)), H, (w, h))

		bgr[msk != 1] = src[msk != 1]

		src = PIL.Image.fromarray(src)
		bgr = PIL.Image.fromarray(bgr)

		return src, bgr		

#--------------------------------------------------------------
# Interface
#--------------------------------------------------------------

# Classes ---------------------- 

class MATTECREATOR_PT_panelMain(bpy.types.Panel):
	bl_label = 'MatteCreator'
	bl_idname = 'MATTECREATOR_PT_panelMain'
	bl_space_type = 'NODE_EDITOR'
	bl_region_type = 'UI'
	bl_category = 'MatteCreator'

	@classmethod
	def poll(cls, context):
		snode = context.space_data
		return snode.tree_type == 'CompositorNodeTree'

	def draw(self, context):
		layout = self.layout	

		row = layout.row()
		row.label(text='Input', icon='REMOVE')

		row = layout.row() # Additional Spacing
		row = layout.row() # Additional Spacing

		# Video SRC
		row = layout.row()
		row.prop(context.scene, "MATTECREATOR_VAR_videoSource", text='Video File')

		# Clean Plate Image
		row = layout.row()
		row.prop(context.scene, 'MATTECREATOR_VAR_cleanPlate', text='Clean Plate')

		# Model Checkpoint
		row = layout.row()
		row.label(text='Model Checkpoint')
		row.label(text=bpy.context.scene.MATTECREATOR_VAR_modelPath)
		row.operator(MATTECREATOR_OT_setModelCheckpointPath.bl_idname, text='', icon='FILE_FOLDER')

		# Device
		row = layout.row()
		row.prop(context.scene, 'MATTECREATOR_HYPERPARAM_device', text='Device')

		row = layout.row() # Additional Spacing
		row = layout.row() # Additional Spacing

		row = layout.row()
		row.label(text='Output', icon='REMOVE')	

		row = layout.row() # Additional Spacing
		row = layout.row() # Additional Spacing
		
		# Output Directory
		row = layout.row()
		row.prop(context.scene, 'MATTECREATOR_HYPERPARAM_outputDirectory', text='Output Directory')

		# Output Format
		row = layout.row()
		row.prop(context.scene, 'MATTECREATOR_HYPERPARAM_outputFormat', text='Format')		

		row = layout.row() # Additional Spacing
		row = layout.row() # Additional Spacing

		row = layout.row()
		row.operator(MATTECREATOR_OT_extractMatte.bl_idname, text='Extract Matte', icon='COMMUNITY')

		

class MATTECREATOR_PT_panelInitialSetup(bpy.types.Panel):
	bl_label = 'Initial Setup'
	bl_idname = 'MATTECREATOR_PT_panelInitialSetup'
	bl_space_type = 'NODE_EDITOR'
	bl_region_type = 'UI'
	bl_category = 'MatteCreator'
	bl_parent_id = 'MATTECREATOR_PT_panelMain'

	@classmethod
	def poll(cls, context):
		snode = context.space_data
		return snode.tree_type == 'CompositorNodeTree'

	def draw(self, context):	
		layout = self.layout		 
		row = layout.row()
		row.operator(MATTECREATOR_OT_installPackages.bl_idname, text='Install Packages', icon_value=727)

class MATTECREATOR_PT_panelAdvanced(bpy.types.Panel):
	bl_label = 'Advanced'
	bl_idname = 'MATTECREATOR_PT_panelAdvanced'
	bl_space_type = 'NODE_EDITOR'
	bl_region_type = 'UI'
	bl_category = 'MatteCreator'
	bl_parent_id = 'MATTECREATOR_PT_panelMain'
	bl_options = {'DEFAULT_CLOSED'}

	@classmethod
	def poll(cls, context):
		snode = context.space_data
		return snode.tree_type == 'CompositorNodeTree'

	def draw(self, context):
		layout = self.layout

		# Model Type
		row = layout.row()
		row.prop(context.scene, 'MATTECREATOR_HYPERPARAM_modelType', text='Model')

		# Model Backbone
		row = layout.row()
		row.prop(context.scene, 'MATTECREATOR_HYPERPARAM_modelBackbone', text='Backbone')

		# Model Backbone Scale
		row = layout.row()
		row.label(text='Scale: ')
		row.prop(context.scene, 'MATTECREATOR_HYPERPARAM_modelBackboneScale', text='')	

		# Refine Mode
		row = layout.row()
		row.prop(context.scene, 'MATTECREATOR_HYPERPARAM_modelRefineMode', text='Refine Mode')

		# Refine Pixels
		row = layout.row()
		row.prop(context.scene, 'MATTECREATOR_HYPERPARAM_refineSamplePixels', text='Refine Pixels')

		# Refine Threshold
		row = layout.row()
		row.prop(context.scene, 'MATTECREATOR_HYPERPARAM_refineThreshold', text='Refine Threshold')

		# Refine Kernel Size
		row = layout.row()
		row.prop(context.scene, 'MATTECREATOR_HYPERPARAM_refineKernelSize', text='Kernel Size')

		# Preprocess Alignment
		row = layout.row()
		row.prop(context.scene, 'MATTECREATOR_HYPERPARAM_preprocessAlignment', text='Preprocess Alignment')

		# Video Target BGR
		row = layout.row()
		row.prop(context.scene, 'MATTECREATOR_HYPERPARAM_videoTargetBGR', text='Video Target BGR')

		# Output Types
		row = layout.row()
		row.label(text='Output Layers:')
		row = layout.row()
		row.prop(context.scene, 'MATTECREATOR_HYPERPARAM_outputCom', text='Composite')
		row = layout.row()
		row.prop(context.scene, 'MATTECREATOR_HYPERPARAM_outputPha', text='Alpha Matte')
		row = layout.row()
		row.prop(context.scene, 'MATTECREATOR_HYPERPARAM_outputFgr', text='Foreground')
		row = layout.row()
		row.prop(context.scene, 'MATTECREATOR_HYPERPARAM_outputErr', text='Error')
		row = layout.row()
		row.prop(context.scene, 'MATTECREATOR_HYPERPARAM_outputRef', text='Reference')
	


		

#--------------------------------------------------------------
# Register 
#--------------------------------------------------------------

classes_interface = (MATTECREATOR_PT_panelMain, MATTECREATOR_PT_panelAdvanced)
classes_functionality = (MATTECREATOR_OT_extractMatte, MATTECREATOR_OT_installPackages, MATTECREATOR_OT_setModelCheckpointPath)

def register():

	# Register Classes
	for c in classes_interface:
		bpy.utils.register_class(c)
	for c in classes_functionality:
		bpy.utils.register_class(c)

	# Register Panels Based on installed Dependencies

	if MATTECREATOR_MISSING_DEPENDENCIES:
		bpy.utils.register_class(MATTECREATOR_PT_panelInitialSetup)

	# File Variables

	bpy.types.Scene.MATTECREATOR_VAR_videoSource = bpy.props.PointerProperty(name='', type=bpy.types.Image, description='Select a Source Video')
	bpy.types.Scene.MATTECREATOR_VAR_cleanPlate = bpy.props.PointerProperty(name='', type=bpy.types.Image, description='Select a Clean Plate')
	bpy.types.Scene.MATTECREATOR_VAR_modelPath = bpy.props.StringProperty(name='', default='')
	bpy.types.Scene.MATTECREATOR_VAR_outputDir = bpy.props.StringProperty(name='', default='')
		
	# Hyperparameters
	bpy.types.Scene.MATTECREATOR_HYPERPARAM_modelType = bpy.props.EnumProperty(name='MATTECREATOR_HYPERPARAM_modelType', items=[('mattingbase', 'mattingbase', ''), ('mattingrefine', 'mattingrefine', '')])
	bpy.types.Scene.MATTECREATOR_HYPERPARAM_modelBackbone = bpy.props.EnumProperty(name='MATTECREATOR_HYPERPARAM_modelBackbone', items=[('resnet101', 'resnet101', ''), ('resnet50', 'resnet50', ''), ('mobilenetv2', 'mobilenetv2', '')]) 
	bpy.types.Scene.MATTECREATOR_HYPERPARAM_modelBackboneScale = bpy.props.FloatProperty(name='MATTECREATOR_HYPERPARAM_modelBackboneScale', soft_min=0.1, soft_max=1.0, default=0.25)
	bpy.types.Scene.MATTECREATOR_HYPERPARAM_modelCheckpoint = bpy.props.StringProperty(name='MATTECREATOR_HYPERPARAM_modelCheckpoint') # Replace with file selector with filter glob, need to make persistent too
	bpy.types.Scene.MATTECREATOR_HYPERPARAM_modelRefineMode = bpy.props.EnumProperty(name='MATTECREATOR_HYPERPARAM_modelRefineMode', items=[('full', 'full', ''), ('sampling', 'sampling', ''), ('thresholding', 'thresholding', '')])
	bpy.types.Scene.MATTECREATOR_HYPERPARAM_refineSamplePixels = bpy.props.IntProperty(name='MATTECREATOR_HYPERPARAM_refineSamplePixels', soft_min=10000, soft_max=320000, default=80000)	
	bpy.types.Scene.MATTECREATOR_HYPERPARAM_refineThreshold = bpy.props.FloatProperty(name='MATTECREATOR_HYPERPARAM_refineThreshold', soft_min=0.1, soft_max=0.9, default=0.7)
	bpy.types.Scene.MATTECREATOR_HYPERPARAM_refineKernelSize = bpy.props.IntProperty(name='MATTECREATOR_HYPERPARAM_refineKernelSize', soft_min=2, soft_max=5, default=3)
	
	bpy.types.Scene.MATTECREATOR_HYPERPARAM_videoBGR = bpy.props.StringProperty(name='MATTECREATOR_HYPERPARAM_videoBGR') # Replace with file picker
	bpy.types.Scene.MATTECREATOR_HYPERPARAM_videoTargetBGR = bpy.props.StringProperty(name='MATTECREATOR_HYPERPARAM_videoTargetBGR') # Replace with file picker

	bpy.types.Scene.MATTECREATOR_HYPERPARAM_device = bpy.props.EnumProperty(name='MATTECREATOR_HYPERPARAM_device', items=[('cuda', 'cuda (GPU)', ''), ('cpu', 'cpu', '')], default='cuda')
	bpy.types.Scene.MATTECREATOR_HYPERPARAM_preprocessAlignment = bpy.props.BoolProperty(name='MATTECREATOR_HYPERPARAM_preprocessAlignment', default=False)

	bpy.types.Scene.MATTECREATOR_HYPERPARAM_outputDirectory = bpy.props.StringProperty(name='MATTECREATOR_HYPERPARAM_outputDirectory')

	bpy.types.Scene.MATTECREATOR_HYPERPARAM_outputCom = bpy.props.BoolProperty(name='MATTECREATOR_HYPERPARAM_outputCom', default=True)
	bpy.types.Scene.MATTECREATOR_HYPERPARAM_outputPha = bpy.props.BoolProperty(name='MATTECREATOR_HYPERPARAM_outputPha', default=False)
	bpy.types.Scene.MATTECREATOR_HYPERPARAM_outputFgr = bpy.props.BoolProperty(name='MATTECREATOR_HYPERPARAM_outputFgr', default=False)
	bpy.types.Scene.MATTECREATOR_HYPERPARAM_outputErr = bpy.props.BoolProperty(name='MATTECREATOR_HYPERPARAM_outputErr', default=False)
	bpy.types.Scene.MATTECREATOR_HYPERPARAM_outputRef = bpy.props.BoolProperty(name='MATTECREATOR_HYPERPARAM_outputRef', default=False)

	bpy.types.Scene.MATTECREATOR_HYPERPARAM_outputFormat = bpy.props.EnumProperty(name='MATTECREATOR_HYPERPARAM_outputFormat', items=[('video', 'Video', ''), ('image_sequences', 'Image Sequence', '')], default='video')

	# Possibly Temp
	bpy.types.Scene.MATTECREATOR_HYPERPARAM_resizeX = bpy.props.IntProperty(name='MATTECREATOR_HYPERPARAM_resizeX')
	bpy.types.Scene.MATTECREATOR_HYPERPARAM_resizeY = bpy.props.IntProperty(name='MATTECREATOR_HYPERPARAM_resizeY')	


def unregister():

	# Unregister
	for c in reversed(classes_interface):
		bpy.utils.unregister_class(c)
	for c in reversed(classes_functionality):
		bpy.utils.unregister_class(c)

	# Unregister Panels based on Installed Dependencies

	if MATTECREATOR_MISSING_DEPENDENCIES:
		bpy.utils.unregister_class(MATTECREATOR_PT_panelInitialSetup)

	# File Variables
	del bpy.types.Scene.MATTECREATOR_VAR_videoSource
	del bpy.types.Scene.MATTECREATOR_VAR_cleanPlate
	del bpy.types.Scene.MATTECREATOR_VAR_outputDir
	del bpy.types.Scene.MATTECREATOR_VAR_modelPath

	# Hyperparamaters
	del bpy.types.Scene.MATTECREATOR_HYPERPARAM_modelType
	del bpy.types.Scene.MATTECREATOR_HYPERPARAM_modelBackbone
	del bpy.types.Scene.MATTECREATOR_HYPERPARAM_modelBackboneScale
	del bpy.types.Scene.MATTECREATOR_HYPERPARAM_modelCheckpoint

	del bpy.types.Scene.MATTECREATOR_HYPERPARAM_modelRefineMode
	del bpy.types.Scene.MATTECREATOR_HYPERPARAM_refineSamplePixels
	del bpy.types.Scene.MATTECREATOR_HYPERPARAM_refineThreshold
	del bpy.types.Scene.MATTECREATOR_HYPERPARAM_refineKernelSize

	del bpy.types.Scene.MATTECREATOR_HYPERPARAM_videoSource
	del bpy.types.Scene.MATTECREATOR_HYPERPARAM_videoBGR
	del bpy.types.Scene.MATTECREATOR_HYPERPARAM_videoTargetBGR

	del bpy.types.Scene.MATTECREATOR_HYPERPARAM_device
	del bpy.types.Scene.MATTECREATOR_HYPERPARAM_preprocessAlignment

	del bpy.types.Scene.MATTECREATOR_HYPERPARAM_outputDirectory 

	del bpy.types.Scene.MATTECREATOR_HYPERPARAM_outputCom
	del bpy.types.Scene.MATTECREATOR_HYPERPARAM_outputPha
	del bpy.types.Scene.MATTECREATOR_HYPERPARAM_outputFgr
	del bpy.types.Scene.MATTECREATOR_HYPERPARAM_outputErr
	del bpy.types.Scene.MATTECREATOR_HYPERPARAM_outputRef

	del bpy.types.Scene.MATTECREATOR_HYPERPARAM_outputFormat

	del bpy.types.Scene.MATTECREATOR_HYPERPARAM_resizeX
	del bpy.types.Scene.MATTECREATOR_HYPERPARAM_resizeY

if __name__ == '__main__':
	register()