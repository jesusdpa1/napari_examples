import sys
import os
import datetime
import glob

from PyQt5 import QtWidgets, QtCore

import napari
from PIL import Image
from PIL.TiffTags import TAGS

import numpy as np
import tensorflow as tf
import dask.array as da
from tqdm import tqdm

import yaml
import json
import h5py

from flowdec import psf as fd_psf # Deconvolution

##
import utils
import processing

# Create a dictionary to store global variables
experiment_global = {'path_to_file': [], 'experiment_json': [], 'experiment_check': [], 'multi_psf': []}

# Sets the color of the progress bar to green, it can be change to whatever color
PROGRESSBAR_COMPLETED_STYLE = """

QProgressBar::chunk {
    background-color: green;
}
"""
# LOAD AND PRINT METADATA
class metadata(QtWidgets.QWidget):
    def __init__(self):
        super(metadata, self).__init__()
        self.layout = QtWidgets.QGridLayout()
        # Add toolbar and items
        self.toolbox = QtWidgets.QToolBox()
        self.boxview = QtWidgets.QGroupBox()
        self.metalbl = QtWidgets.QLabel('Experiment Metadata')
        self.txtlist = QtWidgets.QTextEdit()
        self.txtlist.setReadOnly(True)
        with open(experiment_global['path_to_file']) as f:
            metadata_dic = json.load(f)
            experiment_global['experiment_json'] = metadata_dic
        final_text = json.dumps(metadata_dic, indent=3)
        final_text = final_text.replace('{', '')
        final_text = final_text.replace('}', '')
        self.txtlist.setText(final_text)
        self.layout.addWidget(self.metalbl, 0, 0)
        self.layout.addWidget(self.txtlist, 1, 0)
        self.setLayout(self.layout)
        self.toolbox.addItem(self.boxview, 'Metadata')


class check_experiment(QtWidgets.QWidget):
    def __init__(self):
        super(check_experiment, self).__init__()
        self.layout = QtWidgets.QGridLayout()
        # Add toolbar and items
        self.toolbox = QtWidgets.QToolBox()
        self.boxview = QtWidgets.QGroupBox()
        self.filechecklbl = QtWidgets.QLabel('File Check')
        self.txtlist = QtWidgets.QTextEdit()
        self.txtlist.setReadOnly(True)
        self.btn_update = QtWidgets.QPushButton('Update')

        self.check_num_files()

        final_text = json.dumps(experiment_global['experiment_check'], indent=3)
        final_text = final_text.replace('{', '')
        final_text = final_text.replace('}', '')
        self.txtlist.setText(final_text)
        self.layout.addWidget(self.filechecklbl, 0, 0)
        self.layout.addWidget(self.txtlist, 1, 0)
        self.layout.addWidget(self.btn_update, 2, 0)
        self.btn_update.clicked.connect(self.btnstate_update)
        self.setLayout(self.layout)
        self.toolbox.addItem(self.boxview, 'Experiment Check')

    def check_num_files(self):
        # Create FILE CHECK JSON
        metadata_dic = experiment_global['experiment_json']
        filecheck_json = {'experiment_name': metadata_dic['name'],
                         'number_of_cycles': metadata_dic['numCycles'],
                         'number_of_channels': metadata_dic['numChannels'],
                         'number_of_Zplanes': metadata_dic['numZPlanes'],
                         'number_of_tiles': metadata_dic['numTiles'],
                         'region_Width': metadata_dic['regionWidth'],
                         'region_Height': metadata_dic['regionHeight'],
                         'number_tileWidth': metadata_dic['tileWidth'],
                         'number_tileHeight': metadata_dic['tileHeight'],
                         'number_cycle_folders': {},
                         'files_per_cycle': {}
                         }
        # Check number of folders
        path_base = os.path.dirname(experiment_global['path_to_file'])
        list_path_cyc = glob.glob(os.path.join(path_base, 'cyc*'))
        filecheck_json['number_cycle_folders'].update({'theoretical': metadata_dic['numCycles'],
                                                       'real': len(list_path_cyc)})
        # Check number of files
        theoretical_numfiles = metadata_dic['numChannels'] * metadata_dic['numZPlanes'] * metadata_dic['numTiles']
        for path_to_cyc in list_path_cyc:
            list_path_img = glob.glob(os.path.join(path_to_cyc, '*CH*'))
            dic_numfiles = {'theoretical': theoretical_numfiles, 'real': len(list_path_img),
                            'pass': theoretical_numfiles == len(list_path_img)}
            filecheck_json['files_per_cycle'].update({os.path.basename(path_to_cyc): dic_numfiles})

        experiment_global['experiment_check'] = filecheck_json

    @QtCore.pyqtSlot()
    def btnstate_update(self):
        self.check_num_files()

        final_text = json.dumps(experiment_global['experiment_check'], indent=3)
        final_text = final_text.replace('{', '')
        final_text = final_text.replace('}', '')
        self.txtlist.setText(final_text)

class gui_control(QtWidgets.QWidget):
    def __init__(self):
        super(gui_control, self).__init__()
        self.boxview = QtWidgets.QGroupBox()
        self.layout = QtWidgets.QGridLayout()
        self.btn_psf = QtWidgets.QPushButton('Create PSF')
        self.btn_deconvolve = QtWidgets.QPushButton('Deconvolution')
        self.btn_bestplane = QtWidgets.QPushButton('Best Plane')
        self.layout.addWidget(self.btn_psf, 0, 0)
        self.layout.addWidget(self.btn_deconvolve, 1, 0)
        self.layout.addWidget(self.btn_bestplane, 2, 0)
        self.setLayout(self.layout)
        self.btn_psf.clicked.connect(self.btnstate_psf)
        self.btn_deconvolve.clicked.connect(self.btnstate_deconvolve)
        self.btn_bestplane.clicked.connect(self.btnstate_bestplane)

        self.bar = QtWidgets.QProgressBar(self)
        self.bar.setStyleSheet(PROGRESSBAR_COMPLETED_STYLE)
        self.bar.setGeometry(200, 150, 200, 30)
        viewer.window.add_dock_widget(self.bar, area='bottom', name='progress_bar')

    def create_psf(self):
        experiment_metadata = experiment_global['experiment_json']
        args = dict(
            # Set psf dimensions to match volumes
            size_x=int(experiment_metadata['tileWidth']),
            size_y=int(experiment_metadata['tileHeight']),
            size_z=int(experiment_metadata['numZPlanes']),

            # Magnification factor
            m=float(experiment_metadata['magnification']),

            # Numerical aperture
            na=float(experiment_metadata['aperture']),

            # Axial resolution in microns (nm in akoya config)
            res_axial=float(experiment_metadata['zPitch']) / 1000.,

            # Lateral resolution in microns (nm in akoya config)
            res_lateral=float(experiment_metadata['xyResolution']) / 1000.,

            # Immersion refractive index
            ni0=1.0,

            # Set "particle position" in Gibson-Lannie to 0 which gives a
            # Born & Wolf kernel as a degenerate case
            pz=0.
        )

        multi_psf = {}
        psf_struct = {'wavelength': [], 'psf': []}
        ch_names = list(experiment_metadata['wavelengths'])

        for n in range(len(ch_names)):
            wavelength = float(experiment_metadata['wavelengths'][n])

            psf_single = fd_psf.GibsonLanni(**{**args, **{'wavelength': wavelength / 1000.}}).generate()
            psf_struct = {'wavelength': wavelength, 'psf': psf_single}
            multi_psf.update({''.join(['CH', str(n+1)]): psf_struct})
            psf_struct = {'wavelength': [], 'psf': []}
            p_value = ((n + 1) * 100)/ len(ch_names)
            self.bar.setValue(p_value)

        self.bar.setValue(p_value)
        experiment_global['multi_psf'] = multi_psf

    @QtCore.pyqtSlot()
    def btnstate_psf(self):
        print('Creating PSF')
        # PROGRESS BAR
        p_value = 0
        self.bar.setValue(p_value)
        self.create_psf()

    def btnstate_deconvolve(self):
        print('Processing')
        p_value = 0
        self.bar.setValue(p_value)
        path_to_folder = os.path.dirname(experiment_global['path_to_file'])
        metadata = experiment_global['experiment_json']
        numCycles = metadata['numCycles']

        # Get all the channels in one directory

        # Channel directory
        ch_names = list(metadata['channel_names'])
        ch_dic = {}
        ZPlane = metadata['numZPlanes']
        iterations = 30

        for cycle_n in range(numCycles):
            cycle = ''.join(['cyc', str(cycle_n + 1).zfill(3)])
            for n in range(len(ch_names)):
                path_to_ch = os.path.join(path_to_folder, ''.join([cycle, '*']), ''.join(['*', ch_names[n], '.tif']))
                path_to_chlist = glob.glob(path_to_ch)
                lazy_arrays = utils.load_img(path_to_chlist)
                psf_single = experiment_global['multi_psf'][ch_names[n]]['psf']
                path_to_save = os.path.join(path_to_folder, datetime.date.today().strftime("processed_%m_%d_%Y"),
                                            'deconvolution')
                struct_name = ''.join(['deconvolution_', cycle, '_', ch_names[n], '.h5'])
                processing.img_deconvolution(lazy_arrays, psf_single, ZPlane, iterations, path_to_save, struct_name) #
                p_value = ((n + 1) * 100)/len(ch_names)
                self.bar.setValue(p_value)
#------------------------------------------ BEST PLANE SELECTION-------------------------------------------------------
    # def btnstate_bestplane(self):
    #     print('best plane')
    #     p_value = 0
    #     self.bar.setValue(p_value)
    #     path_to_weights = # assign path
    #     bestplane_model = utils.bestplane_model(path_to_weights)
    #     path_to_deconvolve = glob.glob(os.path.join(experiment_global['path_to_file'], 'processed', 'deconvolution', '*.h5'))
    #     path_to_save = os.path.join(experiment_global['path_to_file'], 'processed', 'bestplane')
    #     for n in range(len(path_to_deconvolve)):
    #         f = h5py.File(path_to_deconvolve[n])
    #         d = f['/data']
    #         metadata = experiment_global['experiment_json']
    #         lazy_arrays = da.from_array(d, chunks=(1000, 1000))
    #         ZPlane = metadata['numZPlanes']
    #         struct_name = os.path.basename(path_to_deconvolve[n]).replace('deconvolution', 'bestplane')
    #         processing.select_bestplane(bestplane_model, lazy_arrays, ZPlane, path_to_save, struct_name)
    #         p_value = ((n + 1) * 100) / len(path_to_deconvolve)
    #         self.bar.setValue(p_value)
    #     # file_order = utils.get_fileorder(metadata)
    #
    #     # img_big = processing.img_reconstruction(da.stack(img_append), metadata, file_order)



class gui_tab(QtWidgets.QWidget):
    def __init__(self):
        super(gui_tab, self).__init__()
        self.title = 'Gui Control'
        self.layout = QtWidgets.QGridLayout()
        self.tabview = QtWidgets.QTabWidget()
        self.tabview.addTab(gui_control(), 'Functions')
        self.tabview.addTab(metadata(), 'Metadata')
        self.tabview.addTab(check_experiment(), 'Experiment_check')
        self.layout.addWidget(self.tabview, 0, 0)
        self.setLayout(self.layout)

class file_explorer(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'File Explorer'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.openFileNameDialog()
        self.show()

    def openFileNameDialog(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        path_to_file, _ = QtWidgets.QFileDialog.getOpenFileName(self, self.title, "",
                                                  "All Files (*)", options=options)
        if path_to_file:
            experiment_global['path_to_file'] = path_to_file
            viewer.window.add_dock_widget(gui_tab(), area='right', name='gui_control')
            path_to_folder = os.path.dirname(experiment_global['path_to_file'])
            utils.create_folderstruct(path_to_folder)
            # viewer.window.add_dock_widget(check_experiment(), area='right', name='file_check')

class load_experiment(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.boxview = QtWidgets.QGroupBox()
        self.layout = QtWidgets.QGridLayout()
        self.btn_load = QtWidgets.QPushButton('Load Experiment')
        # self.btn_process = QtWidgets.QPushButton('Process')
        self.layout.addWidget(self.btn_load, 0, 0)
        # self.layout.addWidget(self.btn_process, 0, 1)
        self.setLayout(self.layout)
        self.btn_load.clicked.connect(self.btnstate_load)
        # self.btn_process.clicked.connect(self.btnstate_process)

    @QtCore.pyqtSlot()
    def btnstate_load(self):
        file_explorer()
#%%
logo = np.array(Image.open('./resources/logo.png')) # Add a random image to force start of napari. I think they fix this issue
with napari.gui_qt():
    viewer = napari.view_image(logo)
    viewer.window.add_dock_widget(load_experiment(), area='left', name='load images')
