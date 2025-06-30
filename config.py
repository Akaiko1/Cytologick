import configparser
import os

#region General
# CURRENT_SLIDE = 'current\slide-2022-11-11T11-10-38-R1-S18.mrxs'
CURRENT_SLIDE = os.path.abspath('slide-2022-09-12T15-38-25-R1-S2.mrxs') # 'current\slide-2022-09-12T15-38-25-R1-S2.mrxs'
CURRENT_SLIDE_XML = 'current\slide-2022-09-12T15-38-25-R1-S2\Data0021.dat'
OPENSLIDE_PATH = os.path.abspath('openslide\\bin') # 'E:\\Github\\DemetraAI\\openslide\\bin'
HDD_SLIDES = os.path.abspath('current')
HDD_SLIDES_SVS = 'E:\\CYTOLOGY_2'
TEMP_FOLDER = 'temp'
#endregion

#region Neural Network
DATASET_FOLDER = 'dataset'
MASKS_FOLDER = 'masks' # Inside DATASET_FOLDER
IMAGES_FOLDER = 'rois' # Inside DATASET_FOLDER
IMAGE_CHUNK = (256, 256)
IMAGE_SHAPE = (128, 128)
CLASSES = 3
LABELS = {
    'LSIL': 2,
    'HSIL': 2,
    'Group HSIL': 2,
    'ASCH': 2,
    'Group atypical': 2,
    'ASCUS': 2,
    'Atypical': 2,
    'Atypical naked': 2,
} 
#endregion

#region GUI
SLIDE_DIR = './current' # Only for GUI
UNET_PRED_MODE = 'remote' # 'smooth', 'direct'
#endregion

#region Dataset
EXCLUDE_DUPLICATES = False
BROADEN_INDIVIDUAL_RECT = 1000
#endregion

#region Web
IP_EXPOSED = '127.0.0.1'
#endregion

# Define a function to load settings from .ini file
def load_settings_from_ini():
    config = configparser.ConfigParser()
    ini_file = 'config.ini'  # Name of your .ini file
    if os.path.exists(ini_file):
        print('config.ini detected')
        config.read(ini_file)
        
        # Update global variables based on .ini file
        global CURRENT_SLIDE, CURRENT_SLIDE_XML, OPENSLIDE_PATH, HDD_SLIDES, HDD_SLIDES_SVS, TEMP_FOLDER
        global DATASET_FOLDER, MASKS_FOLDER, IMAGES_FOLDER, IMAGE_CHUNK, IMAGE_SHAPE, CLASSES, LABELS
        global SLIDE_DIR, UNET_PRED_MODE, EXCLUDE_DUPLICATES, BROADEN_INDIVIDUAL_RECT, IP_EXPOSED

        # General Section
        if 'General' in config:
            if 'CURRENT_SLIDE' in config['General']:
                CURRENT_SLIDE = os.path.abspath(config['General']['CURRENT_SLIDE'])
            if 'CURRENT_SLIDE_XML' in config['General']:
                CURRENT_SLIDE_XML = config['General']['CURRENT_SLIDE_XML']
            if 'OPENSLIDE_PATH' in config['General']:
                OPENSLIDE_PATH = os.path.abspath(config['General']['OPENSLIDE_PATH'])
            if 'HDD_SLIDES' in config['General']:
                HDD_SLIDES = os.path.abspath(config['General']['HDD_SLIDES'])
            if 'HDD_SLIDES_SVS' in config['General']:
                HDD_SLIDES_SVS = config['General']['HDD_SLIDES_SVS']
            if 'TEMP_FOLDER' in config['General']:
                TEMP_FOLDER = config['General']['TEMP_FOLDER']

        # Neural Network Section
        if 'Neural Network' in config:
            if 'DATASET_FOLDER' in config['Neural Network']:
                DATASET_FOLDER = config['Neural Network']['DATASET_FOLDER']
            if 'MASKS_FOLDER' in config['Neural Network']:
                MASKS_FOLDER = config['Neural Network']['MASKS_FOLDER']
            if 'IMAGES_FOLDER' in config['Neural Network']:
                IMAGES_FOLDER = config['Neural Network']['IMAGES_FOLDER']
            if 'IMAGE_CHUNK' in config['Neural Network']:
                IMAGE_CHUNK = tuple(map(int, config['Neural Network']['IMAGE_CHUNK'].split(',')))
            if 'IMAGE_SHAPE' in config['Neural Network']:
                IMAGE_SHAPE = tuple(map(int, config['Neural Network']['IMAGE_SHAPE'].split(',')))
            if 'CLASSES' in config['Neural Network']:
                CLASSES = int(config['Neural Network']['CLASSES'])
            if 'LABELS' in config['Neural Network']:
                LABELS = eval(config['Neural Network']['LABELS'])

        # GUI Section
        if 'GUI' in config:
            if 'SLIDE_DIR' in config['GUI']:
                SLIDE_DIR = config['GUI']['SLIDE_DIR']
            if 'UNET_PRED_MODE' in config['GUI']:
                UNET_PRED_MODE = config['GUI']['UNET_PRED_MODE']

        # Dataset Section
        if 'Dataset' in config:
            if 'EXCLUDE_DUPLICATES' in config['Dataset']:
                EXCLUDE_DUPLICATES = config.getboolean('Dataset', 'EXCLUDE_DUPLICATES')
            if 'BROADEN_INDIVIDUAL_RECT' in config['Dataset']:
                BROADEN_INDIVIDUAL_RECT = int(config['Dataset']['BROADEN_INDIVIDUAL_RECT'])

        # Web Section
        if 'Web' in config:
            if 'IP_EXPOSED' in config['Web']:
                IP_EXPOSED = config['Web']['IP_EXPOSED']

# Call the function to load settings
load_settings_from_ini()
