import os

#region General
CURRENT_SLIDE = 'current\slide-2022-09-12T15-38-25-R1-S2.mrxs'
CURRENT_SLIDE_XML = 'current\slide-2022-09-12T15-38-25-R1-S2\Data0021.dat'
OPENSLIDE_PATH = os.path.abspath('openslide\\bin') # 'E:\\Github\\DemetraAI\\openslide\\bin'
HDD_SLIDES = 'R:\\CYTOLOGY'
TEMP_FOLDER = 'temp'
#endregion

#region Neural Network
DATASET_FOLDER = 'dataset'
MASKS_FOLDER = 'masks' # Inside DATASET_FOLDER
IMAGES_FOLDER = 'rois' # Inside DATASET_FOLDER
IMAGE_SHAPE = (128, 128)
CLASSES = 3
LABELS = {
    'LSIL': 2,
    'HSIL': 2,
    'HSUL': 2,
    'Group HSIL': 2,
    'ASCH': 2,
    'Group atypical': 2,
    'ASCUS': 2,
    'Atypical': 2,
    'Atipical': 2,
    'Atypical naked': 2,
} 
#endregion

#region GUI
SLIDE_DIR = './current' # Only for GUI
UNET_PRED_MODE = 'smooth'  # 'direct'
#endregion

#region Dataset
EXCLUDE_DUPLICATES = False
BROADEN_INDIVIDUAL_RECT = 1000
#endregion

#region Packaging 
# Must be commented when Python scripts are used!
def user_changes_on_import():
    pass

user_changes_on_import()
#endregion