
#region General
CURRENT_SLIDE = 'current\slide-2022-09-12T15-38-25-R1-S2.mrxs'
CURRENT_SLIDE_XML = 'current\slide-2022-09-12T15-38-25-R1-S2\Data0021.dat'
OPENSLIDE_PATH = 'E:\\Github\\DemetraAI\\openslide\\bin'
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
    'HSIL': 2
}
#endregion

#region GUI
SLIDE_DIR = './current' # Only for GUI
#endregion
