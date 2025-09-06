import configparser
import os
import yaml
import multiprocessing

#region General
# CURRENT_SLIDE = 'current\slide-2022-11-11T11-10-38-R1-S18.mrxs'
CURRENT_SLIDE = os.path.abspath('slide-2022-09-12T15-38-25-R1-S2.mrxs') # 'current\slide-2022-09-12T15-38-25-R1-S2.mrxs'
CURRENT_SLIDE_XML = 'current\\slide-2022-09-12T15-38-25-R1-S2\\Data0021.dat'
OPENSLIDE_PATH = os.path.abspath('openslide\\bin') # 'E:\\Github\\DemetraAI\\openslide\\bin'
HDD_SLIDES = os.path.abspath('current')
HDD_SLIDES_SVS = 'E:\\CYTOLOGY_2'
TEMP_FOLDER = 'temp'
#endregion

#region Neural Network
FRAMEWORK = 'tensorflow'  # 'tensorflow' or 'pytorch'
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
# Theme options:
# - 'auto' (platform QSS)
# - 'qt' (qt_material theme)
# - 'windows' / 'mac' (platform QSS)
# - 'qdarkstyle' (external package, if installed)
# - 'qss' (load a custom QSS from path)
GUI_THEME = 'auto'
GUI_MATERIAL_THEME = 'dark_teal.xml'
GUI_CUSTOM_QSS = ''
#endregion

#region Dataset
EXCLUDE_DUPLICATES = False
BROADEN_INDIVIDUAL_RECT = 1000
#endregion

#region Web
IP_EXPOSED = '127.0.0.1'
# Optional: TensorFlow Serving endpoint for cloud inference/health checks
ENDPOINT_URL = 'http://51.250.28.160:7500'
HEALTH_TIMEOUT = 1.5
#endregion

def load_config():
    config_files = ['config.yaml', 'config.yml', 'config.ini']
    
    for config_file in config_files:
        if os.path.exists(config_file):
            # Only print from the main process to avoid DataLoader worker spam
            if multiprocessing.current_process().name == 'MainProcess':
                print(f'{config_file} detected')
            
            if config_file.endswith(('.yaml', '.yml')):
                _load_yaml_config(config_file)
            elif config_file.endswith('.ini'):
                _load_ini_config(config_file)
            return

def _load_yaml_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    _update_globals_from_dict(config)

def _load_ini_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    
    config_dict = {
        'general': dict(config['General']) if 'General' in config else {},
        'neural_network': dict(config['Neural Network']) if 'Neural Network' in config else {},
        'gui': dict(config['GUI']) if 'GUI' in config else {},
        'dataset': dict(config['Dataset']) if 'Dataset' in config else {},
        'web': dict(config['Web']) if 'Web' in config else {}
    }
    
    _update_globals_from_dict(config_dict)

def _update_globals_from_dict(config):
    global CURRENT_SLIDE, CURRENT_SLIDE_XML, OPENSLIDE_PATH, HDD_SLIDES, HDD_SLIDES_SVS, TEMP_FOLDER
    global FRAMEWORK, DATASET_FOLDER, MASKS_FOLDER, IMAGES_FOLDER, IMAGE_CHUNK, IMAGE_SHAPE, CLASSES, LABELS
    global SLIDE_DIR, UNET_PRED_MODE, GUI_THEME, EXCLUDE_DUPLICATES, BROADEN_INDIVIDUAL_RECT, IP_EXPOSED, ENDPOINT_URL, HEALTH_TIMEOUT
    
    general = config.get('general', {})
    if 'current_slide' in general:
        CURRENT_SLIDE = os.path.abspath(general['current_slide'])
    if 'current_slide_xml' in general:
        CURRENT_SLIDE_XML = general['current_slide_xml']
    if 'openslide_path' in general:
        OPENSLIDE_PATH = os.path.abspath(general['openslide_path'])
    if 'hdd_slides' in general:
        HDD_SLIDES = os.path.abspath(general['hdd_slides'])
    if 'hdd_slides_svs' in general:
        HDD_SLIDES_SVS = general['hdd_slides_svs']
    if 'temp_folder' in general:
        TEMP_FOLDER = general['temp_folder']
    
    neural_network = config.get('neural_network', {})
    if 'framework' in neural_network:
        FRAMEWORK = neural_network['framework']
    if 'dataset_folder' in neural_network:
        DATASET_FOLDER = neural_network['dataset_folder']
    if 'masks_folder' in neural_network:
        MASKS_FOLDER = neural_network['masks_folder']
    if 'images_folder' in neural_network:
        IMAGES_FOLDER = neural_network['images_folder']
    if 'image_chunk' in neural_network:
        IMAGE_CHUNK = tuple(neural_network['image_chunk'])
    if 'image_shape' in neural_network:
        IMAGE_SHAPE = tuple(neural_network['image_shape'])
    if 'classes' in neural_network:
        CLASSES = neural_network['classes']
    if 'labels' in neural_network:
        LABELS = neural_network['labels']
    
    gui = config.get('gui', {})
    if 'slide_dir' in gui:
        SLIDE_DIR = gui['slide_dir']
    if 'unet_pred_mode' in gui:
        UNET_PRED_MODE = gui['unet_pred_mode']
    if 'theme' in gui:
        GUI_THEME = gui['theme']
    if 'material_theme' in gui:
        GUI_MATERIAL_THEME = gui['material_theme']
    if 'custom_qss' in gui:
        GUI_CUSTOM_QSS = gui['custom_qss']
    
    dataset = config.get('dataset', {})
    if 'exclude_duplicates' in dataset:
        EXCLUDE_DUPLICATES = dataset['exclude_duplicates']
    if 'broaden_individual_rect' in dataset:
        BROADEN_INDIVIDUAL_RECT = dataset['broaden_individual_rect']
    
    web = config.get('web', {})
    if 'ip_exposed' in web:
        IP_EXPOSED = web['ip_exposed']
    if 'endpoint_url' in web:
        ENDPOINT_URL = web['endpoint_url']
    if 'health_timeout' in web:
        HEALTH_TIMEOUT = float(web['health_timeout'])

# Load configuration
load_config()
