# Cytologick software (WIP)

## Installation instructions

### Windows 10

1. Install **Conda** and create a new environment with **python3**, version **3.10.6** or later

2. Install Openslide from <https://openslide.org> refer to the manual corresponding to your current OS

3. Activate new env with **cmd** or **powershell** in local folder

4. Run command:
   
   ```bash
   pip install -r requirements.txt
   ```

5. Also run in your activated environment:
   
   ```bash
   conda install tensorflow
   conda install openslide-python # optional if run.py doesn't work
   ```

6. Modify settings at config.py (debug config file with auxillary parameters, a subject to change later)
   
   **OPENSLIDE_PATH** is necessary! This is an absolute path to your OpenSlide installation folder
   
   ```python
   #region General
   # CURRENT_SLIDE = 'current\slide-2022-11-11T11-10-38-R1-S18.mrxs'
   CURRENT_SLIDE = os.path.abspath('slide-2022-09-12T15-38-25-R1-S2.mrxs') # 'current\slide-2022-09-12T15-38-25-R1-S2.mrxs'
   CURRENT_SLIDE_XML = 'current\slide-2022-09-12T15-38-25-R1-S2\Data0021.dat'
   OPENSLIDE_PATH = os.path.abspath('openslide\\bin') # 'E:\\Github\\DemetraAI\\openslide\\bin'
   HDD_SLIDES = os.path.abspath('current')
   HDD_SLIDES_SVS = 'CYTOLOGY_2'
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
   ```

7. Start GUI with:
   
   ```bash
   python run.py
   ```
