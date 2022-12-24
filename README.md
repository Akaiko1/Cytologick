# DemetraAI

## Installation instructions

### Windows 10

1. Install **python3**, version **3.10.6** or later

2. Install Openslide from <https://openslide.org>

3. Create and activate new venv with **cmd** or **powershell** in local folder

4. Run command:

   ``` bash
   pip install -r requirements.txt
   ```

5. Also run:

   ``` bash
   pip install -q git+https://github.com/tensorflow/examples.git
   ```

6. Modify settings at config.py (debug config file with auxillary parameters, a subject to change later)

   **OPENSLIDE_PATH** is necessary! This is an absolute path to your OpenSlide installation folder

   ``` python
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
   ```

7. Start GUI with:

   ``` bash
   python start.py
   ```
