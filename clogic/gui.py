"""
Desktop GUI for Cytologick - PyQt5-based slide viewer with AI analysis.

This module provides a desktop application for:
- Viewing whole-slide microscopy images (MRXS format)
- Selecting regions of interest for AI analysis
- Running inference using local or remote models
- Displaying analysis results with confidence overlays

Classes:
    Preview: Analysis preview window with inference controls
    Menu: Slide selection and zoom level controls  
    Viewer: Main application window with slide navigation
"""

import glob
import os
import socket
from urllib.parse import urlparse

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QHBoxLayout, QLabel, QPushButton,
    QRadioButton, QScrollArea, QSlider, QVBoxLayout, QWidget
)

import config
import clogic.contours as contours
import clogic.graphics as drawing
import clogic.model_loading as model_loading
import clogic.inference_utils as inference_utils

# Framework-specific imports
if config.FRAMEWORK.lower() == 'pytorch':
    import clogic.inference_pytorch as inference
else:
    import clogic.inference as inference

# OpenSlide import with DLL handling for Windows
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(config.OPENSLIDE_PATH):
        import openslide
else:
    import openslide


# =============================================================================
# Preview Window - Analysis results display
# =============================================================================

class Preview(QWidget):
    """
    Analysis preview window showing inference results.
    
    Displays the selected region with optional overlay showing detected
    abnormalities. Provides controls for inference mode selection and
    confidence threshold adjustment.
    
    Attributes:
        modes: Available inference modes ['smooth', 'direct', 'remote']
        mode_names: Human-readable names for each mode
        conf_threshold: Detection confidence threshold (0-1)
    """
    
    # Inference mode options
    modes = ['smooth', 'direct', 'remote']
    mode_names = ['Local: Comprehensive', 'Local: Fast', 'Cloud: Fast']

    def __init__(self, parent: 'Viewer', pixmap: QPixmap):
        """
        Initialize the preview window.
        
        Args:
            parent: Parent Viewer widget (provides model access)
            pixmap: Image to display for analysis
        """
        super().__init__()
        
        self.parent = parent
        self.image = pixmap
        self.map = None  # Overlay map (set after analysis)
        self.conf_threshold = 0.6  # Default 60% confidence
        
        self.setWindowTitle("Preview")
        self._setup_layout()
        self.setMaximumSize(2000, 1200)
    
    def _setup_layout(self):
        """Configure the preview window layout with controls."""
        main_layout = QHBoxLayout()
        
        # --- Control Panel (right side) ---
        control_layout = QVBoxLayout()
        control_widget = QWidget()
        control_widget.setMaximumWidth(200)
        control_widget.setLayout(control_layout)
        
        # Add inference mode radio buttons
        remote_ok = inference_utils.check_remote_available()
        self._add_mode_buttons(control_layout, remote_ok)
        
        # Display area for the image
        self.display = QLabel()
        self.display.paintEvent = self._paint_image
        
        # Results info label
        self.info = QLabel()
        self.info.setText('Analysis \nResults')
        
        # Cloud status indicator
        self.cloud_status = QLabel()
        self.cloud_status.setText('Cloud: Available' if remote_ok else 'Cloud: Unavailable')
        self.cloud_status.setStyleSheet('color: #2ecc71;' if remote_ok else 'color: #e74c3c;')
        
        # Confidence threshold slider
        self.conf_label = QLabel()
        self._update_conf_label()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setSingleStep(1)
        self.slider.setValue(int(self.conf_threshold * 100))
        self.slider.valueChanged.connect(self._on_conf_changed)
        
        # Analyze button
        self.button = QPushButton('Analyze')
        self.button.clicked.connect(self._run_analysis)
        
        # Assemble the control panel
        control_layout.addWidget(self.cloud_status)
        control_layout.addWidget(self.conf_label)
        control_layout.addWidget(self.slider)
        control_layout.addWidget(self.info)
        control_layout.addWidget(self.button)
        
        # Assemble main layout
        main_layout.addWidget(self.display)
        main_layout.addWidget(control_widget)
        
        self.setLayout(main_layout)
        self.resize(self.image.width(), self.image.height())
    
    def _add_mode_buttons(self, layout: QVBoxLayout, remote_ok: bool):
        """
        Add inference mode selection radio buttons.
        
        Args:
            layout: Layout to add buttons to
            remote_ok: Whether remote endpoint is available
        """
        for idx, mode in enumerate(self.modes):
            # Skip cloud option if endpoint not reachable
            if mode == 'remote' and not remote_ok:
                continue
            
            # Skip local modes if no model is loaded
            if self.parent.model is None and mode != 'remote':
                continue
            
            radiobutton = QRadioButton(self.mode_names[idx])
            
            # Fallback to local mode if remote unavailable
            if config.UNET_PRED_MODE == 'remote' and not remote_ok and self.parent.model is not None:
                config.UNET_PRED_MODE = 'direct'
            
            if mode == config.UNET_PRED_MODE:
                radiobutton.setChecked(True)
            
            radiobutton.mode = mode
            radiobutton.toggled.connect(self._on_mode_selected)
            layout.addWidget(radiobutton)
    
    def _update_conf_label(self):
        """Update the confidence threshold label text."""
        self.conf_label.setText(f'Min confidence: {int(self.conf_threshold * 100)}%')
    
    def _on_conf_changed(self, value: int):
        """Handle confidence slider value change."""
        self.conf_threshold = float(value) / 100.0
        self._update_conf_label()
    
    def _on_mode_selected(self):
        """Handle inference mode radio button selection."""
        radioButton = self.sender()
        config.UNET_PRED_MODE = radioButton.mode
    
    def _run_analysis(self):
        """Execute model inference on the preview image."""
        # Load and preprocess the source image
        source = cv2.imread('gui_preview.bmp', 1)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        # Run inference using shared utility
        pathology_map = inference_utils.run_inference(
            source,
            self.parent.model,
            mode=config.UNET_PRED_MODE,
            classes=config.CLASSES,
            shapes=config.IMAGE_SHAPE
        )
        
        # Process results into visualization
        markup, stats = inference_utils.process_pathology_map(
            pathology_map, 
            threshold=self.conf_threshold
        )
        cv2.imwrite('gui_map.png', markup)
        
        # Update display
        self.map = QPixmap('gui_map.png')
        self.info.setText(inference_utils.format_detection_stats(stats))
        self.display.repaint()
    
    def _paint_image(self, event):
        """Custom paint event to draw image with overlay."""
        painter = QPainter(self.display)
        painter.drawPixmap(self.display.rect(), self.image)
        
        if self.map is not None:
            painter.drawPixmap(self.display.rect(), self.map)


# =============================================================================
# Menu Window - Slide and zoom level selection
# =============================================================================

class Menu(QWidget):
    """
    Slide selection menu window.
    
    Provides controls for:
    - Selecting slide files from the configured directory
    - Choosing zoom level for viewing
    """

    def __init__(self, parent: 'Viewer'):
        """
        Initialize the menu window.
        
        Args:
            parent: Parent Viewer widget to update on selection
        """
        super().__init__()
        
        self.parent = parent
        self.levels = []  # Available zoom levels for current slide
        
        # Find all MRXS slide files recursively
        # Find all MRXS slide files recursively from both locations
        self.slide_list = []
        search_paths = [config.HDD_SLIDES, config.SLIDE_DIR]
        
        for path in search_paths:
            if os.path.exists(path):
                self.slide_list.extend(glob.glob(
                    os.path.join(path, '**', '*.mrxs'), 
                    recursive=True
                ))
        
        # Remove duplicates if any
        self.slide_list = sorted(list(set(self.slide_list)))
        
        self.setWindowTitle("Select Slide")
        self._setup_layout()
    
    def _setup_layout(self):
        """Configure the menu layout with dropdowns."""
        layout = QVBoxLayout()

        # Slide file selection
        slide_label = QLabel("Slide:")
        self.slide_select = QComboBox()
        self.slide_select.setMinimumWidth(300)
        self.slide_select.addItems(self.slide_list)
        self.slide_select.currentIndexChanged.connect(self._on_slide_selected)
        self.slide_select.activated.connect(self._on_slide_selected)

        # Zoom level selection
        zoom_label = QLabel("Zoom Level:")
        self.level_select = QComboBox()
        self.level_select.setMinimumWidth(100)
        self.level_select.currentIndexChanged.connect(self._on_level_selected)

        layout.addWidget(slide_label)
        layout.addWidget(self.slide_select)
        layout.addWidget(zoom_label)
        layout.addWidget(self.level_select)
        self.setLayout(layout)
    
    def _on_slide_selected(self, index: int):
        """Handle slide file selection."""
        slide_path = self.slide_list[index]
        self.parent.current_slide = openslide.OpenSlide(slide_path)
        self.levels = self.parent.current_slide.level_dimensions

        # Populate zoom levels with descriptive labels (limit to < 4 to prevent memory issues)
        self.level_select.clear()
        for i, dims in enumerate(self.levels):
            if i >= 4:
                break
            w, h = dims
            self.level_select.addItem(f"Level {i} ({w}x{h})")
    
    def _on_level_selected(self, level_index: int):
        """Handle zoom level selection."""
        if level_index < 0:
            return
        
        # Calculate position from the end (higher index = more zoom)
        position = len(self.levels) - level_index - 1
        
        # Calculate scaling coefficient for region preview
        self.parent.prop = int(
            self.levels[0][0] / self.levels[position][0]
        )
        
        # Read and display the slide at selected level
        width, height = self.levels[position]
        slide = self.parent.current_slide.read_region(
            (0, 0), position, (width, height)
        )
        slide.save('gui_temp.bmp', 'bmp')
        
        # Update viewer display
        self.parent.slideImage = QPixmap('gui_temp.bmp')
        self.parent.image.resize(
            self.parent.slideImage.width(), 
            self.parent.slideImage.height()
        )
        
        # Center the scroll position
        self.parent.scrollArea.horizontalScrollBar().setValue(
            int(self.parent.image.width() / 3)
        )
        self.parent.scrollArea.verticalScrollBar().setValue(
            int(self.parent.image.height() / 2.5)
        )
        self.parent.image.repaint()


# =============================================================================
# Main Viewer Window
# =============================================================================

class Viewer(QWidget):
    """
    Main application window for slide viewing and region selection.
    
    Provides:
    - Scrollable slide view with drag-to-select regions
    - Region preview and analysis via Preview window
    - Status display showing coordinates
    
    Attributes:
        model: Loaded inference model (PyTorch or TensorFlow)
        current_slide: Currently open OpenSlide object
        slideImage: QPixmap of current slide view
    """
    
    # Class-level state
    doDraw = False      # Whether to draw selection rectangle
    drag = False        # Whether mouse is being dragged
    slideImage = None   # Current slide image
    model = None        # Loaded model

    def __init__(self):
        """Initialize the main viewer window."""
        super().__init__()
        
        self.appw, self.apph = 800, 800
        self.current_slide = None
        
        # Selection coordinates
        self.p_x = self.p_y = 0  # Press position
        self.r_x = self.r_y = 0  # Release position
        self.prop = 1  # Scaling coefficient
        
        self.setWindowTitle("Cytologick")
        self.setGeometry(100, 100, self.appw, self.apph)
        self._setup_layout()
        self.show()
        
        # Show slide menu
        self.slide_menu = Menu(self)
        self.slide_menu.show()
        
        # Load AI model
        self._load_model()
    
    def _load_model(self):
        """Load the local inference model."""
        self.model = model_loading.load_local_model()
    
    def _setup_layout(self):
        """Configure the viewer window layout."""
        # Main display area with scrolling
        display = QHBoxLayout()
        
        self.image = QLabel()
        self.image.mousePressEvent = self._on_mouse_press
        self.image.mouseMoveEvent = self._on_mouse_move
        self.image.mouseReleaseEvent = self._on_mouse_release
        self.image.paintEvent = self._paint_rect
        
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidget(self.image)
        display.addWidget(self.scrollArea)
        
        # Status terminal at bottom
        wrapping = QVBoxLayout()
        
        self.terminal = QLabel()
        self.terminal.setStyleSheet(
            "background-color: black; color: white; padding: 5px"
        )
        self.terminal.resize(25, 200)
        
        wrapping.addLayout(display)
        wrapping.addWidget(self.terminal)
        self.setLayout(wrapping)
    
    def _show_preview(self):
        """Open a preview window for the selected region."""
        # Calculate region coordinates with scaling
        x = int(self.p_x * self.prop)
        y = int(self.p_y * self.prop)
        width = int((self.r_x - self.p_x) * self.prop)
        height = int((self.r_y - self.p_y) * self.prop)
        
        if not width or not height:
            return
        
        # Handle negative dimensions (drag direction)
        if width < 0:
            x, width = x + width, abs(width)
        if height < 0:
            y, height = y + height, abs(height)
        
        # Correct size to match model requirements
        width, height = drawing.get_corrected_size(
            width, height, config.IMAGE_CHUNK[0]
        )
        
        # Read region from slide
        region = self.current_slide.read_region((x, y), 0, (width, height))
        region.save('gui_preview.bmp', 'bmp')
        
        # Show preview window
        preview = Preview(self, QPixmap('gui_preview.bmp'))
        preview.show()
    
    # --- Mouse Event Handlers ---
    
    def _on_mouse_press(self, event):
        """Record starting position of drag selection."""
        self.p_x = event.pos().x()
        self.p_y = event.pos().y()
        self.drag = True
        self.terminal.setText(f'Click coordinates: {self.p_x}, {self.p_y}')
    
    def _on_mouse_move(self, event):
        """Update selection rectangle during drag."""
        if not self.drag:
            return
        
        self.r_x = event.pos().x()
        self.r_y = event.pos().y()
        self.doDraw = True
        self.image.repaint()
    
    def _on_mouse_release(self, event):
        """Complete selection and show preview."""
        self.r_x = event.pos().x()
        self.r_y = event.pos().y()
        self.doDraw = False
        self.drag = False
        
        self.terminal.setText(f'Release coordinates: {self.r_x}, {self.r_y}')
        self._show_preview()
    
    def _paint_rect(self, event):
        """Custom paint event to draw slide and selection rectangle."""
        if self.slideImage is None:
            return
        
        painter = QPainter(self.image)
        painter.drawPixmap(self.image.rect(), self.slideImage)
        
        if not self.doDraw:
            return
        
        # Draw selection rectangle
        painter.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.green, Qt.DiagCrossPattern))
        painter.drawRect(
            self.p_x, self.p_y, 
            self.r_x - self.p_x, self.r_y - self.p_y
        )
