#include "mainwindow.h"
#include "menuwindow.h"
#include "previewwindow.h"
#include "graphics.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QScrollBar>
#include <QPainter>
#include <QMouseEvent>
#include <QMessageBox>
#include <QApplication>
#include <opencv2/imgcodecs.hpp>

namespace cytologick {

// ============================================================================
// ImageLabel implementation
// ============================================================================

ImageLabel::ImageLabel(MainWindow* mainWindow, QWidget* parent)
    : QLabel(parent)
    , m_mainWindow(mainWindow)
{
    setMouseTracking(true);
}

void ImageLabel::setSlideImage(const QPixmap& pixmap) {
    m_slideImage = pixmap;
    resize(pixmap.size());
    update();
}

void ImageLabel::setSelectionRect(const QPoint& start, const QPoint& end, bool visible) {
    m_selStart = start;
    m_selEnd = end;
    m_showSelection = visible;
    update();
}

void ImageLabel::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);

    if (m_slideImage.isNull()) {
        QLabel::paintEvent(event);
        return;
    }

    QPainter painter(this);
    painter.drawPixmap(rect(), m_slideImage);

    // Draw selection rectangle if dragging
    if (m_showSelection) {
        painter.setPen(QPen(Qt::black, 2, Qt::SolidLine));
        painter.setBrush(QBrush(Qt::green, Qt::DiagCrossPattern));
        QRect selectionRect(m_selStart, m_selEnd);
        painter.drawRect(selectionRect.normalized());
    }
}

void ImageLabel::mousePressEvent(QMouseEvent* event) {
    m_selStart = event->pos();
    m_showSelection = false;
    m_mainWindow->updateStatusBar(QString("Click: %1, %2").arg(m_selStart.x()).arg(m_selStart.y()));
    QLabel::mousePressEvent(event);
}

void ImageLabel::mouseMoveEvent(QMouseEvent* event) {
    if (event->buttons() & Qt::LeftButton) {
        m_selEnd = event->pos();
        m_showSelection = true;
        update();
    }
    QLabel::mouseMoveEvent(event);
}

void ImageLabel::mouseReleaseEvent(QMouseEvent* event) {
    m_selEnd = event->pos();
    m_showSelection = false;
    m_mainWindow->updateStatusBar(QString("Release: %1, %2").arg(m_selEnd.x()).arg(m_selEnd.y()));
    m_mainWindow->onSelectionComplete(m_selStart, m_selEnd);
    update();
    QLabel::mouseReleaseEvent(event);
}

// ============================================================================
// MainWindow implementation
// ============================================================================

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    m_config = Config::load();
    setupUi();
    loadModel();

    // Show slide selection menu
    showSlideMenu();
}

MainWindow::~MainWindow() = default;

void MainWindow::setupUi() {
    setWindowTitle("Cytologick");
    setGeometry(100, 100, 800, 800);

    // Central widget with layout
    QWidget* centralWidget = new QWidget(this);
    QVBoxLayout* mainLayout = new QVBoxLayout(centralWidget);

    // Scroll area for slide image
    m_scrollArea = new QScrollArea(this);
    m_scrollArea->setWidgetResizable(false);
    m_scrollArea->setAlignment(Qt::AlignCenter);

    // Image display label - custom class with paint override
    m_imageLabel = new ImageLabel(this);
    m_scrollArea->setWidget(m_imageLabel);

    // Status bar at bottom
    m_statusLabel = new QLabel("Ready");
    m_statusLabel->setStyleSheet("background-color: black; color: white; padding: 5px;");
    m_statusLabel->setFixedHeight(30);

    mainLayout->addWidget(m_scrollArea);
    mainLayout->addWidget(m_statusLabel);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    setCentralWidget(centralWidget);
}

void MainWindow::loadModel() {
    auto modelPath = m_config.findModelFile();

    if (modelPath.empty()) {
        updateStatusBar("No model found in _main/ directory");
        return;
    }

    updateStatusBar("Loading model: " + QString::fromStdString(modelPath.filename().string()));
    QApplication::processEvents();

    if (m_inference.loadModel(modelPath, true)) {
        updateStatusBar("Model loaded: " + QString::fromStdString(modelPath.filename().string()));
    } else {
        updateStatusBar("Failed to load model: " + QString::fromStdString(m_inference.getLastError()));
    }
}

void MainWindow::showSlideMenu() {
    m_menuWindow = std::make_unique<MenuWindow>(this);
    m_menuWindow->show();
}

void MainWindow::setSlideImage(const QPixmap& pixmap, int scaleFactor) {
    m_scaleFactor = scaleFactor;
    m_imageLabel->setSlideImage(pixmap);

    // Center scroll position
    m_scrollArea->horizontalScrollBar()->setValue(m_imageLabel->width() / 3);
    m_scrollArea->verticalScrollBar()->setValue(m_imageLabel->height() / 3);
}

void MainWindow::updateStatusBar(const QString& message) {
    m_statusLabel->setText(message);
}

void MainWindow::onSelectionComplete(const QPoint& pressPos, const QPoint& releasePos) {
    m_pressPos = pressPos;
    m_releasePos = releasePos;
    showPreview();
}

void MainWindow::showPreview() {
    if (!m_slideReader.isOpen()) {
        QMessageBox::warning(this, "Error", "No slide is open");
        return;
    }

    // Calculate region coordinates with scaling
    int x = m_pressPos.x() * m_scaleFactor;
    int y = m_pressPos.y() * m_scaleFactor;
    int width = (m_releasePos.x() - m_pressPos.x()) * m_scaleFactor;
    int height = (m_releasePos.y() - m_pressPos.y()) * m_scaleFactor;

    // Skip if no selection
    if (width == 0 || height == 0) return;

    // Handle negative dimensions (drag direction)
    if (width < 0) {
        x += width;
        width = -width;
    }
    if (height < 0) {
        y += height;
        height = -height;
    }

    // Correct size to match model requirements
    auto [correctedWidth, correctedHeight] = getCorrectedSize(width, height, m_config.imageChunk[0]);
    width = correctedWidth;
    height = correctedHeight;

    if (width <= 0 || height <= 0) return;

    // Read region from slide
    cv::Mat region = m_slideReader.readRegionRGB(x, y, 0, width, height);
    if (region.empty()) {
        QMessageBox::warning(this, "Error", "Failed to read region from slide");
        return;
    }

    // Save for preview
    cv::imwrite("gui_preview.bmp", region);

    // Convert to QPixmap
    QImage qImage(region.data, region.cols, region.rows, region.step, QImage::Format_RGB888);
    QPixmap pixmap = QPixmap::fromImage(qImage.copy());

    // Show preview window
    PreviewWindow* preview = new PreviewWindow(this, pixmap, region);
    preview->setAttribute(Qt::WA_DeleteOnClose);
    preview->show();
}

} // namespace cytologick
