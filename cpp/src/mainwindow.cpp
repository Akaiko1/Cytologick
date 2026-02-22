#include "mainwindow.h"
#include "menuwindow.h"
#include "previewwindow.h"
#include "graphics.h"
#include "workingdir_prefs.h"

#include <QVBoxLayout>
#include <QScrollBar>
#include <QPainter>
#include <QMouseEvent>
#include <QMessageBox>
#include <QApplication>
#include <QMenuBar>
#include <QMenu>
#include <QFileDialog>
#include <QDir>
#include <QTimer>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cytologick {

namespace {
constexpr int kOverviewOverlayWidth = 240;
constexpr int kOverviewOverlayHeight = 240;
constexpr int kOverviewOverlayMargin = 12;
}

// ============================================================================
// OverviewMapWidget implementation
// ============================================================================

OverviewMapWidget::OverviewMapWidget(QWidget* parent)
    : QWidget(parent)
{
    setMinimumSize(220, 220);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setMouseTracking(true);
}

void OverviewMapWidget::setOverview(const QImage& image, double downsampleToLevel0, const QSize& level0Size) {
    m_overview = image;
    m_overviewDownsample = (downsampleToLevel0 <= 0.0) ? 1.0 : downsampleToLevel0;
    m_level0Size = level0Size;
    update();
}

void OverviewMapWidget::setViewportLevel0(const QRectF& viewportLevel0) {
    m_viewportLevel0 = viewportLevel0;
    update();
}

void OverviewMapWidget::clear() {
    m_overview = QImage();
    m_level0Size = QSize();
    m_viewportLevel0 = QRectF();
    m_overviewDownsample = 1.0;
    update();
}

QRect OverviewMapWidget::imageTargetRect() const {
    if (m_overview.isNull() || width() <= 4 || height() <= 4) return {};

    const QRect bounds = rect().adjusted(4, 4, -4, -4);
    const QSize imgSize = m_overview.size();
    QSize fit = imgSize;
    fit.scale(bounds.size(), Qt::KeepAspectRatio);
    const int x = bounds.x() + (bounds.width() - fit.width()) / 2;
    const int y = bounds.y() + (bounds.height() - fit.height()) / 2;
    return QRect(x, y, fit.width(), fit.height());
}

QPointF OverviewMapWidget::widgetToLevel0(const QPoint& pos) const {
    const QRect target = imageTargetRect();
    if (target.isEmpty() || m_overview.isNull()) return QPointF();

    const QPoint clamped(
        std::max(target.left(), std::min(pos.x(), target.right())),
        std::max(target.top(), std::min(pos.y(), target.bottom()))
    );
    const double xNorm = (clamped.x() - target.left()) / std::max(1.0, static_cast<double>(target.width()));
    const double yNorm = (clamped.y() - target.top()) / std::max(1.0, static_cast<double>(target.height()));
    const double xOverview = xNorm * m_overview.width();
    const double yOverview = yNorm * m_overview.height();
    return QPointF(xOverview * m_overviewDownsample, yOverview * m_overviewDownsample);
}

void OverviewMapWidget::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);
    QPainter p(this);
    p.fillRect(rect(), QColor(24, 24, 24));

    const QRect target = imageTargetRect();
    if (target.isEmpty() || m_overview.isNull()) {
        p.setPen(QColor(190, 190, 190));
        p.drawText(rect(), Qt::AlignCenter, "No overview");
        return;
    }

    p.drawImage(target, m_overview);
    p.setPen(QPen(QColor(255, 255, 255, 120), 1));
    p.setBrush(Qt::NoBrush);
    p.drawRect(target);

    if (!m_viewportLevel0.isEmpty() && m_overviewDownsample > 0.0) {
        const double sx = target.width() / std::max(1.0, static_cast<double>(m_overview.width()));
        const double sy = target.height() / std::max(1.0, static_cast<double>(m_overview.height()));

        QRectF vr(
            target.left() + (m_viewportLevel0.left() / m_overviewDownsample) * sx,
            target.top() + (m_viewportLevel0.top() / m_overviewDownsample) * sy,
            (m_viewportLevel0.width() / m_overviewDownsample) * sx,
            (m_viewportLevel0.height() / m_overviewDownsample) * sy
        );
        p.setPen(QPen(QColor(255, 80, 80, 230), 2));
        p.drawRect(vr);
    }
}

void OverviewMapWidget::mousePressEvent(QMouseEvent* event) {
    if (!event || event->button() != Qt::LeftButton) return;
    m_mouseDown = true;
    emit jumpRequestedLevel0(widgetToLevel0(event->pos()));
}

void OverviewMapWidget::mouseMoveEvent(QMouseEvent* event) {
    if (!event || !m_mouseDown) return;
    emit jumpRequestedLevel0(widgetToLevel0(event->pos()));
}

void OverviewMapWidget::mouseReleaseEvent(QMouseEvent* event) {
    if (!event) return;
    if (event->button() == Qt::LeftButton) {
        m_mouseDown = false;
    }
}

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
    m_useVirtualCanvas = false;
    m_slideImage = pixmap;
    m_cachedRegionImage = QImage();
    m_cachedRegionRect = QRect();
    resize(pixmap.size());
    update();
}

void ImageLabel::setVirtualCanvas(const QSize& size, int levelIndex, double downsample) {
    m_useVirtualCanvas = true;
    m_slideImage = QPixmap();
    m_levelIndex = levelIndex;
    m_downsample = downsample > 0.0 ? downsample : 1.0;
    m_cachedRegionImage = QImage();
    m_cachedRegionRect = QRect();

    const int w = std::max(1, size.width());
    const int h = std::max(1, size.height());
    resize(w, h);
    update();
}

void ImageLabel::invalidateRegionCache() {
    m_cachedRegionImage = QImage();
    m_cachedRegionRect = QRect();
    update();
}

void ImageLabel::setSelectionRect(const QPoint& start, const QPoint& end, bool visible) {
    m_selStart = start;
    m_selEnd = end;
    m_showSelection = visible;
    update();
}

QRect ImageLabel::computeCacheRect(const QRect& targetRect) const {
    constexpr int kPadding = 256;
    QRect expanded = targetRect.adjusted(-kPadding, -kPadding, kPadding, kPadding);
    return expanded.intersected(rect());
}

void ImageLabel::paintEvent(QPaintEvent* event) {
    if (!m_useVirtualCanvas) {
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
        return;
    }

    if (!m_mainWindow || !m_mainWindow->getSlideReader().isOpen()) {
        QLabel::paintEvent(event);
        return;
    }

    QPainter painter(this);
    const QRect targetRect = (event ? event->rect() : rect()).intersected(rect());
    if (!targetRect.isEmpty()) {
        if (m_cachedRegionImage.isNull() || !m_cachedRegionRect.contains(targetRect)) {
            const QRect cacheRect = computeCacheRect(targetRect);
            if (!cacheRect.isEmpty()) {
                const int64_t xLevel0 = static_cast<int64_t>(std::llround(cacheRect.x() * m_downsample));
                const int64_t yLevel0 = static_cast<int64_t>(std::llround(cacheRect.y() * m_downsample));
                const int64_t w = cacheRect.width();
                const int64_t h = cacheRect.height();

                cv::Mat region = m_mainWindow->getSlideReader().readRegionRGB(
                    xLevel0, yLevel0, m_levelIndex, w, h
                );
                if (!region.empty()) {
                    QImage img(region.data, region.cols, region.rows, region.step, QImage::Format_RGB888);
                    m_cachedRegionImage = img.copy();
                    m_cachedRegionRect = cacheRect;
                }
            }
        }

        if (!m_cachedRegionImage.isNull()) {
            const QRect sourceRect = targetRect.translated(-m_cachedRegionRect.topLeft());
            painter.drawImage(targetRect.topLeft(), m_cachedRegionImage, sourceRect);
        }
    }

    if (m_showSelection) {
        painter.setPen(QPen(Qt::black, 2, Qt::SolidLine));
        painter.setBrush(QBrush(Qt::green, Qt::DiagCrossPattern));
        painter.drawRect(QRect(m_selStart, m_selEnd).normalized());
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
    if (const auto rememberedDir = loadRememberedWorkingDir(); !rememberedDir.empty()) {
        applyWorkingDirectory(rememberedDir, false);
    }
    setupUi();
    loadModel();

    // Show slide selection menu
    showSlideMenu();
}

MainWindow::~MainWindow() = default;

void MainWindow::setupUi() {
    setWindowTitle("Cytologick");
    setGeometry(100, 100, 800, 800);

    QMenu* fileMenu = menuBar()->addMenu("File");
    QAction* selectSlideAction = fileMenu->addAction("Select Slide...");
    QAction* selectWorkingDirAction = fileMenu->addAction("Select Working Folder...");
    connect(selectSlideAction, &QAction::triggered, this, &MainWindow::showSlideMenu);
    connect(selectWorkingDirAction, &QAction::triggered, this, &MainWindow::onSelectWorkingDirectory);

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

    connect(m_scrollArea->horizontalScrollBar(), &QScrollBar::valueChanged, this, [this]() {
        updateOverviewViewport();
    });
    connect(m_scrollArea->verticalScrollBar(), &QScrollBar::valueChanged, this, [this]() {
        updateOverviewViewport();
    });

    // Status bar at bottom
    m_statusLabel = new QLabel("Ready");
    m_statusLabel->setStyleSheet("background-color: black; color: white; padding: 5px;");
    m_statusLabel->setFixedHeight(30);

    mainLayout->addWidget(m_scrollArea);
    mainLayout->addWidget(m_statusLabel);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    setCentralWidget(centralWidget);

    m_overviewMap = new OverviewMapWidget(m_scrollArea->viewport());
    m_overviewMap->setFixedSize(kOverviewOverlayWidth, kOverviewOverlayHeight);
    m_overviewMap->setStyleSheet(
        "background-color: rgba(24, 24, 24, 220);"
        "border: 1px solid rgba(255, 255, 255, 80);"
        "border-radius: 6px;"
    );
    m_overviewMap->show();
    m_overviewMap->raise();
    connect(m_overviewMap, &OverviewMapWidget::jumpRequestedLevel0, this, &MainWindow::onMapJumpRequested);
    m_scrollArea->viewport()->installEventFilter(this);
    positionOverviewMapOverlay();
}

bool MainWindow::eventFilter(QObject* watched, QEvent* event) {
    if (m_scrollArea && watched == m_scrollArea->viewport() && event) {
        if (event->type() == QEvent::Resize || event->type() == QEvent::Show) {
            positionOverviewMapOverlay();
        }
    }
    return QMainWindow::eventFilter(watched, event);
}

void MainWindow::positionOverviewMapOverlay() {
    if (!m_overviewMap || !m_scrollArea) return;
    QWidget* viewport = m_scrollArea->viewport();
    if (!viewport) return;

    const int x = std::max(
        kOverviewOverlayMargin,
        viewport->width() - m_overviewMap->width() - kOverviewOverlayMargin
    );
    const int y = std::max(
        kOverviewOverlayMargin,
        viewport->height() - m_overviewMap->height() - kOverviewOverlayMargin
    );

    m_overviewMap->move(x, y);
    m_overviewMap->raise();
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

    // Position menu window to the right edge of main window
    QPoint mainPos = pos();
    int menuX = mainPos.x() + width();
    int menuY = mainPos.y();
    m_menuWindow->move(menuX, menuY);

    m_menuWindow->show();
}

void MainWindow::onSlideOpened(const std::filesystem::path& path) {
    const bool changed = (path != m_slidePath);
    m_slidePath = path;
    m_levelLoaded = false;
    if (changed) {
        rebuildOverviewMap();
    }
}

void MainWindow::loadSlideLevel(int levelIndex) {
    if (!m_slideReader.isOpen()) return;

    QPointF oldCenterLevel0;
    bool hasOldCenter = false;
    if (m_levelLoaded && m_scrollArea && m_scaleFactor > 0.0) {
        const QSize vp = m_scrollArea->viewport()->size();
        if (vp.width() > 0 && vp.height() > 0) {
            const double cxLevel = m_scrollArea->horizontalScrollBar()->value() + vp.width() / 2.0;
            const double cyLevel = m_scrollArea->verticalScrollBar()->value() + vp.height() / 2.0;
            oldCenterLevel0 = QPointF(cxLevel * m_scaleFactor, cyLevel * m_scaleFactor);
            hasOldCenter = true;
        }
    }

    const auto [w64, h64] = m_slideReader.getLevelDimensions(levelIndex);
    if (w64 <= 0 || h64 <= 0) return;

    const int w = static_cast<int>(std::min<int64_t>(w64, std::numeric_limits<int>::max()));
    const int h = static_cast<int>(std::min<int64_t>(h64, std::numeric_limits<int>::max()));
    const double downsample = m_slideReader.getLevelDownsample(levelIndex);
    m_currentLevel = levelIndex;
    m_scaleFactor = downsample > 0.0 ? downsample : 1.0;

    m_imageLabel->setVirtualCanvas(QSize(w, h), levelIndex, m_scaleFactor);
    m_imageLabel->invalidateRegionCache();
    if (!m_levelLoaded) {
        m_levelLoaded = true;
    }

    if (hasOldCenter) {
        QTimer::singleShot(0, this, [this, oldCenterLevel0]() {
            centerViewOnLevel0(oldCenterLevel0);
        });
    } else {
        QTimer::singleShot(0, this, [this]() {
            m_scrollArea->horizontalScrollBar()->setValue(m_imageLabel->width() / 3);
            m_scrollArea->verticalScrollBar()->setValue(m_imageLabel->height() / 3);
            updateOverviewViewport();
        });
    }

    updateStatusBar(QString("Level %1 loaded (%2x%3), downsample=%4")
        .arg(levelIndex)
        .arg(w)
        .arg(h)
        .arg(m_scaleFactor, 0, 'f', 3));
}

void MainWindow::onSelectWorkingDirectory() {
    QString startDir = QDir::currentPath();
    if (!m_config.slideDir.empty()) {
        startDir = QString::fromStdString(m_config.slideDir.string());
    } else if (!m_config.hddSlides.empty()) {
        startDir = QString::fromStdString(m_config.hddSlides.string());
    }

    const QString selectedDir = QFileDialog::getExistingDirectory(
        this,
        "Select Working Folder",
        startDir,
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks
    );
    if (selectedDir.isEmpty()) return;

    applyWorkingDirectory(std::filesystem::path(selectedDir.toStdString()), true);
    updateStatusBar(QString("Working folder set: %1").arg(selectedDir));
    showSlideMenu();
}

void MainWindow::applyWorkingDirectory(const std::filesystem::path& dir, bool persist) {
    if (dir.empty()) return;
    m_config.slideDir = dir;
    m_config.hddSlides.clear();
    if (persist) {
        saveRememberedWorkingDir(dir);
    }
}

void MainWindow::rebuildOverviewMap() {
    if (!m_overviewMap) return;
    m_overviewMap->clear();
    if (!m_slideReader.isOpen()) return;

    const int levels = m_slideReader.getLevelCount();
    if (levels <= 0) return;

    const int overviewLevel = levels - 1;
    auto [ow, oh] = m_slideReader.getLevelDimensions(overviewLevel);
    if (ow <= 0 || oh <= 0) return;

    cv::Mat rgb = m_slideReader.readRegionRGB(0, 0, overviewLevel, ow, oh);
    if (rgb.empty()) return;

    const int maxSide = std::max(rgb.cols, rgb.rows);
    double effectiveDownsample = m_slideReader.getLevelDownsample(overviewLevel);
    cv::Mat rgbScaled = rgb;
    if (maxSide > 2048) {
        const double scale = 2048.0 / static_cast<double>(maxSide);
        cv::resize(rgb, rgbScaled, cv::Size(), scale, scale, cv::INTER_AREA);
        if (scale > 1e-9) {
            effectiveDownsample /= scale;
        }
    }

    QImage img(rgbScaled.data, rgbScaled.cols, rgbScaled.rows, rgbScaled.step, QImage::Format_RGB888);
    auto [l0w, l0h] = m_slideReader.getLevelDimensions(0);
    m_overviewMap->setOverview(
        img.copy(),
        effectiveDownsample,
        QSize(static_cast<int>(l0w), static_cast<int>(l0h))
    );
    updateOverviewViewport();
}

void MainWindow::updateOverviewViewport() {
    if (!m_overviewMap || !m_scrollArea || !m_slideReader.isOpen()) return;
    if (m_scaleFactor <= 0.0) return;

    const QSize vp = m_scrollArea->viewport()->size();
    const QRectF vr0(
        m_scrollArea->horizontalScrollBar()->value() * m_scaleFactor,
        m_scrollArea->verticalScrollBar()->value() * m_scaleFactor,
        vp.width() * m_scaleFactor,
        vp.height() * m_scaleFactor
    );
    m_overviewMap->setViewportLevel0(vr0);
}

void MainWindow::centerViewOnLevel0(const QPointF& centerLevel0) {
    if (!m_scrollArea || m_scaleFactor <= 0.0) return;
    const QSize vp = m_scrollArea->viewport()->size();
    const double cxLevel = centerLevel0.x() / m_scaleFactor;
    const double cyLevel = centerLevel0.y() / m_scaleFactor;

    const int newH = static_cast<int>(std::llround(cxLevel - vp.width() / 2.0));
    const int newV = static_cast<int>(std::llround(cyLevel - vp.height() / 2.0));

    m_scrollArea->horizontalScrollBar()->setValue(newH);
    m_scrollArea->verticalScrollBar()->setValue(newV);
    updateOverviewViewport();
}

void MainWindow::onMapJumpRequested(QPointF centerLevel0) {
    centerViewOnLevel0(centerLevel0);
}

void MainWindow::setSlideImage(const QPixmap& pixmap, double scaleFactor) {
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

    const auto [w0, h0] = m_slideReader.getLevelDimensions(0);
    if (w0 <= 0 || h0 <= 0) return;

    const double ds = m_slideReader.getLevelDownsample(m_currentLevel);
    const double downsample = ds > 0.0 ? ds : 1.0;

    // Python-equivalent selection mapping (without QRect +1 semantics).
    double x = static_cast<double>(m_pressPos.x()) * downsample;
    double y = static_cast<double>(m_pressPos.y()) * downsample;
    double width = static_cast<double>(m_releasePos.x() - m_pressPos.x()) * downsample;
    double height = static_cast<double>(m_releasePos.y() - m_pressPos.y()) * downsample;

    if (std::abs(width) < 1e-9 || std::abs(height) < 1e-9) return;

    if (width < 0.0) {
        x += width;
        width = -width;
    }
    if (height < 0.0) {
        y += height;
        height = -height;
    }

    int xi = static_cast<int>(std::floor(x));
    int yi = static_cast<int>(std::floor(y));
    int wi = std::max(1, static_cast<int>(std::floor(width)));
    int hi = std::max(1, static_cast<int>(std::floor(height)));

    xi = std::clamp(xi, 0, static_cast<int>(w0) - 1);
    yi = std::clamp(yi, 0, static_cast<int>(h0) - 1);
    if (xi + wi > static_cast<int>(w0)) wi = static_cast<int>(w0) - xi;
    if (yi + hi > static_cast<int>(h0)) hi = static_cast<int>(h0) - yi;
    if (wi <= 0 || hi <= 0) return;

    // Read region from slide
    cv::Mat region = m_slideReader.readRegionRGB(xi, yi, 0, wi, hi);
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
