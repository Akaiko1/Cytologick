#include "mainwindow.h"
#include "menuwindow.h"
#include "previewwindow.h"
#include "graphics.h"
#include "workingdir_prefs.h"
#include "annotation_overlay.h"
#include "annotator/annotation_io.h"

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
#include <QRegularExpression>
#include <QLineF>
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

int rectNumber(const QString& label) {
    static const QRegularExpression re("^\\s*rect\\s*(\\d+)\\s*$", QRegularExpression::CaseInsensitiveOption);
    const auto m = re.match(label.trimmed());
    if (!m.hasMatch()) return 0;
    bool ok = false;
    const int v = m.captured(1).toInt(&ok);
    return ok ? v : 0;
}

bool hasPositiveIntersection(const QRectF& a, const QRectF& b) {
    const QRectF inter = a.intersected(b);
    return !inter.isEmpty() && inter.width() > 0.0 && inter.height() > 0.0;
}

QRectF computeBbox(const std::vector<QPointF>& points) {
    if (points.empty()) return QRectF();
    double minx = points[0].x();
    double miny = points[0].y();
    double maxx = points[0].x();
    double maxy = points[0].y();
    for (const auto& p : points) {
        minx = std::min(minx, p.x());
        miny = std::min(miny, p.y());
        maxx = std::max(maxx, p.x());
        maxy = std::max(maxy, p.y());
    }
    return QRectF(QPointF(minx, miny), QPointF(maxx, maxy));
}
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
    setAttribute(Qt::WA_TranslucentBackground, true);
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

void OverviewMapWidget::setRectOverlaysLevel0(const std::vector<QRectF>& rects) {
    m_rectOverlaysLevel0 = rects;
    update();
}

void OverviewMapWidget::clear() {
    m_overview = QImage();
    m_level0Size = QSize();
    m_viewportLevel0 = QRectF();
    m_rectOverlaysLevel0.clear();
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
    p.fillRect(rect(), QColor(24, 24, 24, 130));

    const QRect target = imageTargetRect();
    if (target.isEmpty() || m_overview.isNull()) {
        p.setPen(QColor(190, 190, 190));
        p.drawText(rect(), Qt::AlignCenter, "No overview");
        return;
    }

    p.setOpacity(0.82);
    p.drawImage(target, m_overview);
    p.setOpacity(1.0);
    p.setPen(QPen(QColor(255, 255, 255, 120), 1));
    p.setBrush(Qt::NoBrush);
    p.drawRect(target);

    if (!m_rectOverlaysLevel0.empty() && m_overviewDownsample > 0.0) {
        const double sx = target.width() / std::max(1.0, static_cast<double>(m_overview.width()));
        const double sy = target.height() / std::max(1.0, static_cast<double>(m_overview.height()));
        p.setPen(QPen(QColor(0, 235, 255, 210), 1));
        p.setBrush(QColor(0, 235, 255, 40));
        for (const auto& r0 : m_rectOverlaysLevel0) {
            if (!r0.isValid()) continue;
            QRectF rr(
                target.left() + (r0.left() / m_overviewDownsample) * sx,
                target.top() + (r0.top() / m_overviewDownsample) * sy,
                (r0.width() / m_overviewDownsample) * sx,
                (r0.height() / m_overviewDownsample) * sy
            );
            p.drawRect(rr);
        }
    }

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

QScrollArea* ImageLabel::owningScrollArea() const {
    QWidget* w = parentWidget();
    while (w) {
        if (auto* sa = qobject_cast<QScrollArea*>(w)) {
            return sa;
        }
        w = w->parentWidget();
    }
    return nullptr;
}

void ImageLabel::paintEvent(QPaintEvent* event) {
    if (!m_useVirtualCanvas) {
        if (m_slideImage.isNull()) {
            QLabel::paintEvent(event);
            return;
        }

        QPainter painter(this);
        const QRect drawRect = (event ? event->rect() : rect()).intersected(rect());
        painter.drawPixmap(rect(), m_slideImage);

        if (m_mainWindow) {
            m_mainWindow->drawSlideMarkup(
                painter,
                drawRect,
                m_mainWindow->getScaleFactor()
            );
        }

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

    if (m_mainWindow) {
        m_mainWindow->drawSlideMarkup(painter, targetRect, m_downsample);
    }

    if (m_showSelection) {
        painter.setPen(QPen(Qt::black, 2, Qt::SolidLine));
        painter.setBrush(QBrush(Qt::green, Qt::DiagCrossPattern));
        painter.drawRect(QRect(m_selStart, m_selEnd).normalized());
    }
}

void ImageLabel::mousePressEvent(QMouseEvent* event) {
    if (!event) return;

    if (event->button() == Qt::RightButton) {
        m_panning = true;
        m_panMoved = false;
        m_panLastGlobal = event->globalPosition().toPoint();
        setCursor(Qt::ClosedHandCursor);
        event->accept();
        return;
    }

    if (event->button() == Qt::LeftButton) {
        m_leftSelecting = true;
        m_selStart = event->pos();
        m_selEnd = m_selStart;
        m_showSelection = false;
        event->accept();
        return;
    }

    m_selStart = event->pos();
    m_showSelection = false;
    QLabel::mousePressEvent(event);
}

void ImageLabel::mouseMoveEvent(QMouseEvent* event) {
    if (!event) return;

    if (m_panning && (event->buttons() & Qt::RightButton)) {
        const QPoint nowGlobal = event->globalPosition().toPoint();
        const QPoint delta = nowGlobal - m_panLastGlobal;
        if (!delta.isNull()) {
            m_panMoved = true;
            if (QScrollArea* sa = owningScrollArea()) {
                sa->horizontalScrollBar()->setValue(sa->horizontalScrollBar()->value() - delta.x());
                sa->verticalScrollBar()->setValue(sa->verticalScrollBar()->value() - delta.y());
            }
            m_panLastGlobal = nowGlobal;
        }
        event->accept();
        return;
    }

    if (m_leftSelecting && (event->buttons() & Qt::LeftButton)) {
        m_selEnd = event->pos();
        m_showSelection = true;
        update();
        event->accept();
        return;
    }
    QLabel::mouseMoveEvent(event);
}

void ImageLabel::mouseReleaseEvent(QMouseEvent* event) {
    if (!event) return;

    if (event->button() == Qt::RightButton && m_panning) {
        unsetCursor();
        m_panning = false;
        m_panMoved = false;
        event->accept();
        return;
    }

    if (event->button() == Qt::LeftButton && m_leftSelecting) {
        m_selEnd = event->pos();
        m_leftSelecting = false;
        m_showSelection = false;
        m_mainWindow->onSelectionComplete(m_selStart, m_selEnd);
        update();
        event->accept();
        return;
    }

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

    QMenu* viewMenu = menuBar()->addMenu("View");
    m_toggleSlideMarkupAction = viewMenu->addAction("Show Markup");
    m_toggleSlideMarkupAction->setCheckable(true);
    m_toggleSlideMarkupAction->setChecked(true);
    connect(m_toggleSlideMarkupAction, &QAction::toggled, this, [this](bool checked) {
        m_showSlideMarkup = checked;
        if (m_imageLabel) m_imageLabel->update();
    });

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
        updateCenterStatus();
    });
    connect(m_scrollArea->verticalScrollBar(), &QScrollBar::valueChanged, this, [this]() {
        updateOverviewViewport();
        updateCenterStatus();
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
        "background: transparent;"
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
    reloadSlideAnnotations();
    if (changed) {
        rebuildOverviewMap();
    }
    updateCenterStatus();
}

std::filesystem::path MainWindow::annotationPathForSlide() const {
    if (m_slidePath.empty()) return {};
    auto out = m_slidePath;
    out.replace_extension(".json");
    return out;
}

std::filesystem::path MainWindow::xmlPathForSlide() const {
    if (m_slidePath.empty()) return {};
    auto out = m_slidePath;
    out.replace_extension(".xml");
    return out;
}

void MainWindow::reloadSlideAnnotations() {
    m_slideAnnotations.clear();
    if (m_slidePath.empty()) {
        if (m_imageLabel) m_imageLabel->update();
        return;
    }

    QString err;
    const auto jsonPath = annotationPathForSlide();
    if (!jsonPath.empty() && std::filesystem::exists(jsonPath)) {
        if (!annotation_io::loadJson(jsonPath, m_slideAnnotations, &err)) {
            m_slideAnnotations.clear();
            updateStatusBar(QString("Failed to load JSON markup: %1").arg(err));
        }
    }

    const auto xmlPath = xmlPathForSlide();
    if (!xmlPath.empty() && std::filesystem::exists(xmlPath)) {
        err.clear();
        const int added = annotation_io::mergeFromAsapXml(xmlPath, m_slideAnnotations, &err);
        if (added < 0) {
            updateStatusBar(QString("Failed to parse XML markup: %1").arg(err));
        }
    }

    syncOverviewRectOverlays();

    if (m_imageLabel) m_imageLabel->update();
}

void MainWindow::syncOverviewRectOverlays() {
    if (!m_overviewMap) return;

    std::vector<QRectF> rects;
    rects.reserve(m_slideAnnotations.size());
    for (const auto& a : m_slideAnnotations) {
        if (a.isRect && a.bboxLevel0.isValid()) {
            rects.push_back(a.bboxLevel0);
        }
    }
    m_overviewMap->setRectOverlaysLevel0(rects);
}

void MainWindow::drawSlideMarkup(QPainter& painter, const QRect& viewRect, double downsample) const {
    if (!m_showSlideMarkup) return;
    if (m_slideAnnotations.empty()) return;
    if (!m_imageLabel) return;
    if (downsample <= 0.0) downsample = 1.0;

    painter.save();
    painter.setClipRect(viewRect, Qt::IntersectClip);
    annotation_overlay::drawAnnotations(
        painter,
        m_slideAnnotations,
        downsample,
        QPointF(0.0, 0.0),
        QRectF(
            0.0,
            0.0,
            static_cast<double>(m_imageLabel->width()),
            static_cast<double>(m_imageLabel->height())
        )
    );
    painter.restore();
}

bool MainWindow::savePreviewAnnotations(const QRect& previewRectLevel0,
                                        const std::vector<Annotation>& regions,
                                        QString* error,
                                        int* addedCount) {
    if (addedCount) *addedCount = 0;
    if (!m_slideReader.isOpen() || m_slidePath.empty()) {
        if (error) *error = "No slide is open";
        return false;
    }

    QRectF rect0 = QRectF(previewRectLevel0).normalized();
    if (rect0.width() <= 1.0 || rect0.height() <= 1.0) {
        if (error) *error = "Selected rect is too small";
        return false;
    }

    for (const auto& a : m_slideAnnotations) {
        if (!a.isRect) continue;
        if (hasPositiveIntersection(a.bboxLevel0.normalized(), rect0)) {
            if (error) *error = "Selected rect intersects an existing rect";
            return false;
        }
    }

    std::vector<Annotation> merged = m_slideAnnotations;
    int maxRectN = 0;
    for (const auto& a : merged) {
        maxRectN = std::max(maxRectN, rectNumber(a.label));
    }
    const int nextRectN = maxRectN + 1;

    Annotation rectAnno;
    rectAnno.label = QString("rect %1").arg(nextRectN);
    rectAnno.isRect = true;
    rectAnno.bboxLevel0 = rect0;
    const double x1 = rect0.left();
    const double y1 = rect0.top();
    const double x2 = rect0.right();
    const double y2 = rect0.bottom();
    rectAnno.pointsLevel0 = {
        QPointF(x1, y1),
        QPointF(x2, y1),
        QPointF(x2, y2),
        QPointF(x1, y2),
        QPointF(x1, y1),
    };
    merged.push_back(rectAnno);

    int regionAdded = 0;
    for (auto r : regions) {
        r.isRect = false;
        if (r.pointsLevel0.size() >= 3 &&
            QLineF(r.pointsLevel0.front(), r.pointsLevel0.back()).length() > 1e-6) {
            r.pointsLevel0.push_back(r.pointsLevel0.front());
        }
        if (!r.bboxLevel0.isValid()) {
            r.bboxLevel0 = computeBbox(r.pointsLevel0);
        }
        if (!r.bboxLevel0.isValid()) continue;
        merged.push_back(std::move(r));
        ++regionAdded;
    }

    const auto jsonPath = annotationPathForSlide();
    if (jsonPath.empty()) {
        if (error) *error = "Cannot resolve slide JSON path";
        return false;
    }

    QString saveErr;
    if (!annotation_io::saveJson(jsonPath, merged, &saveErr)) {
        if (error) *error = saveErr;
        return false;
    }

    m_slideAnnotations = std::move(merged);
    syncOverviewRectOverlays();
    if (m_imageLabel) m_imageLabel->update();

    if (addedCount) *addedCount = regionAdded + 1;
    updateStatusBar(QString("Saved: rect %1 + %2 region(s)").arg(nextRectN).arg(regionAdded));
    return true;
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
    updateCenterStatus();
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
    syncOverviewRectOverlays();
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
    updateCenterStatus();
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
    updateCenterStatus();
}

void MainWindow::updateStatusBar(const QString& message) {
    m_statusMessage = message;
    updateCenterStatus();
}

void MainWindow::updateCenterStatus() {
    if (!m_statusLabel) return;

    if (!m_scrollArea || !m_slideReader.isOpen() || m_scaleFactor <= 0.0) {
        m_statusLabel->setText(m_statusMessage.isEmpty() ? QStringLiteral("Ready") : m_statusMessage);
        return;
    }

    const QSize vp = m_scrollArea->viewport()->size();
    const double cxLevel = m_scrollArea->horizontalScrollBar()->value() + vp.width() / 2.0;
    const double cyLevel = m_scrollArea->verticalScrollBar()->value() + vp.height() / 2.0;
    const double cx0 = cxLevel * m_scaleFactor;
    const double cy0 = cyLevel * m_scaleFactor;

    const QString centerText = QString("Center L0: (%1, %2) | Level %3")
                                   .arg(static_cast<int>(std::llround(cx0)))
                                   .arg(static_cast<int>(std::llround(cy0)))
                                   .arg(m_currentLevel);
    if (m_statusMessage.isEmpty()) {
        m_statusLabel->setText(centerText);
    } else {
        m_statusLabel->setText(QString("%1 | %2").arg(m_statusMessage, centerText));
    }
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
    PreviewWindow* preview = new PreviewWindow(
        this,
        pixmap,
        region,
        m_slideAnnotations,
        QRect(xi, yi, wi, hi)
    );
    preview->setAttribute(Qt::WA_DeleteOnClose);
    preview->show();
}

} // namespace cytologick
