#include "window.h"
#include "annotation_io.h"
#include "menuwindow.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QDockWidget>
#include <QPainter>
#include <QPainterPath>
#include <QMouseEvent>
#include <QMessageBox>
#include <QTimer>
#include <QRegularExpression>
#include <QScrollBar>
#include <QEvent>
#include <QLineF>

#include <algorithm>
#include <cmath>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

namespace cytologick {

namespace {

double pointToSegmentDistance(const QPointF& p, const QPointF& a, const QPointF& b) {
    const double abx = b.x() - a.x();
    const double aby = b.y() - a.y();
    const double apx = p.x() - a.x();
    const double apy = p.y() - a.y();
    const double ab2 = abx * abx + aby * aby;
    if (ab2 <= 1e-12) {
        const double dx = p.x() - a.x();
        const double dy = p.y() - a.y();
        return std::sqrt(dx * dx + dy * dy);
    }
    double t = (apx * abx + apy * aby) / ab2;
    t = std::max(0.0, std::min(1.0, t));
    const double cx = a.x() + t * abx;
    const double cy = a.y() + t * aby;
    const double dx = p.x() - cx;
    const double dy = p.y() - cy;
    return std::sqrt(dx * dx + dy * dy);
}

}  // namespace

// -----------------------------------------------------------------------------
// OverviewMapWidget
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// AnnotatorImageLabel
// -----------------------------------------------------------------------------

AnnotatorImageLabel::AnnotatorImageLabel(QWidget* parent)
    : QLabel(parent)
{
    setMouseTracking(true);
}

void AnnotatorImageLabel::setSlideImage(const QPixmap& pixmap) {
    m_slideImage = pixmap;
    m_useVirtualCanvas = false;
    m_cachedRegionImage = QImage();
    m_cachedRegionRect = QRect();
    m_regionProvider = {};
    resize(pixmap.size());
    update();
}

void AnnotatorImageLabel::setVirtualCanvasSize(const QSize& size) {
    m_useVirtualCanvas = true;
    m_slideImage = QPixmap();
    m_cachedRegionImage = QImage();
    m_cachedRegionRect = QRect();
    resize(size);
    update();
}

void AnnotatorImageLabel::setRegionProvider(std::function<QImage(const QRect& levelRect)> provider) {
    m_regionProvider = std::move(provider);
    invalidateRegionCache();
}

void AnnotatorImageLabel::invalidateRegionCache() {
    m_cachedRegionImage = QImage();
    m_cachedRegionRect = QRect();
    update();
}

void AnnotatorImageLabel::setTool(Tool tool) {
    m_tool = tool;
    m_dragging = false;
    update();
}

void AnnotatorImageLabel::setPreviewPolygonEnabled(bool enabled) {
    m_previewPolygon = enabled;
    update();
}

void AnnotatorImageLabel::setAnnotations(const std::vector<Annotation>* annos) {
    m_annotations = annos;
    update();
}

void AnnotatorImageLabel::setSelectedAnnotationIndex(int index) {
    m_selectedAnnotation = index;
    m_draggingVertex = false;
    m_dragVertexIndex = -1;
    m_dragAnnotationIndex = -1;
    update();
}

void AnnotatorImageLabel::setCurrentPolygon(const std::vector<QPointF>* pointsLevel0, double downsample) {
    m_currentPolyLevel0.clear();
    if (pointsLevel0) {
        m_currentPolyLevel0 = *pointsLevel0;
    }
    m_downsample = downsample;
    update();
}

void AnnotatorImageLabel::setMapping(double downsample, int levelIndex) {
    m_downsample = (downsample <= 0.0) ? 1.0 : downsample;
    m_levelIndex = levelIndex;
}

QPointF AnnotatorImageLabel::viewToLevel0(const QPoint& viewPos) const {
    // viewPos is in pixels of the currently rendered level image.
    const double x0 = static_cast<double>(viewPos.x()) * m_downsample;
    const double y0 = static_cast<double>(viewPos.y()) * m_downsample;
    return QPointF(x0, y0);
}

QRect AnnotatorImageLabel::computeCacheRect(const QRect& targetRect) const {
    constexpr int kMargin = 256;
    constexpr int kMaxSide = 4096;

    QRect target = targetRect.intersected(rect());
    if (target.isEmpty()) return {};

    const int desiredW = std::min({kMaxSide, width(), target.width() + 2 * kMargin});
    const int desiredH = std::min({kMaxSide, height(), target.height() + 2 * kMargin});

    const QPoint c = target.center();
    int left = c.x() - desiredW / 2;
    int top = c.y() - desiredH / 2;

    left = std::max(0, std::min(left, width() - desiredW));
    top = std::max(0, std::min(top, height() - desiredH));

    return QRect(left, top, desiredW, desiredH);
}

int AnnotatorImageLabel::hitTestVertex(const QPoint& viewPos, int* outVertexIndex) const {
    if (outVertexIndex) *outVertexIndex = -1;
    if (!m_annotations) return -1;

    const double r = 8.0;
    const auto testAnnotation = [&](int annoIndex) -> bool {
        if (annoIndex < 0 || annoIndex >= static_cast<int>(m_annotations->size())) return false;
        const auto& a = (*m_annotations)[annoIndex];
        if (a.isRect || a.pointsLevel0.empty()) return false;
        for (int i = 0; i < static_cast<int>(a.pointsLevel0.size()); ++i) {
            const QPointF pv(a.pointsLevel0[i].x() / m_downsample, a.pointsLevel0[i].y() / m_downsample);
            const double dx = pv.x() - viewPos.x();
            const double dy = pv.y() - viewPos.y();
            if (dx * dx + dy * dy <= r * r) {
                if (outVertexIndex) *outVertexIndex = i;
                return true;
            }
        }
        return false;
    };

    if (testAnnotation(m_selectedAnnotation)) {
        return m_selectedAnnotation;
    }
    for (int i = 0; i < static_cast<int>(m_annotations->size()); ++i) {
        if (i == m_selectedAnnotation) continue;
        if (testAnnotation(i)) {
            return i;
        }
    }
    return -1;
}

int AnnotatorImageLabel::hitTestAnnotation(const QPoint& viewPos) const {
    if (!m_annotations) return -1;
    const QPointF p(viewPos);

    for (int i = static_cast<int>(m_annotations->size()) - 1; i >= 0; --i) {
        const auto& a = (*m_annotations)[i];
        if (a.isRect) {
            QRectF vbox(
                a.bboxLevel0.left() / m_downsample,
                a.bboxLevel0.top() / m_downsample,
                a.bboxLevel0.width() / m_downsample,
                a.bboxLevel0.height() / m_downsample
            );
            if (vbox.adjusted(-4, -4, 4, 4).contains(p)) {
                return i;
            }
            continue;
        }

        if (a.pointsLevel0.size() < 3) continue;
        QPolygonF poly;
        for (const auto& p0 : a.pointsLevel0) {
            poly << QPointF(p0.x() / m_downsample, p0.y() / m_downsample);
        }

        QPainterPath path;
        path.addPolygon(poly);
        if (path.contains(p)) {
            return i;
        }

        for (int k = 1; k < poly.size(); ++k) {
            if (pointToSegmentDistance(p, poly[k - 1], poly[k]) <= 5.0) {
                return i;
            }
        }
    }
    return -1;
}

QColor AnnotatorImageLabel::classColor(const QString& label, bool selected) const {
    QString key = label.trimmed().toLower();
    if (key.startsWith("rect")) {
        key = "rect";
    }

    QColor c;
    if (key == "rect") {
        c = QColor(0, 210, 255);
    } else {
        const uint h = qHash(key) % 360;
        c = QColor::fromHsv(static_cast<int>(h), 190, 255);
    }
    c.setAlpha(selected ? 240 : 210);
    return c;
}

void AnnotatorImageLabel::paintEvent(QPaintEvent* event) {
    QPainter painter(this);
    QRect target = rect();
    if (event) {
        target = event->rect().intersected(rect());
    }
    if (target.isEmpty()) {
        return;
    }

    if (m_useVirtualCanvas && m_regionProvider) {
        if (!m_cachedRegionRect.contains(target) || m_cachedRegionImage.isNull()) {
            const QRect requestRect = computeCacheRect(target);
            if (!requestRect.isEmpty()) {
                m_cachedRegionImage = m_regionProvider(requestRect);
                m_cachedRegionRect = requestRect;
            }
        }

        if (!m_cachedRegionImage.isNull() && m_cachedRegionRect.contains(target)) {
            const QRect srcRect = target.translated(-m_cachedRegionRect.topLeft());
            painter.drawImage(target.topLeft(), m_cachedRegionImage, srcRect);
        } else {
            painter.fillRect(target, Qt::black);
        }
    } else if (!m_slideImage.isNull()) {
        painter.drawPixmap(target, m_slideImage, target);
    } else {
        painter.fillRect(target, Qt::black);
    }

    // Draw existing annotations.
    if (m_annotations) {
        painter.setRenderHint(QPainter::Antialiasing, true);

        for (int i = 0; i < static_cast<int>(m_annotations->size()); ++i) {
            const auto& a = (*m_annotations)[i];
            const bool selected = (i == m_selectedAnnotation);
            const QColor color = classColor(a.label, selected);
            QPen pen(color, selected ? 3 : 2);
            painter.setPen(pen);
            painter.setBrush(Qt::NoBrush);

            QRectF vbox(
                a.bboxLevel0.left() / m_downsample,
                a.bboxLevel0.top() / m_downsample,
                a.bboxLevel0.width() / m_downsample,
                a.bboxLevel0.height() / m_downsample
            );
            if (a.isRect) {
                painter.drawRect(vbox);
            } else if (!a.pointsLevel0.empty()) {
                QPolygonF poly;
                for (const auto& p0 : a.pointsLevel0) {
                    poly << QPointF(p0.x() / m_downsample, p0.y() / m_downsample);
                }
                painter.drawPolyline(poly);

                if (selected) {
                    painter.setBrush(QBrush(QColor(255, 255, 255, 230)));
                    for (const auto& pv : poly) {
                        painter.drawEllipse(pv, 4, 4);
                    }
                }
            }

            // Label
            painter.setPen(QPen(QColor(255, 255, 255, 220), 1));
            painter.drawText(vbox.topLeft() + QPointF(4, 14), a.label);
        }
    }

    // Draw selection rectangle while dragging.
    if (m_tool == Tool::Rect && m_dragging) {
        painter.setPen(QPen(Qt::black, 2, Qt::SolidLine));
        painter.setBrush(QBrush(Qt::green, Qt::DiagCrossPattern));
        QRect selectionRect(m_dragStart, m_dragEnd);
        painter.drawRect(selectionRect.normalized());
    }

    // Draw current polygon (preview).
    if (m_tool == Tool::Polygon && m_previewPolygon && !m_currentPolyLevel0.empty()) {
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.setPen(QPen(QColor(255, 255, 0, 220), 2));
        QPolygonF poly;
        for (const auto& p0 : m_currentPolyLevel0) {
            poly << QPointF(p0.x() / m_downsample, p0.y() / m_downsample);
        }
        painter.drawPolyline(poly);
        for (const auto& pv : poly) {
            painter.setBrush(QBrush(QColor(255, 255, 0, 180)));
            painter.drawEllipse(pv, 3, 3);
        }
    }
}

void AnnotatorImageLabel::mousePressEvent(QMouseEvent* event) {
    if (!event) return;

    const QPoint pos = event->pos();
    emit mouseMovedLevel0(viewToLevel0(pos));

    if (m_tool == Tool::Select && event->button() == Qt::LeftButton) {
        int vertexIndex = -1;
        int annoIdx = hitTestVertex(pos, &vertexIndex);
        if (annoIdx >= 0 && vertexIndex >= 0) {
            m_selectedAnnotation = annoIdx;
            m_draggingVertex = true;
            m_dragAnnotationIndex = annoIdx;
            m_dragVertexIndex = vertexIndex;
            emit annotationSelected(annoIdx);
            update();
        } else {
            annoIdx = hitTestAnnotation(pos);
            m_selectedAnnotation = annoIdx;
            m_draggingVertex = false;
            m_dragAnnotationIndex = -1;
            m_dragVertexIndex = -1;
            emit annotationSelected(annoIdx);
            update();
        }
    } else if (m_tool == Tool::Rect) {
        if (event->button() == Qt::LeftButton) {
            m_dragging = true;
            m_dragStart = pos;
            m_dragEnd = pos;
            update();
        }
    } else if (m_tool == Tool::Polygon) {
        if (event->button() == Qt::LeftButton) {
            m_currentPolyLevel0.push_back(viewToLevel0(pos));
            update();
        } else if (event->button() == Qt::RightButton) {
            // Right click completes polygon (if it has at least 3 points).
            if (m_currentPolyLevel0.size() >= 3) {
                emit polygonCompletedLevel0(m_currentPolyLevel0);
            }
        }
    }

    QLabel::mousePressEvent(event);
}

void AnnotatorImageLabel::mouseMoveEvent(QMouseEvent* event) {
    if (!event) return;
    const QPoint pos = event->pos();
    emit mouseMovedLevel0(viewToLevel0(pos));

    if (m_tool == Tool::Select && m_draggingVertex && m_dragAnnotationIndex >= 0 && m_dragVertexIndex >= 0) {
        emit polygonVertexMoved(m_dragAnnotationIndex, m_dragVertexIndex, viewToLevel0(pos));
    }

    if (m_tool == Tool::Rect && m_dragging) {
        m_dragEnd = pos;
        update();
    }

    QLabel::mouseMoveEvent(event);
}

void AnnotatorImageLabel::mouseReleaseEvent(QMouseEvent* event) {
    if (!event) return;

    if (m_tool == Tool::Select && event->button() == Qt::LeftButton) {
        m_draggingVertex = false;
        m_dragAnnotationIndex = -1;
        m_dragVertexIndex = -1;
    }

    if (m_tool == Tool::Rect && m_dragging && event->button() == Qt::LeftButton) {
        m_dragEnd = event->pos();
        m_dragging = false;

        QRect r(m_dragStart, m_dragEnd);
        r = r.normalized();
        if (r.width() > 1 && r.height() > 1) {
            const QPointF tl0 = viewToLevel0(r.topLeft());
            const QPointF br0 = viewToLevel0(r.bottomRight());
            QRectF rect0(QPointF(std::min(tl0.x(), br0.x()), std::min(tl0.y(), br0.y())),
                         QPointF(std::max(tl0.x(), br0.x()), std::max(tl0.y(), br0.y())));
            emit rectDrawnLevel0(rect0);
        }
        update();
    }

    QLabel::mouseReleaseEvent(event);
}

void AnnotatorImageLabel::mouseDoubleClickEvent(QMouseEvent* event) {
    if (!event) return;
    if (m_tool == Tool::Polygon && event->button() == Qt::LeftButton) {
        if (m_currentPolyLevel0.size() >= 3) {
            emit polygonCompletedLevel0(m_currentPolyLevel0);
        }
    }
    QLabel::mouseDoubleClickEvent(event);
}

// -----------------------------------------------------------------------------
// AnnotatorWindow
// -----------------------------------------------------------------------------

AnnotatorWindow::AnnotatorWindow(QWidget* parent)
    : QMainWindow(parent)
{
    m_config = Config::load();
    setupUi();
    rebuildLabelList();

    // Opening the menu triggers a potentially expensive recursive slide scan.
    // Defer it until after the window is shown and the event loop is running,
    // otherwise the app looks like it "doesn't start".
    QTimer::singleShot(0, this, &AnnotatorWindow::showSlideMenu);
}

AnnotatorWindow::~AnnotatorWindow() = default;

void AnnotatorWindow::onSlideOpened(const fs::path& path) {
    m_slidePath = path;
    clearAnnotations();
    rebuildLabelList();
    reloadAnnotationsFromDisk();
    m_selectedAnnotation = -1;
    m_levelLoaded = false;
    m_imageLabel->setSelectedAnnotationIndex(-1);
    rebuildOverviewMap();

    updateStatus(QString("Slide selected: %1 | loaded %2 annotation(s)")
        .arg(QString::fromStdString(m_slidePath.filename().string()))
        .arg(m_annotations.size()));
    m_imageLabel->update();
}

void AnnotatorWindow::setupUi() {
    setWindowTitle("Cytologick Annotator");
    setGeometry(80, 80, 1200, 800);

    QWidget* central = new QWidget(this);
    QVBoxLayout* mainLayout = new QVBoxLayout(central);

    m_scrollArea = new QScrollArea(this);
    m_scrollArea->setWidgetResizable(false);
    m_scrollArea->setAlignment(Qt::AlignCenter);
    m_scrollArea->viewport()->installEventFilter(this);

    m_imageLabel = new AnnotatorImageLabel(this);
    m_imageLabel->setAnnotations(&m_annotations);
    m_scrollArea->setWidget(m_imageLabel);

    m_statusLabel = new QLabel("Open a slide to start annotating");
    m_statusLabel->setStyleSheet("background-color: black; color: white; padding: 5px;");
    m_statusLabel->setFixedHeight(30);

    mainLayout->addWidget(m_scrollArea);
    mainLayout->addWidget(m_statusLabel);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);
    setCentralWidget(central);

    // Right-side tools dock.
    QDockWidget* dock = new QDockWidget("Tools", this);
    dock->setAllowedAreas(Qt::RightDockWidgetArea);
    QWidget* dockW = new QWidget(dock);
    QVBoxLayout* dockLayout = new QVBoxLayout(dockW);

    m_toolRect = new QToolButton(dockW);
    m_toolRect->setText("Rect");
    m_toolRect->setCheckable(true);
    m_toolRect->setChecked(true);
    m_toolPoly = new QToolButton(dockW);
    m_toolPoly->setText("Polygon");
    m_toolPoly->setCheckable(true);
    m_toolSelect = new QToolButton(dockW);
    m_toolSelect->setText("Select");
    m_toolSelect->setCheckable(true);

    connect(m_toolRect, &QToolButton::clicked, this, &AnnotatorWindow::onToolRect);
    connect(m_toolPoly, &QToolButton::clicked, this, &AnnotatorWindow::onToolPolygon);
    connect(m_toolSelect, &QToolButton::clicked, this, &AnnotatorWindow::onToolSelect);

    dockLayout->addWidget(m_toolRect);
    dockLayout->addWidget(m_toolPoly);
    dockLayout->addWidget(m_toolSelect);

    dockLayout->addSpacing(10);

    m_labelCombo = new QComboBox(dockW);
    dockLayout->addWidget(new QLabel("Label:", dockW));
    dockLayout->addWidget(m_labelCombo);

    m_btnNewRect = new QPushButton("New rect N", dockW);
    m_btnSave = new QPushButton("Save JSON", dockW);
    m_btnDelete = new QPushButton("Delete selected", dockW);

    connect(m_btnNewRect, &QPushButton::clicked, this, &AnnotatorWindow::onNewRect);
    connect(m_btnSave, &QPushButton::clicked, this, &AnnotatorWindow::onSaveJson);
    connect(m_btnDelete, &QPushButton::clicked, this, &AnnotatorWindow::onDeleteSelected);

    dockLayout->addWidget(m_btnNewRect);
    dockLayout->addWidget(m_btnSave);
    dockLayout->addWidget(m_btnDelete);

    dockLayout->addSpacing(10);

    m_annotationList = new QListWidget(dockW);
    dockLayout->addWidget(new QLabel("Annotations:", dockW));
    dockLayout->addWidget(m_annotationList, 1);
    connect(m_annotationList, &QListWidget::itemSelectionChanged, this, &AnnotatorWindow::onAnnotationSelectionChanged);
    connect(m_annotationList, &QListWidget::itemDoubleClicked, this, &AnnotatorWindow::onAnnotationItemDoubleClicked);

    dockLayout->addSpacing(8);
    dockLayout->addWidget(new QLabel("Navigator:", dockW));
    m_overviewMap = new OverviewMapWidget(dockW);
    dockLayout->addWidget(m_overviewMap);
    connect(m_overviewMap, &OverviewMapWidget::jumpRequestedLevel0, this, &AnnotatorWindow::onMapJumpRequested);

    dockW->setLayout(dockLayout);
    dock->setWidget(dockW);
    addDockWidget(Qt::RightDockWidgetArea, dock);

    // Image label signals.
    connect(m_imageLabel, &AnnotatorImageLabel::rectDrawnLevel0, this, &AnnotatorWindow::onRectDrawn);
    connect(m_imageLabel, &AnnotatorImageLabel::polygonCompletedLevel0, this, &AnnotatorWindow::onPolygonCompleted);
    connect(m_imageLabel, &AnnotatorImageLabel::mouseMovedLevel0, this, &AnnotatorWindow::onMouseMoved);
    connect(m_imageLabel, &AnnotatorImageLabel::annotationSelected, this, &AnnotatorWindow::onImageAnnotationSelected);
    connect(m_imageLabel, &AnnotatorImageLabel::polygonVertexMoved, this, &AnnotatorWindow::onPolygonVertexMoved);

    connect(m_scrollArea->horizontalScrollBar(), &QScrollBar::valueChanged, this, [this]() {
        updateOverviewViewport();
    });
    connect(m_scrollArea->verticalScrollBar(), &QScrollBar::valueChanged, this, [this]() {
        updateOverviewViewport();
    });
}

void AnnotatorWindow::rebuildLabelList() {
    m_labelCombo->clear();

    // Rect labels are auto-generated; include a few common labels for polygons.
    // Also include labels from config (neural_network.labels keys) if present.
    std::vector<std::string> labels = m_config.annotationLabels;
    if (labels.empty()) {
        labels = {"HSIL", "LSIL", "ASCUS", "ASCH", "Group HSIL", "Group atypical", "Atypical", "Atypical naked"};
    }

    for (const auto& l : labels) {
        m_labelCombo->addItem(QString::fromStdString(l));
    }
}

void AnnotatorWindow::clearAnnotations() {
    m_annotations.clear();
    if (m_annotationList) {
        m_annotationList->clear();
    }
    m_rectCounter = 0;
    m_selectedAnnotation = -1;
    if (m_imageLabel) {
        m_imageLabel->setSelectedAnnotationIndex(-1);
    }
}

void AnnotatorWindow::reloadAnnotationsFromDisk() {
    if (m_slidePath.empty()) return;

    const fs::path jsonPath = annotationPathForSlide();
    if (!jsonPath.empty() && fs::exists(jsonPath)) {
        QString err;
        std::vector<Annotation> loaded;
        if (!annotation_io::loadJson(jsonPath, loaded, &err)) {
            QMessageBox::warning(this, "Annotations",
                QString("Failed to load annotations:\n%1\n\n%2")
                    .arg(QString::fromStdString(jsonPath.string()))
                    .arg(err));
        } else {
            m_annotations = std::move(loaded);
        }
    }

    const fs::path xmlPath = xmlPathForSlide();
    if (!xmlPath.empty() && fs::exists(xmlPath)) {
        QString err;
        const int added = annotation_io::mergeFromAsapXml(xmlPath, m_annotations, &err);
        if (added < 0) {
            QMessageBox::warning(this, "XML import",
                QString("Failed to parse XML annotations:\n%1\n\n%2")
                    .arg(QString::fromStdString(xmlPath.string()))
                    .arg(err));
        }
        if (added > 0 && !jsonPath.empty()) {
            if (!annotation_io::saveJson(jsonPath, m_annotations, &err)) {
                QMessageBox::warning(this, "Annotations",
                    QString("Imported %1 item(s) from XML but failed to save JSON:\n%2")
                        .arg(added).arg(err));
            }
        }
    }

    syncAnnotationUiFromData();
}

void AnnotatorWindow::ensureLabelPresent(const QString& label) {
    if (!m_labelCombo) return;
    const QString trimmed = label.trimmed();
    if (trimmed.isEmpty()) return;
    if (m_labelCombo->findText(trimmed) >= 0) return;
    m_labelCombo->addItem(trimmed);
}

void AnnotatorWindow::addAnnotationToUi(const Annotation& a) {
    if (!m_annotationList) return;
    m_annotationList->addItem(a.label);
}

void AnnotatorWindow::syncAnnotationUiFromData() {
    if (m_annotationList) {
        m_annotationList->clear();
    }
    for (const auto& a : m_annotations) {
        ensureLabelPresent(a.label);
        addAnnotationToUi(a);
    }
    recomputeRectCounter();
    if (m_imageLabel) {
        m_imageLabel->update();
    }
}

void AnnotatorWindow::recomputeRectCounter() {
    int maxN = 0;
    QRegularExpression re("^\\s*rect\\s*(\\d+)\\s*$", QRegularExpression::CaseInsensitiveOption);
    for (const auto& a : m_annotations) {
        auto m = re.match(a.label);
        if (!m.hasMatch()) continue;
        bool ok = false;
        int n = m.captured(1).toInt(&ok);
        if (ok) maxN = std::max(maxN, n);
    }
    m_rectCounter = maxN;
}

void AnnotatorWindow::showSlideMenu() {
    m_menuWindow = std::make_unique<AnnotatorMenuWindow>(this);
    QPoint mainPos = pos();
    m_menuWindow->move(mainPos.x() + width(), mainPos.y());
    m_menuWindow->show();
}

void AnnotatorWindow::loadSlideLevelPreview(int levelIndex) {
    if (!m_slideReader.isOpen()) return;
    const double oldDownsample = m_currentDownsample;
    QPointF oldCenterLevel0;
    bool hasOldCenter = false;
    if (m_levelLoaded && m_imageLabel && m_scrollArea && !m_slidePath.empty() && oldDownsample > 0.0) {
        const QSize vp = m_scrollArea->viewport()->size();
        if (vp.width() > 0 && vp.height() > 0) {
            const double cxLevel = m_scrollArea->horizontalScrollBar()->value() + vp.width() / 2.0;
            const double cyLevel = m_scrollArea->verticalScrollBar()->value() + vp.height() / 2.0;
            oldCenterLevel0 = QPointF(cxLevel * oldDownsample, cyLevel * oldDownsample);
            hasOldCenter = true;
        }
    }

    m_currentLevel = levelIndex;
    m_currentDownsample = m_slideReader.getLevelDownsample(levelIndex);

    auto [w, h] = m_slideReader.getLevelDimensions(levelIndex);
    if (w <= 0 || h <= 0) return;

    // Use viewport-based rendering: only visible region is read from OpenSlide.
    // This avoids full-frame allocations for large levels (0..2) that freeze the app.
    m_imageLabel->setVirtualCanvasSize(QSize(static_cast<int>(w), static_cast<int>(h)));
    m_imageLabel->setRegionProvider([this, levelIndex](const QRect& levelRect) -> QImage {
        if (levelIndex != m_currentLevel) return QImage();
        return readLevelRegionImage(levelRect);
    });
    m_imageLabel->setMapping(m_currentDownsample, levelIndex);
    m_imageLabel->setCurrentPolygon(nullptr, m_currentDownsample);
    m_imageLabel->invalidateRegionCache();

    if (hasOldCenter) {
        QTimer::singleShot(0, this, [this, oldCenterLevel0]() {
            centerViewOnLevel0(oldCenterLevel0);
        });
    } else {
        QTimer::singleShot(0, this, [this]() {
            updateOverviewViewport();
        });
    }

    updateStatus(QString("Slide: %1 | level=%2 | downsample=%3 | mode=viewport")
        .arg(QString::fromStdString(m_slidePath.filename().string()))
        .arg(levelIndex)
        .arg(m_currentDownsample, 0, 'f', 3));
    m_levelLoaded = true;
}

QString AnnotatorWindow::currentLabel() const {
    if (!m_labelCombo) return "Atypical";
    return m_labelCombo->currentText().trimmed();
}

QString AnnotatorWindow::nextRectLabel() {
    m_rectCounter += 1;
    return QString("rect %1").arg(m_rectCounter);
}

void AnnotatorWindow::clearCurrentPolygon() {
    m_imageLabel->setCurrentPolygon(nullptr, m_currentDownsample);
}

fs::path AnnotatorWindow::annotationPathForSlide() const {
    if (m_slidePath.empty()) return {};
    fs::path out = m_slidePath;
    out.replace_extension(".json");
    return out;
}

fs::path AnnotatorWindow::xmlPathForSlide() const {
    if (m_slidePath.empty()) return {};
    fs::path out = m_slidePath;
    out.replace_extension(".xml");
    return out;
}

void AnnotatorWindow::updateStatus(const QString& msg) {
    if (m_statusLabel) m_statusLabel->setText(msg);
}

QImage AnnotatorWindow::readLevelRegionImage(const QRect& levelRect) const {
    if (!m_slideReader.isOpen()) return {};

    auto [w, h] = m_slideReader.getLevelDimensions(m_currentLevel);
    if (w <= 0 || h <= 0) return {};

    const QRect bounds(0, 0, static_cast<int>(w), static_cast<int>(h));
    const QRect clipped = levelRect.intersected(bounds);
    if (clipped.isEmpty()) return {};

    const int64_t x0 = static_cast<int64_t>(std::llround(clipped.x() * m_currentDownsample));
    const int64_t y0 = static_cast<int64_t>(std::llround(clipped.y() * m_currentDownsample));
    const int64_t rw = clipped.width();
    const int64_t rh = clipped.height();

    cv::Mat rgb = m_slideReader.readRegionRGB(x0, y0, m_currentLevel, rw, rh);
    if (rgb.empty()) return {};

    QImage img(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    return img.copy();
}

void AnnotatorWindow::rebuildOverviewMap() {
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
    m_overviewMap->setOverview(img.copy(), effectiveDownsample, QSize(static_cast<int>(l0w), static_cast<int>(l0h)));
    updateOverviewViewport();
}

void AnnotatorWindow::updateOverviewViewport() {
    if (!m_overviewMap || !m_scrollArea || !m_slideReader.isOpen()) return;
    if (m_currentDownsample <= 0.0) return;

    const QSize vp = m_scrollArea->viewport()->size();
    const QRectF vr0(
        m_scrollArea->horizontalScrollBar()->value() * m_currentDownsample,
        m_scrollArea->verticalScrollBar()->value() * m_currentDownsample,
        vp.width() * m_currentDownsample,
        vp.height() * m_currentDownsample
    );
    m_overviewMap->setViewportLevel0(vr0);
}

void AnnotatorWindow::centerViewOnLevel0(const QPointF& centerLevel0) {
    if (!m_scrollArea || m_currentDownsample <= 0.0) return;
    const QSize vp = m_scrollArea->viewport()->size();
    const double cxLevel = centerLevel0.x() / m_currentDownsample;
    const double cyLevel = centerLevel0.y() / m_currentDownsample;

    const int newH = static_cast<int>(std::llround(cxLevel - vp.width() / 2.0));
    const int newV = static_cast<int>(std::llround(cyLevel - vp.height() / 2.0));

    m_scrollArea->horizontalScrollBar()->setValue(newH);
    m_scrollArea->verticalScrollBar()->setValue(newV);
    updateOverviewViewport();
}

void AnnotatorWindow::onRectDrawn(QRectF rectLevel0) {
    if (m_slidePath.empty()) return;

    Annotation a;
    a.label = nextRectLabel();
    a.isRect = true;
    a.bboxLevel0 = rectLevel0;

    // Store rectangle points as a closed poly (optional, kept for compatibility/tools).
    const double x1 = rectLevel0.left();
    const double y1 = rectLevel0.top();
    const double x2 = rectLevel0.right();
    const double y2 = rectLevel0.bottom();
    a.pointsLevel0 = {
        QPointF(x1, y1),
        QPointF(x2, y1),
        QPointF(x2, y2),
        QPointF(x1, y2),
        QPointF(x1, y1),
    };

    m_annotations.push_back(a);
    m_annotationList->addItem(a.label);
    const int idx = static_cast<int>(m_annotations.size()) - 1;
    m_annotationList->setCurrentRow(idx);
    m_selectedAnnotation = idx;
    m_imageLabel->setSelectedAnnotationIndex(idx);
    updateStatus(QString("Added %1").arg(a.label));
    m_imageLabel->update();
}

void AnnotatorWindow::onPolygonCompleted(std::vector<QPointF> pointsLevel0) {
    if (m_slidePath.empty()) return;
    if (pointsLevel0.size() < 3) return;

    // Compute bbox.
    double minx = pointsLevel0[0].x();
    double miny = pointsLevel0[0].y();
    double maxx = pointsLevel0[0].x();
    double maxy = pointsLevel0[0].y();
    for (const auto& p : pointsLevel0) {
        minx = std::min(minx, p.x());
        miny = std::min(miny, p.y());
        maxx = std::max(maxx, p.x());
        maxy = std::max(maxy, p.y());
    }

    Annotation a;
    a.label = currentLabel();
    a.isRect = false;
    a.pointsLevel0 = std::move(pointsLevel0);
    a.bboxLevel0 = QRectF(QPointF(minx, miny), QPointF(maxx, maxy));

    m_annotations.push_back(a);
    m_annotationList->addItem(a.label);
    const int idx = static_cast<int>(m_annotations.size()) - 1;
    m_annotationList->setCurrentRow(idx);
    m_selectedAnnotation = idx;
    m_imageLabel->setSelectedAnnotationIndex(idx);
    updateStatus(QString("Added polygon: %1").arg(a.label));

    // Reset polygon tool state.
    m_imageLabel->setCurrentPolygon(nullptr, m_currentDownsample);
    m_imageLabel->update();
}

void AnnotatorWindow::onMouseMoved(QPointF posLevel0) {
    updateStatus(QString("Slide: %1 | L0 x=%2 y=%3 | level=%4 downsample=%5")
        .arg(QString::fromStdString(m_slidePath.filename().string()))
        .arg(static_cast<int>(posLevel0.x()))
        .arg(static_cast<int>(posLevel0.y()))
        .arg(m_currentLevel)
        .arg(m_currentDownsample, 0, 'f', 3));
}

void AnnotatorWindow::onToolRect() {
    m_toolRect->setChecked(true);
    m_toolPoly->setChecked(false);
    m_toolSelect->setChecked(false);
    m_imageLabel->setTool(AnnotatorImageLabel::Tool::Rect);
}

void AnnotatorWindow::onToolPolygon() {
    m_toolRect->setChecked(false);
    m_toolPoly->setChecked(true);
    m_toolSelect->setChecked(false);
    m_imageLabel->setTool(AnnotatorImageLabel::Tool::Polygon);
}

void AnnotatorWindow::onToolSelect() {
    m_toolRect->setChecked(false);
    m_toolPoly->setChecked(false);
    m_toolSelect->setChecked(true);
    m_imageLabel->setTool(AnnotatorImageLabel::Tool::Select);
}

void AnnotatorWindow::onNewRect() {
    // Convenience: switch to rect tool.
    onToolRect();
    updateStatus("Rect tool selected: drag to create rect N");
}

void AnnotatorWindow::onSaveJson() {
    if (m_slidePath.empty()) {
        QMessageBox::information(this, "Save", "No slide selected");
        return;
    }

    fs::path outPath = annotationPathForSlide();
    if (outPath.empty()) return;
    QString err;
    if (!annotation_io::saveJson(outPath, m_annotations, &err)) {
        QMessageBox::warning(this, "Save", err.isEmpty() ? "Failed to save JSON" : err);
        return;
    }

    updateStatus(QString("Saved: %1").arg(QString::fromStdString(outPath.string())));
}

void AnnotatorWindow::onDeleteSelected() {
    auto items = m_annotationList->selectedItems();
    if (items.isEmpty()) return;

    const int row = m_annotationList->row(items[0]);
    if (row < 0 || row >= static_cast<int>(m_annotations.size())) return;

    m_annotations.erase(m_annotations.begin() + row);
    delete m_annotationList->takeItem(row);
    m_selectedAnnotation = -1;
    m_imageLabel->setSelectedAnnotationIndex(-1);
    m_imageLabel->update();
    updateStatus("Deleted annotation");
}

void AnnotatorWindow::onAnnotationSelectionChanged() {
    if (!m_annotationList) return;
    const int row = m_annotationList->currentRow();
    m_selectedAnnotation = row;
    m_imageLabel->setSelectedAnnotationIndex(row);
}

void AnnotatorWindow::onAnnotationItemDoubleClicked(QListWidgetItem* item) {
    if (!m_annotationList || !item) return;
    const int row = m_annotationList->row(item);
    if (row < 0 || row >= static_cast<int>(m_annotations.size())) return;

    m_annotationList->setCurrentRow(row);
    m_selectedAnnotation = row;
    m_imageLabel->setSelectedAnnotationIndex(row);

    const QRectF& bb = m_annotations[row].bboxLevel0;
    const QPointF center = bb.center();
    centerViewOnLevel0(center);
    updateStatus(QString("Centered on: %1").arg(m_annotations[row].label));
}

void AnnotatorWindow::onImageAnnotationSelected(int index) {
    m_selectedAnnotation = index;
    if (!m_annotationList) return;
    if (index < 0 || index >= m_annotationList->count()) {
        m_annotationList->clearSelection();
        m_annotationList->setCurrentRow(-1);
        return;
    }
    m_annotationList->setCurrentRow(index);
}

void AnnotatorWindow::onPolygonVertexMoved(int annotationIndex, int vertexIndex, QPointF newPosLevel0) {
    if (annotationIndex < 0 || annotationIndex >= static_cast<int>(m_annotations.size())) return;
    Annotation& a = m_annotations[annotationIndex];
    if (a.isRect) return;
    if (vertexIndex < 0 || vertexIndex >= static_cast<int>(a.pointsLevel0.size())) return;

    a.pointsLevel0[vertexIndex] = newPosLevel0;

    const bool closed = a.pointsLevel0.size() >= 3 &&
        QLineF(a.pointsLevel0.front(), a.pointsLevel0.back()).length() <= 1e-6;
    if (closed) {
        if (vertexIndex == 0) {
            a.pointsLevel0.back() = newPosLevel0;
        } else if (vertexIndex == static_cast<int>(a.pointsLevel0.size()) - 1) {
            a.pointsLevel0.front() = newPosLevel0;
        }
    }

    double minx = a.pointsLevel0[0].x();
    double miny = a.pointsLevel0[0].y();
    double maxx = a.pointsLevel0[0].x();
    double maxy = a.pointsLevel0[0].y();
    for (const auto& p : a.pointsLevel0) {
        minx = std::min(minx, p.x());
        miny = std::min(miny, p.y());
        maxx = std::max(maxx, p.x());
        maxy = std::max(maxy, p.y());
    }
    a.bboxLevel0 = QRectF(QPointF(minx, miny), QPointF(maxx, maxy));
    m_imageLabel->update();
}

void AnnotatorWindow::onMapJumpRequested(QPointF centerLevel0) {
    centerViewOnLevel0(centerLevel0);
}

bool AnnotatorWindow::eventFilter(QObject* watched, QEvent* event) {
    if (m_scrollArea && watched == m_scrollArea->viewport() && event) {
        if (event->type() == QEvent::Resize) {
            updateOverviewViewport();
        }
    }
    return QMainWindow::eventFilter(watched, event);
}

} // namespace cytologick
