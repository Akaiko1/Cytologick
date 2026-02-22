#pragma once

#include "annotation_types.h"
#include "config.h"
#include "slidereader.h"

#include <QMainWindow>
#include <QScrollArea>
#include <QLabel>
#include <QPixmap>
#include <QComboBox>
#include <QListWidget>
#include <QToolButton>
#include <QPushButton>
#include <QSlider>
#include <QPointF>
#include <QRectF>
#include <QImage>
#include <QString>
#include <QDialog>
#include <QAction>

#include <filesystem>
#include <functional>
#include <vector>

namespace cytologick {

class AnnotatorMenuWindow;

class OverviewMapWidget : public QWidget {
    Q_OBJECT

public:
    explicit OverviewMapWidget(QWidget* parent = nullptr);

    void setOverview(const QImage& image, double downsampleToLevel0, const QSize& level0Size);
    void setViewportLevel0(const QRectF& viewportLevel0);
    void clear();

signals:
    void jumpRequestedLevel0(QPointF centerLevel0);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    QRect imageTargetRect() const;
    QPointF widgetToLevel0(const QPoint& pos) const;

    QImage m_overview;
    QSize m_level0Size;
    QRectF m_viewportLevel0;
    double m_overviewDownsample = 1.0;
    bool m_mouseDown = false;
};

class AnnotatorImageLabel : public QLabel {
    Q_OBJECT

public:
    enum class Tool {
        Rect,
        Polygon,
        Select,
    };

    explicit AnnotatorImageLabel(QWidget* parent = nullptr);

    void setSlideImage(const QPixmap& pixmap);
    void setVirtualCanvasSize(const QSize& size);
    void setRegionProvider(std::function<QImage(const QRect& levelRect)> provider);
    void invalidateRegionCache();
    void setTool(Tool tool);
    void setPreviewPolygonEnabled(bool enabled);
    void setAnnotations(const std::vector<Annotation>* annos);
    void setSelectedAnnotationIndex(int index);
    void setCurrentPolygon(const std::vector<QPointF>* pointsLevel0, double downsample);

    // Mapping parameters for view->level0 conversion.
    void setMapping(double downsample, int levelIndex);

signals:
    void rectDrawnLevel0(QRectF rectLevel0);
    void polygonCompletedLevel0(std::vector<QPointF> pointsLevel0);
    void mouseMovedLevel0(QPointF posLevel0);
    void panByPixels(QPoint delta);
    void annotationSelected(int index);
    void polygonVertexMoved(int annotationIndex, int vertexIndex, QPointF newPosLevel0);
    void annotationLabelDoubleClicked(int index);
    void annotationLabelContextRequested(int index, QPoint globalPos);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;

private:
    QPointF viewToLevel0(const QPoint& viewPos) const;
    QRect computeCacheRect(const QRect& targetRect) const;
    int hitTestVertex(const QPoint& viewPos, int* outVertexIndex) const;
    int hitTestAnnotation(const QPoint& viewPos) const;
    int hitTestAnnotationLabel(const QPoint& viewPos) const;
    QColor classColor(const QString& label, bool selected) const;

    QPixmap m_slideImage;
    bool m_useVirtualCanvas = false;
    std::function<QImage(const QRect& levelRect)> m_regionProvider;
    QImage m_cachedRegionImage;
    QRect m_cachedRegionRect;
    Tool m_tool = Tool::Select;

    bool m_dragging = false;
    QPoint m_dragStart;
    QPoint m_dragEnd;

    bool m_previewPolygon = true;
    std::vector<QPointF> m_currentPolyLevel0;

    // Current slide mapping.
    double m_downsample = 1.0;
    int m_levelIndex = 0;

    const std::vector<Annotation>* m_annotations = nullptr;
    int m_selectedAnnotation = -1;
    bool m_draggingVertex = false;
    int m_dragVertexIndex = -1;
    int m_dragAnnotationIndex = -1;
    bool m_panning = false;
    bool m_panMoved = false;
    QPoint m_panLastGlobal;
    std::vector<QRectF> m_labelChipRectsView;
};

class AnnotatorWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit AnnotatorWindow(QWidget* parent = nullptr);
    ~AnnotatorWindow() override;

    Config& getConfig() { return m_config; }
    SlideReader& getSlideReader() { return m_slideReader; }

    void onSlideOpened(const std::filesystem::path& path);
    void loadSlideLevel(int levelIndex) { loadSlideLevelPreview(levelIndex); }

private slots:
    void showSlideMenu();
    void onRectDrawn(QRectF rectLevel0);
    void onPolygonCompleted(std::vector<QPointF> pointsLevel0);
    void onMouseMoved(QPointF posLevel0);
    void onToolRect();
    void onToolPolygon();
    void onToolSelect();
    void onNewRect();
    void onLoadJson();
    void onSaveJson();
    void onSelectWorkingDirectory();
    void onDeleteSelected();
    void onOpenVertexDeformSettings();
    void onAnnotationSelectionChanged();
    void onAnnotationItemDoubleClicked(QListWidgetItem* item);
    void onImageAnnotationSelected(int index);
    void onImageAnnotationLabelDoubleClicked(int index);
    void onImageAnnotationLabelContextRequested(int index, QPoint globalPos);
    void onPolygonVertexMoved(int annotationIndex, int vertexIndex, QPointF newPosLevel0);
    void onMapJumpRequested(QPointF centerLevel0);

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    enum class SoftFalloff {
        Smooth = 0,
        Linear = 1,
        Gaussian = 2,
    };

    void setupUi();
    void rebuildLabelList();
    void clearCurrentPolygon();
    void updateStatus(const QString& msg);
    void loadSlideLevelPreview(int levelIndex);
    void clearAnnotations();
    void reloadAnnotationsFromDisk();
    void addAnnotationToUi(const Annotation& a);
    void syncAnnotationUiFromData();
    void ensureLabelPresent(const QString& label);
    void recomputeRectCounter();
    QImage readLevelRegionImage(const QRect& levelRect) const;
    int listRowToAnnotationIndex(int row) const;
    int annotationIndexToListRow(int annotationIndex) const;
    void rebuildOverviewMap();
    void updateOverviewViewport();
    void centerViewOnLevel0(const QPointF& centerLevel0);
    void centerOnAnnotation(int annotationIndex);
    double softInfluence(double normalizedDistance) const;
    void applyWorkingDirectory(const std::filesystem::path& dir, bool persist);

    QString currentLabel() const;
    QString nextRectLabel();
    std::filesystem::path annotationPathForSlide() const;
    std::filesystem::path xmlPathForSlide() const;

    // Data
    Config m_config;
    SlideReader m_slideReader;
    std::filesystem::path m_slidePath;
    int m_currentLevel = 0;
    double m_currentDownsample = 1.0;
    int m_rectCounter = 0;
    int m_selectedAnnotation = -1;
    bool m_levelLoaded = false;

    std::vector<Annotation> m_annotations;
    std::vector<int> m_annotationListOrder;

    // UI
    QScrollArea* m_scrollArea = nullptr;
    AnnotatorImageLabel* m_imageLabel = nullptr;
    QLabel* m_statusLabel = nullptr;
    OverviewMapWidget* m_overviewMap = nullptr;

    QComboBox* m_labelCombo = nullptr;
    QListWidget* m_annotationList = nullptr;
    QToolButton* m_toolRect = nullptr;
    QToolButton* m_toolPoly = nullptr;
    QToolButton* m_toolSelect = nullptr;
    QPushButton* m_btnNewRect = nullptr;
    QPushButton* m_btnDelete = nullptr;
    QAction* m_actionSelectSlide = nullptr;
    QAction* m_actionSelectWorkingDir = nullptr;
    QAction* m_actionLoadJson = nullptr;
    QAction* m_actionSaveJson = nullptr;
    QSlider* m_softRadiusSlider = nullptr;
    QLabel* m_softRadiusValueLabel = nullptr;
    QComboBox* m_softFalloffCombo = nullptr;
    int m_softRadiusViewPx = 56;
    SoftFalloff m_softFalloff = SoftFalloff::Smooth;
    QDialog* m_vertexDeformDialog = nullptr;

    std::unique_ptr<AnnotatorMenuWindow> m_menuWindow;
};

} // namespace cytologick
