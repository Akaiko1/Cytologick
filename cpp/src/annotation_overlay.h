#pragma once

#include "annotator/annotation_types.h"

#include <QPainter>
#include <QPixmap>
#include <QPointF>
#include <QRectF>
#include <QSize>

#include <vector>

namespace cytologick::annotation_overlay {

QColor classColor(const QString& label, bool selected = false);

void drawAnnotations(
    QPainter& painter,
    const std::vector<Annotation>& annotations,
    double downsample,
    QPointF level0Origin,
    const QRectF& canvasRect
);

QPixmap renderToPixmap(
    const QSize& size,
    const std::vector<Annotation>& annotations,
    double downsample,
    QPointF level0Origin = QPointF(0.0, 0.0)
);

}  // namespace cytologick::annotation_overlay

