#pragma once

#include <QPointF>
#include <QRectF>
#include <QString>

#include <vector>

namespace cytologick {

struct Annotation {
    QString label;
    std::vector<QPointF> pointsLevel0;  // level-0 pixel coordinates
    QRectF bboxLevel0;                  // [minx,miny,maxx,maxy] in level-0
    bool isRect = false;
};

}  // namespace cytologick

