#include "annotation_overlay.h"

#include <QFontMetrics>
#include <QLineF>
#include <QPainterPath>
#include <QPolygonF>

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace cytologick::annotation_overlay {

QColor classColor(const QString& label, bool selected) {
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

namespace {
QRectF boundsFromPoints(const std::vector<QPointF>& points) {
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
    return QRectF(QPointF(minx, miny), QPointF(maxx, maxy)).normalized();
}
}  // namespace

void drawAnnotations(
    QPainter& painter,
    const std::vector<Annotation>& annotations,
    double downsample,
    QPointF level0Origin,
    const QRectF& canvasRect
) {
    if (annotations.empty()) return;
    if (downsample <= 0.0) downsample = 1.0;

    painter.setRenderHint(QPainter::Antialiasing, true);

    struct LabelDraw {
        QPointF textPos;
        QRectF chip;
        QString text;
    };

    std::vector<LabelDraw> labelsToDraw;
    labelsToDraw.reserve(annotations.size());
    std::vector<QRectF> placedChips;
    placedChips.reserve(annotations.size());

    std::vector<int> drawOrder;
    drawOrder.reserve(static_cast<int>(annotations.size()));
    for (int i = 0; i < static_cast<int>(annotations.size()); ++i) {
        if (annotations[i].isRect) drawOrder.push_back(i);
    }
    for (int i = 0; i < static_cast<int>(annotations.size()); ++i) {
        if (!annotations[i].isRect) drawOrder.push_back(i);
    }

    const QFontMetrics fm(painter.font());
    const QRectF safeCanvas = canvasRect.adjusted(2.0, 2.0, -2.0, -2.0);

    for (int idx : drawOrder) {
        const auto& a = annotations[idx];
        if (a.label.trimmed().isEmpty()) continue;

        QRectF bbox0;
        if (!a.isRect && !a.pointsLevel0.empty()) {
            // Match annotator behavior: polygon geometry is authoritative.
            // Some legacy JSONs may contain stale/incorrect rect field.
            bbox0 = boundsFromPoints(a.pointsLevel0);
        } else {
            bbox0 = a.bboxLevel0.normalized();
            if ((!bbox0.isValid() || bbox0.isEmpty()) && !a.pointsLevel0.empty()) {
                bbox0 = boundsFromPoints(a.pointsLevel0);
            }
        }
        if (!bbox0.isValid() || bbox0.isEmpty()) {
            continue;
        }

        QRectF vbox(
            (bbox0.left() - level0Origin.x()) / downsample,
            (bbox0.top() - level0Origin.y()) / downsample,
            bbox0.width() / downsample,
            bbox0.height() / downsample
        );
        const bool labelCandidateVisible =
            safeCanvas.adjusted(-16.0, -16.0, 16.0, 16.0).intersects(vbox);

        const QColor base = classColor(a.label, false);
        QColor fillColor = base;
        QColor lineColor = base;
        int lineWidth = 2;
        if (a.isRect) {
            // Rects should not occlude polygon regions inside them.
            lineColor = QColor(0, 255, 255, 250);
            lineWidth = 3;
        } else {
            fillColor.setAlpha(48);
            lineColor.setAlpha(220);
        }

        painter.setPen(QPen(lineColor, lineWidth));
        painter.setBrush(a.isRect ? Qt::NoBrush : QBrush(fillColor));

        if (a.isRect) {
            painter.drawRect(vbox);
        } else if (!a.pointsLevel0.empty()) {
            QPolygonF poly;
            poly.reserve(static_cast<int>(a.pointsLevel0.size()));
            for (const auto& p0 : a.pointsLevel0) {
                poly << QPointF(
                    (p0.x() - level0Origin.x()) / downsample,
                    (p0.y() - level0Origin.y()) / downsample
                );
            }

            if (poly.size() >= 2) {
                // Match annotator visibility behavior: always draw contour for polygonal regions.
                // Fill remains for easier class perception.
                if (poly.size() >= 3) {
                    QPolygonF fillPoly = poly;
                    if (QLineF(fillPoly.first(), fillPoly.last()).length() > 1e-6) {
                        fillPoly << fillPoly.first();
                    }
                    QPainterPath path;
                    path.addPolygon(fillPoly);
                    path.closeSubpath();
                    painter.save();
                    painter.setPen(Qt::NoPen);
                    painter.drawPath(path);
                    painter.restore();
                }
                painter.setBrush(Qt::NoBrush);
                painter.drawPolyline(poly);
            }
        }

        if (!labelCandidateVisible) {
            continue;
        }

        const QRect tr = fm.boundingRect(a.label);
        if (tr.width() <= 0 || tr.height() <= 0) continue;

        const double textW = tr.width();
        const double textH = tr.height();
        const QRectF regionBox = vbox.adjusted(-2.0, -2.0, 2.0, 2.0);

        auto evalCandidate = [&](double desiredX, double desiredY, double pref) {
            const double minX = safeCanvas.left();
            const double maxX = std::max(minX, safeCanvas.right() - textW);
            const double minY = safeCanvas.top() + textH;
            const double maxY = std::max(minY, safeCanvas.bottom());
            const double x = std::clamp(desiredX, minX, maxX);
            const double y = std::clamp(desiredY, minY, maxY);
            const QRectF chip(x - 3.0, y - textH + 3.0, textW + 8.0, textH + 4.0);

            double overlapArea = 0.0;
            for (const auto& pc : placedChips) {
                const QRectF inter = chip.intersected(pc);
                if (!inter.isEmpty()) overlapArea += inter.width() * inter.height();
            }
            const QRectF regionInter = chip.intersected(regionBox);
            const double regionArea = regionInter.isEmpty() ? 0.0 : regionInter.width() * regionInter.height();
            const double clampPenalty = std::abs(x - desiredX) + std::abs(y - desiredY);
            const double score = overlapArea * 50.0 + regionArea * 20.0 + clampPenalty * 4.0 + pref;
            return std::tuple<double, QPointF, QRectF>(score, QPointF(x, y), chip);
        };

        std::vector<std::pair<QPointF, double>> requests = {
            {QPointF(vbox.left() + 4.0, vbox.top() - 8.0), 0.0},
            {QPointF(vbox.right() - textW - 4.0, vbox.top() - 8.0), 2.0},
            {QPointF(vbox.center().x() - textW / 2.0, vbox.top() - 8.0), 4.0},
            {QPointF(vbox.left() + 4.0, vbox.bottom() + textH + 3.0), 6.0},
            {QPointF(vbox.right() - textW - 4.0, vbox.bottom() + textH + 3.0), 8.0},
            {QPointF(vbox.center().x() - textW / 2.0, vbox.bottom() + textH + 3.0), 10.0},
            {QPointF(vbox.left() + 4.0, vbox.top() + textH + 2.0), 50.0},
        };

        double bestScore = std::numeric_limits<double>::infinity();
        QPointF bestTextPos;
        QRectF bestChip;
        for (const auto& [pt, pref] : requests) {
            auto [score, textPos, chip] = evalCandidate(pt.x(), pt.y(), pref);
            if (score < bestScore) {
                bestScore = score;
                bestTextPos = textPos;
                bestChip = chip;
            }
        }

        if (bestChip.isValid()) {
            placedChips.push_back(bestChip);
            labelsToDraw.push_back({bestTextPos, bestChip, a.label});
        }
    }

    for (const auto& ld : labelsToDraw) {
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(255, 255, 255, 210));
        painter.drawRoundedRect(ld.chip, 3.0, 3.0);
        painter.setPen(QPen(QColor(0, 0, 0, 235), 1));
        painter.drawText(ld.textPos, ld.text);
    }
}

QPixmap renderToPixmap(
    const QSize& size,
    const std::vector<Annotation>& annotations,
    double downsample,
    QPointF level0Origin
) {
    if (size.width() <= 0 || size.height() <= 0 || annotations.empty()) {
        return QPixmap();
    }

    QImage img(size, QImage::Format_ARGB32_Premultiplied);
    img.fill(Qt::transparent);

    QPainter painter(&img);
    drawAnnotations(
        painter,
        annotations,
        downsample,
        level0Origin,
        QRectF(0.0, 0.0, static_cast<double>(size.width()), static_cast<double>(size.height()))
    );
    painter.end();

    return QPixmap::fromImage(img);
}

}  // namespace cytologick::annotation_overlay
