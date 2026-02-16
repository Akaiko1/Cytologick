#include "annotation_io.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLineF>
#include <QSaveFile>
#include <QXmlStreamReader>

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace fs = std::filesystem;

namespace cytologick::annotation_io {

namespace {

void closePolygonIfNeeded(Annotation& a) {
    if (a.isRect) return;
    if (a.pointsLevel0.size() < 3) return;
    if (QLineF(a.pointsLevel0.front(), a.pointsLevel0.back()).length() <= 1e-6) return;
    a.pointsLevel0.push_back(a.pointsLevel0.front());
}

std::string dedupKey(const Annotation& a) {
    std::string key;
    const std::string label = a.label.trimmed().toLower().toStdString();
    key.reserve(label.size() + a.pointsLevel0.size() * 18 + 64);
    key.append(label);
    key.push_back('|');
    key.push_back(a.isRect ? '1' : '0');
    key.push_back('|');

    const auto bx1 = static_cast<long long>(std::llround(a.bboxLevel0.left()));
    const auto by1 = static_cast<long long>(std::llround(a.bboxLevel0.top()));
    const auto bx2 = static_cast<long long>(std::llround(a.bboxLevel0.right()));
    const auto by2 = static_cast<long long>(std::llround(a.bboxLevel0.bottom()));
    key.append(std::to_string(bx1));
    key.push_back(',');
    key.append(std::to_string(by1));
    key.push_back(',');
    key.append(std::to_string(bx2));
    key.push_back(',');
    key.append(std::to_string(by2));
    key.push_back('|');

    key.append(std::to_string(a.pointsLevel0.size()));
    key.push_back('|');
    for (const auto& p : a.pointsLevel0) {
        const auto x = static_cast<long long>(std::llround(p.x()));
        const auto y = static_cast<long long>(std::llround(p.y()));
        key.append(std::to_string(x));
        key.push_back(',');
        key.append(std::to_string(y));
        key.push_back(';');
    }
    return key;
}

void ensureBbox(Annotation& a) {
    if (!a.pointsLevel0.empty()) {
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
    }
}

}  // namespace

bool loadJson(const fs::path& jsonPath, std::vector<Annotation>& out, QString* error) {
    out.clear();

    QFile f(QString::fromStdString(jsonPath.string()));
    if (!f.open(QIODevice::ReadOnly)) {
        if (error) *error = "Failed to open JSON for reading";
        return false;
    }
    const QByteArray bytes = f.readAll();
    f.close();

    QJsonParseError parseErr;
    const QJsonDocument doc = QJsonDocument::fromJson(bytes, &parseErr);
    if (parseErr.error != QJsonParseError::NoError) {
        if (error) *error = parseErr.errorString();
        return false;
    }
    if (doc.isNull()) {
        return true;
    }
    if (!doc.isArray()) {
        if (error) *error = "JSON root is not an array";
        return false;
    }

    const QJsonArray arr = doc.array();
    out.reserve(arr.size());
    for (const auto& v : arr) {
        if (!v.isObject()) continue;
        const QJsonObject obj = v.toObject();

        Annotation a;
        a.label = obj.value("label").toString().trimmed();
        if (a.label.isEmpty()) continue;
        a.isRect = a.label.toLower().contains("rect");

        const QJsonValue pv = obj.value("points");
        if (pv.isArray()) {
            const QJsonArray pointsArr = pv.toArray();
            a.pointsLevel0.reserve(pointsArr.size());
            for (const auto& p : pointsArr) {
                if (!p.isArray()) continue;
                const QJsonArray pa = p.toArray();
                if (pa.size() < 2) continue;
                a.pointsLevel0.emplace_back(pa.at(0).toDouble(), pa.at(1).toDouble());
            }
        }
        closePolygonIfNeeded(a);

        bool haveRect = false;
        const QJsonValue rv = obj.value("rect");
        if (rv.isArray()) {
            const QJsonArray ra = rv.toArray();
            if (ra.size() >= 4) {
                const double minx = ra.at(0).toDouble();
                const double miny = ra.at(1).toDouble();
                const double maxx = ra.at(2).toDouble();
                const double maxy = ra.at(3).toDouble();
                a.bboxLevel0 = QRectF(QPointF(minx, miny), QPointF(maxx, maxy));
                haveRect = true;
            }
        }
        if (!haveRect) {
            ensureBbox(a);
        }

        out.push_back(std::move(a));
    }

    return true;
}

bool saveJson(const fs::path& jsonPath, const std::vector<Annotation>& annotations, QString* error) {
    if (jsonPath.empty()) {
        if (error) *error = "Output path is empty";
        return false;
    }

    QJsonArray arr;
    for (const auto& a : annotations) {
        std::vector<QPointF> pointsOut = a.pointsLevel0;
        if (!a.isRect && pointsOut.size() >= 3 &&
            QLineF(pointsOut.front(), pointsOut.back()).length() > 1e-6) {
            pointsOut.push_back(pointsOut.front());
        }

        QJsonObject obj;
        obj["label"] = a.label;

        QJsonArray points;
        for (const auto& p : pointsOut) {
            QJsonArray pt;
            pt.append(static_cast<int>(std::llround(p.x())));
            pt.append(static_cast<int>(std::llround(p.y())));
            points.append(pt);
        }
        obj["points"] = points;

        const int minx = static_cast<int>(std::llround(a.bboxLevel0.left()));
        const int miny = static_cast<int>(std::llround(a.bboxLevel0.top()));
        const int maxx = static_cast<int>(std::llround(a.bboxLevel0.right()));
        const int maxy = static_cast<int>(std::llround(a.bboxLevel0.bottom()));
        QJsonArray rect;
        rect.append(minx);
        rect.append(miny);
        rect.append(maxx);
        rect.append(maxy);
        obj["rect"] = rect;

        arr.append(obj);
    }

    QJsonDocument doc(arr);
    QSaveFile f(QString::fromStdString(jsonPath.string()));
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        if (error) *error = "Failed to open JSON for writing";
        return false;
    }
    f.write(doc.toJson(QJsonDocument::Indented));
    if (!f.commit()) {
        if (error) *error = "Failed to commit JSON file";
        return false;
    }
    return true;
}

int mergeFromAsapXml(const fs::path& xmlPath, std::vector<Annotation>& annotations, QString* error) {
    QFile f(QString::fromStdString(xmlPath.string()));
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (error) *error = "Failed to open XML for reading";
        return -1;
    }

    QXmlStreamReader xml(&f);
    struct Coord {
        int order = 0;
        QPointF p;
    };

    std::vector<Annotation> parsed;
    QString curName;
    QString curType;
    std::vector<Coord> curCoords;
    bool inAnnotation = false;
    int fallbackOrder = 0;

    auto flushAnnotation = [&]() {
        if (!inAnnotation) return;
        inAnnotation = false;
        if (curCoords.empty()) {
            curCoords.clear();
            return;
        }

        std::sort(curCoords.begin(), curCoords.end(), [](const Coord& a, const Coord& b) {
            return a.order < b.order;
        });

        Annotation a;
        a.label = curName.trimmed();
        if (a.label.isEmpty()) {
            a.label = curType.trimmed();
            if (a.label.isEmpty()) a.label = "Imported";
        }

        const QString t = curType.trimmed().toLower();
        a.isRect = (t == "rectangle") || a.label.toLower().contains("rect");

        a.pointsLevel0.reserve(curCoords.size());
        for (const auto& c : curCoords) {
            a.pointsLevel0.push_back(c.p);
        }
        closePolygonIfNeeded(a);
        ensureBbox(a);
        parsed.push_back(std::move(a));
        curCoords.clear();
    };

    while (!xml.atEnd()) {
        xml.readNext();
        if (xml.isStartElement()) {
            const auto name = xml.name();
            if (name == "Annotation") {
                flushAnnotation();
                inAnnotation = true;
                fallbackOrder = 0;
                curCoords.clear();
                const auto attrs = xml.attributes();
                curName = attrs.value("Name").toString();
                curType = attrs.value("Type").toString();
            } else if (inAnnotation && name == "Coordinate") {
                const auto attrs = xml.attributes();
                bool okX = false;
                bool okY = false;
                const double x = attrs.value("X").toString().toDouble(&okX);
                const double y = attrs.value("Y").toString().toDouble(&okY);
                if (!okX || !okY) continue;

                bool okOrder = false;
                int order = attrs.value("Order").toString().toInt(&okOrder);
                if (!okOrder) order = fallbackOrder;
                fallbackOrder = std::max(fallbackOrder + 1, order + 1);
                curCoords.push_back(Coord{order, QPointF(x, y)});
            }
        } else if (xml.isEndElement() && xml.name() == "Annotation") {
            flushAnnotation();
        }
    }
    flushAnnotation();
    f.close();

    if (xml.hasError()) {
        if (error) *error = xml.errorString();
        return -1;
    }

    std::unordered_set<std::string> seen;
    seen.reserve(annotations.size() + parsed.size());
    for (const auto& a : annotations) {
        seen.insert(dedupKey(a));
    }

    int added = 0;
    for (auto& a : parsed) {
        const std::string key = dedupKey(a);
        if (seen.find(key) != seen.end()) continue;
        seen.insert(key);
        annotations.push_back(std::move(a));
        ++added;
    }

    return added;
}

}  // namespace cytologick::annotation_io
