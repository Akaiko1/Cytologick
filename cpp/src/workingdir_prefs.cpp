#include "workingdir_prefs.h"

#include <QDir>
#include <QFileInfo>
#include <QSettings>
#include <QString>

namespace fs = std::filesystem;

namespace cytologick {

namespace {

constexpr auto kSettingsOrg = "Cytologick";
constexpr auto kSettingsApp = "Desktop";
constexpr auto kWorkingDirKey = "runtime/working_dir";

} // namespace

fs::path loadRememberedWorkingDir() {
    QSettings settings(QString::fromLatin1(kSettingsOrg), QString::fromLatin1(kSettingsApp));
    const QString raw = settings.value(QString::fromLatin1(kWorkingDirKey)).toString().trimmed();
    if (raw.isEmpty()) {
        return {};
    }

    const QString cleaned = QDir::cleanPath(raw);
    const QFileInfo info(cleaned);
    if (!info.exists() || !info.isDir()) {
        return {};
    }

    return fs::path(cleaned.toStdString());
}

void saveRememberedWorkingDir(const fs::path& dir) {
    if (dir.empty()) return;

    QSettings settings(QString::fromLatin1(kSettingsOrg), QString::fromLatin1(kSettingsApp));
    const QString cleaned = QDir::cleanPath(QString::fromStdString(dir.string()));
    settings.setValue(QString::fromLatin1(kWorkingDirKey), cleaned);
    settings.sync();
}

} // namespace cytologick

