#include "mainwindow.h"
#include <QApplication>
#include <QStyleFactory>
#include <iostream>

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    // Set application info
    QCoreApplication::setApplicationName("Cytologick");
    QCoreApplication::setApplicationVersion("1.0.0");
    QCoreApplication::setOrganizationName("Cytologick");

    // Use Fusion style for consistent look across platforms
    app.setStyle(QStyleFactory::create("Fusion"));

    // Dark palette for modern look
    QPalette darkPalette;
    darkPalette.setColor(QPalette::Window, QColor(53, 53, 53));
    darkPalette.setColor(QPalette::WindowText, Qt::white);
    darkPalette.setColor(QPalette::Base, QColor(25, 25, 25));
    darkPalette.setColor(QPalette::AlternateBase, QColor(53, 53, 53));
    darkPalette.setColor(QPalette::ToolTipBase, Qt::white);
    darkPalette.setColor(QPalette::ToolTipText, Qt::white);
    darkPalette.setColor(QPalette::Text, Qt::white);
    darkPalette.setColor(QPalette::Button, QColor(53, 53, 53));
    darkPalette.setColor(QPalette::ButtonText, Qt::white);
    darkPalette.setColor(QPalette::BrightText, Qt::red);
    darkPalette.setColor(QPalette::Link, QColor(42, 130, 218));
    darkPalette.setColor(QPalette::Highlight, QColor(42, 130, 218));
    darkPalette.setColor(QPalette::HighlightedText, Qt::black);
    app.setPalette(darkPalette);

    // Stylesheet for additional styling
    app.setStyleSheet(
        "QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }"
        "QPushButton { padding: 5px 15px; }"
        "QComboBox { padding: 3px; }"
        "QSlider::groove:horizontal { height: 8px; background: #1e1e1e; border-radius: 4px; }"
        "QSlider::handle:horizontal { background: #2a82da; width: 16px; margin: -4px 0; border-radius: 8px; }"
    );

    std::cout << "Starting Cytologick..." << std::endl;

    cytologick::MainWindow mainWindow;
    mainWindow.show();

    return app.exec();
}
