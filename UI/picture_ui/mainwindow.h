#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMediaPlayer>
#include <QVideoWidget>
#include <QMediaPlaylist>


namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_actions_triggered();

    void on_actions_2_triggered();

private:
    Ui::MainWindow *ui;
    QMediaPlayer *MediaPlayer;
    QMediaPlaylist *MediaPlaylist;

private:
    QImage* PaintRect(QImage& img, int x, int y, int width, int height, QColor &color);
};

#endif // MAINWINDOW_H
