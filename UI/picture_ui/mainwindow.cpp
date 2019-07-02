#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QDebug>
#include <QMessageBox>
#include <QPen>
#include <QPainter>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    MediaPlayer = new QMediaPlayer(this);
    MediaPlaylist = new QMediaPlaylist(MediaPlayer);
    MediaPlayer->setVideoOutput(ui->widget_2);
}

MainWindow::~MainWindow()
{
    delete ui;
}
struct coodinates
{
    int x;
    int y;
    int width;
    int height;
};

QImage* MainWindow::PaintRect(QImage& img, int x, int y, int width, int height, QColor &color)
{
    for (int i = x; i < x+width; i++)
    {

        img.setPixelColor(i, y, color);
        img.setPixelColor(i, y+height, color);

    }
    for (int j = y; j < y+height; j++)
    {
        img.setPixelColor(x, j, color);
        img.setPixelColor(x+width, j, color);
    }
    return &img;
}

void MainWindow::on_actions_triggered()
{
    QString fileName=QFileDialog::getOpenFileName(this,
                                                    tr("选择图像"),
                                                    "G:\\",
                                                    tr("Images(*.jpg *.png)")
                                                    );
    if(!(fileName.isEmpty()))
    {
    QString StrWidth,StrHeigth;


    QImage* img=new QImage,*scaledimg=new QImage;

    if(!(img->load(fileName)))
    {
        QMessageBox::information(this,
                                 tr("打开图像失败"),
                                 tr("打开图像失败！")
                                 );
        delete img;
        return;
    }

    *scaledimg = img->scaled(ui->label->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    QColor* col = new QColor(255,255,0,255);
    coodinates c;
    c.x=20;
    c.y=100;
    c.width=300;
    c.height=100;
    QString str="car 0.99";
    //框
    scaledimg = PaintRect(*scaledimg,c.x,c.y,c.width,c.height, *col);
    //字
    QPainter painter(scaledimg); //为图片构造一个QPainter
    painter.setCompositionMode(QPainter::CompositionMode_SourceIn);//画刷的组合模式CompositionMode_SourceOut这个模式为目标图像在上。
    //画笔和字体
    QPen pen = painter.pen();
    pen.setColor(Qt::yellow);
    QFont font = painter.font();
    font.setBold(true);//加粗
    font.setPixelSize(15);//字体大小
    painter.setPen(pen);
    painter.setFont(font);
    painter.drawText(c.x,c.y,str);

    ui->label->setPixmap(QPixmap::fromImage(*scaledimg));



    }
}

void MainWindow::on_actions_2_triggered()
{
    QString path = QFileDialog::getOpenFileName(this,
                                                tr("选择视频"),
                                                "G:\\",
                                                tr("Vedios(*.mp4 *.flv)")
                                                );
        if(path.isEmpty())
            return;
        qDebug() << __FILE__ << __LINE__ << path;
        MediaPlaylist->clear();
        MediaPlaylist->addMedia(QUrl::fromLocalFile(path));
        MediaPlaylist->setCurrentIndex(0);
        MediaPlayer->setPlaylist(MediaPlaylist);
        MediaPlayer->play();
}
