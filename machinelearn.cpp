#include "Interface/machinelearn.h"
#include "Interface/imbaprocess.h"
#include "ui_machinelearn.h"
#include <QMessageBox>
#include <iostream>
#include <QFileDialog>
#include <QSettings>
#include <QDebug>
#include <QStandardPaths>
#include <memory>
#include <QString>
#include <QDateTime>
#include <opencv2/opencv.hpp>
#include <QDateTime>
#include <opencv2/core/core.hpp>
#include <QTime>
#include <QIcon>
#include <fstream>
#include <string>
#pragma execution_character_set("utf-8")

namespace MachineLerrnSpace {
    bool StarBit;//视频分析开始标志位
    cv::Mat VideoImageFrame;//视频帧
}


//类中采用了静态的（static）的qwidge域或其子类，因为静态和全局对象进入main函数之前就产生了，所以早于main函数里的qapplication对象,所以这里直接给空，需要的时候再实例化使用
std::unique_ptr<machinelearn> machinelearn::BasicMachineLearn = nullptr;  // 直接实例化静态成员变量

machinelearn::machinelearn(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::machinelearn)
{
    ui->setupUi(this);
    this->setWindowIcon(QIcon(":/new/prefix1/Sourceimage/Inference.png"));
    this->setWindowTitle(QString("机器学习"));
    ui->toolButton_LocalVideo->setIcon(QIcon(":/new/prefix1/Sourceimage/Video.png"));
    ui->toolButton_Camera->setIcon(QIcon(":/new/prefix1/Sourceimage/Camera.png"));
    ui->toolButton_StopVideo->setIcon(QIcon(":/new/prefix1/Sourceimage/stop.png"));
    ui->toolButton_Imagedata->setIcon(QIcon(":/new/prefix1/Sourceimage/image.png"));
    ui->toolButton_Kmeans->setIcon(QIcon(":/new/prefix1/Sourceimage/Kmeans.png"));
    ui->toolButton_KNN->setIcon(QIcon(":/new/prefix1/Sourceimage/Knearest.png"));
    ui->toolButton_SVM->setIcon(QIcon(":/new/prefix1/Sourceimage/SVM.png"));
    ui->toolButton_SVMHOGDection->setIcon(QIcon(":/new/prefix1/Sourceimage/SVM+HOG.png"));
    ui->textEdit_MachineLearnInforShow->setReadOnly(true);//信息区设置为只读
    ui->textEdit_MachineLearnInforShow->setTextColor(QColor(0, 120, 0));//设置文本颜色
    ui->textEdit_MachineLearnInforShow->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);//设置垂直滚动条策略
    ui->textEdit_MachineLearnInforShow->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);//设置水平滚动条策略
    ui->label_MachineLearnDataShow->clear();
    ui->textEdit_MachineLearnInforShow->clear();
}

machinelearn::~machinelearn()
{
    delete ui;
}
//打开本地图像文件
void machinelearn::on_toolButton_Imagedata_clicked(){
    MachineLerrnSpace::StarBit=0;
    //配置文件完整路径
    QString config_path = qApp->applicationDirPath() + "/config/Setting.ini";
    //通过QSetting类创建配置ini格式文件路径
    std::unique_ptr<QSettings> pIniSet(new QSettings(config_path, QSettings::IniFormat));
    //将配置文件中的值加载进来转换为QString类型存储在LastImagePath中
    QString lastPath = pIniSet->value("/LastImagePath/path").toString();
    if(lastPath.isEmpty())
    {
        //系统标准的图片存储路径
        lastPath = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    }
    QString fileName = QFileDialog::getOpenFileName(this, "请选择图片", lastPath, "图片(*.png *.jpg);;");
    if(fileName.isEmpty())
    {
        //加载失败，向信息区中写入信息
        QDateTime currentDateTime = QDateTime::currentDateTime();
        QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
        ui->textEdit_MachineLearnInforShow->append(QString("本地图片数据采集:%1，图像加载失败，文件路径为空").arg(formattedTime));
        ui->label_MachineLearnDataShow->clear();
        return;
    }
    //将加载成功的图像显示到Qlabel中
    QPixmap *pix = new QPixmap(fileName);
    pix->scaled(ui->label_MachineLearnDataShow->size(),Qt::KeepAspectRatio);
    ui->label_MachineLearnDataShow->setScaledContents(true);
    ui->label_MachineLearnDataShow->setPixmap(*pix);
    std::unique_ptr<ImbaProcess> imageChange=std::make_unique<ImbaProcess>();
    MachineLerrnSpace::VideoImageFrame=imageChange->QPixmapToMat(*pix);
    //加载成功，向信息区中写入信息
    QDateTime currentDateTime = QDateTime::currentDateTime();
    QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
    ui->textEdit_MachineLearnInforShow->append(QString("本地图片数据采集:%1，图像加载成功").arg(formattedTime));
    delete pix;
    pix = nullptr;
    //将文件路径中最后一个斜杠的位置找到，截取最后一个斜杠前面的路径放入INI文件中去
    int end = fileName.lastIndexOf("/");
    QString _path = fileName.left(end);
    pIniSet->setValue("/LastImagePath/path", _path);
}

//打开本地视频文件
void machinelearn::on_toolButton_LocalVideo_clicked(){
    //允许加载视频
    MachineLerrnSpace::StarBit=1;
    //配置文件完整路径
    QString config_path = qApp->applicationDirPath() + "/config/Setting.ini";
    //通过QSetting类创建配置ini格式文件路径
    std::unique_ptr<QSettings> pIniSet(new QSettings(config_path, QSettings::IniFormat));
    //将配置文件中的值加载进来转换为QString类型存储在LastImagePath中
    QString lastPath = pIniSet->value("/LastVideoPath/path").toString();

    if(lastPath.isEmpty())
    {
        //系统标准的视频存储路径
        lastPath = QStandardPaths::writableLocation(QStandardPaths::MoviesLocation);
    }
    QString fileName = QFileDialog::getOpenFileName(this, "请选择视频", lastPath, "视频(*.avi *.mp4);;");

        if(fileName.isEmpty())
        {
            //加载失败，向信息区中写入信息
            QDateTime currentDateTime = QDateTime::currentDateTime();
            QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
            ui->textEdit_MachineLearnInforShow->append(QString("本地视频数据采集:%1，视频加载失败，文件路径为空").arg(formattedTime));
            ui->label_MachineLearnDataShow->clear();
            MachineLerrnSpace::StarBit=0;
            return;
        }
        //将视频加载到Qlabel中
        cv::VideoCapture Capture;
        std::unique_ptr<ImbaProcess> imageChange=std::make_unique<ImbaProcess>();
        bool ret=Capture.open(fileName.toStdString());
        int height =Capture.get(cv::CAP_PROP_FRAME_HEIGHT);//视频高
        int width  =Capture.get(cv::CAP_PROP_FRAME_WIDTH);//视频宽
        double fps =Capture.get(cv::CAP_PROP_FPS);//视频帧率
        double count =Capture.get(cv::CAP_PROP_FRAME_COUNT);//总帧率
        if(!ret)return;//加载失败直接返回
        //加载成功，向信息区中写入信息
        QDateTime currentDateTime = QDateTime::currentDateTime();
        QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
        ui->textEdit_MachineLearnInforShow->append(QString("本地视频数据采集:%1，Height:%2,Width:%3,FPS:%4,Count:%5").arg(formattedTime).arg(height).arg(width).arg(fps).arg(count));
        while (true) {
              bool state=Capture.read(MachineLerrnSpace::VideoImageFrame);
              if(!state)break;
              //将Mat对象转换为QpixMap对象,然后将图像大小设置成Qlabel大小并赋值到Qlabel中去
              QPixmap pix =imageChange->matToQPixmap(MachineLerrnSpace::VideoImageFrame);
              pix.scaled(ui->label_MachineLearnDataShow->size());
              ui->label_MachineLearnDataShow->setScaledContents(true);
              ui->label_MachineLearnDataShow->setPixmap(pix);
              //检测到停止采集键键直接退出
              if(!MachineLerrnSpace::StarBit){
                 break;
              }
              cv::waitKey(1);
        }
        //将文件路径中最后一个斜杠的位置找到，截取最后一个斜杠前面的路径放入INI文件中去
        Capture.release();
        int end = fileName.lastIndexOf("/");
        QString _path = fileName.left(end);
        pIniSet->setValue("/LastVideoPath/path", _path);
}

//打开摄像头
void machinelearn::on_toolButton_Camera_clicked(){
    //使用OPencv相机数据
    MachineLerrnSpace::StarBit=1;
    cv::VideoCapture Capture;
    Capture.open(ui->spinBox_CameraNumber->value());//摄像头编号索引为0，加载方式为ANY
    int height =Capture.get(cv::CAP_PROP_FRAME_HEIGHT);//视频高
    int width  =Capture.get(cv::CAP_PROP_FRAME_WIDTH);//视频宽
    double fps =Capture.get(cv::CAP_PROP_FPS);//视频帧率
    double count =Capture.get(cv::CAP_PROP_FRAME_COUNT);//总帧率
    //在信息区显示当前加载视频的信息格式
    QDateTime currentDateTime = QDateTime::currentDateTime();
    QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
    ui->textEdit_MachineLearnInforShow->append(QString("实时相机数据采集:%1，Height:%2,Width:%3,FPS:%4,Count:%5").arg(formattedTime).arg(height).arg(width).arg(fps).arg(count));
    //将加载的视频写进Qlabel中
    std::unique_ptr<ImbaProcess> imageChange=std::make_unique<ImbaProcess>();
    while(true){
      //读帧
      bool ret = Capture.read(MachineLerrnSpace::VideoImageFrame);
      if (!ret)break;
      //将Mat对象转换为QpixMap对象,然后将图像大小设置成Qlabel大小并赋值到Qlabel中去
      QPixmap pix = imageChange->matToQPixmap(MachineLerrnSpace::VideoImageFrame);
      pix.scaled(ui->label_MachineLearnDataShow->size(),Qt::KeepAspectRatio);
      ui->label_MachineLearnDataShow->setScaledContents(true);
      ui->label_MachineLearnDataShow->setPixmap(pix);
      //检测到停止采集键键直接退出
      if(!MachineLerrnSpace::StarBit){
         break;
      }
      cv::waitKey(1);
    }
    Capture.release();
}

//Kmeans分类
void machinelearn::on_toolButton_Kmeans_clicked(){
    if(MachineLerrnSpace::VideoImageFrame.empty()){
        //加载失败，向信息区中写入信息
        QDateTime currentDateTime = QDateTime::currentDateTime();
        QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
        ui->textEdit_MachineLearnInforShow->append(QString("本地视频数据采集:%1，数据加载失败，加载目标为空").arg(formattedTime));
        return;
      }
     //创建颜色表
    cv::Scalar ClolrTable[]={
        cv::Scalar(255,0,0),
        cv::Scalar(0,255,0),
        cv::Scalar(0,0,255),
        cv::Scalar(255,255,0),
        cv::Scalar(0,255,255),
    };
    //图像信息
    int Width=MachineLerrnSpace::VideoImageFrame.cols;
    int Height=MachineLerrnSpace::VideoImageFrame.rows;
    int dim   =MachineLerrnSpace::VideoImageFrame.channels();
    //初始化定义
    std::unique_ptr<ImbaProcess> imageChange=std::make_unique<ImbaProcess>();
    int sampleCount=Width*Height;
    cv::Mat Label;
    cv::Mat Centers;
    //将一个三维数据(w,h,c)转换为一个二维数据(特征，数据),也就是将一个RGB的三维数据转换为一个二维的样本数据
    cv::Mat Sample_Data=MachineLerrnSpace::VideoImageFrame.reshape(3,sampleCount);
    cv::Mat data;
    Sample_Data.convertTo(data,CV_32F);//将样本数据转换为浮点数
    //运行Kmeans算法
    cv::TermCriteria Criteria=cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT,10,0.1);
    cv::kmeans(data,ui->spinBox_KmeansNumber->value(),Label,Criteria,ui->spinBox_KmeansNumber->value(),cv::KMEANS_PP_CENTERS,Centers);
    //将图像分割结果显示出来
    cv::Mat Result=cv::Mat::zeros(MachineLerrnSpace::VideoImageFrame.size(),MachineLerrnSpace::VideoImageFrame.type());
    for (int i=0;i<Height;i++ ) {
        for (int j=0;j<Width;j++) {
             //因为输出的标签是(Width*Height)行，1列的数据
             int index=i*Width+j;
             int LabelValue=Label.at<int>(index,0);
             Result.at<cv::Vec3b>(i,j)[0]=ClolrTable[LabelValue][0];
             Result.at<cv::Vec3b>(i,j)[1]=ClolrTable[LabelValue][1];
             Result.at<cv::Vec3b>(i,j)[2]=ClolrTable[LabelValue][2];
        }
    }
    //判断是否需要绘制色卡
    if(ui->checkBox_PaintColorCard->checkState()==Qt::Checked){
        //色卡比率容器
        std::vector<float>ColorRate(ui->spinBox_KmeansNumber->value());
        //统计每个簇的数量
        for (int k=0;k<Label.rows;k++ ) {
            ColorRate[Label.at<int>(k,0)]++;
        }
        //计算比率
        for (int t=0;t<ui->spinBox_KmeansNumber->value();t++ ){
             ColorRate[t]/=sampleCount;
        }
        //绘制色卡
        int offset=0;
        cv::Mat card=cv::Mat::zeros(50,Width,CV_8UC3);
        for (int l=0;l<ui->spinBox_KmeansNumber->value();l++ ) {
            //色卡矩形
            cv::Rect rect;
            rect.y=0;
            rect.x=offset;
            rect.height=20;
            rect.width=round(ColorRate[l]*Width);
            //色卡x轴偏移
            offset+=rect.width;
            int b=Centers.at<float>(l,0);
            int g=Centers.at<float>(l,1);
            int r=Centers.at<float>(l,2);
            //色卡颜色
            cv::rectangle(Result,rect,cv::Scalar(b,g,r),-1,8,0);
            //色卡比率显示
            int Radio=ColorRate[l]*100;
            cv::putText(Result,cv::format("Rate:%d",Radio),cv::Point(rect.x,rect.y+10),cv::FONT_HERSHEY_SIMPLEX,0.3,cv::Scalar(255,0,0),1,8,0);
        }
    }
    //将Mat对象转换为QpixMap对象,然后将图像大小设置成Qlabel大小并赋值到Qlabel中去
    QPixmap pix = imageChange->matToQPixmap(Result);
    pix.scaled(ui->label_MachineLearnDataShow->size(),Qt::KeepAspectRatio);
    ui->label_MachineLearnDataShow->setScaledContents(true);
    ui->label_MachineLearnDataShow->setPixmap(pix);
}
//KNN分类
void machinelearn::on_toolButton_KNN_clicked(){
     /*一、首先读入样本图像digits.png,构建样本数据与标签数据*/
     QDateTime currentDateTime = QDateTime::currentDateTime();
     QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
     ui->textEdit_MachineLearnInforShow->append(QString("%1:KNN分类开始，准备数据集中...").arg(formattedTime));
     cv::Mat Sampledata=cv::imread("E:/DeskTop/SongXinFile/Qt Design/SongXinDemo/trainData/digits.png",cv::IMREAD_GRAYSCALE);
     //分割为5000个单独的数字图像
     cv::Mat images=cv::Mat::zeros(5000,400,CV_8UC1);//存储每个分割后的手写数字像素值
     cv::Mat labels=cv::Mat::zeros(5000,1  ,CV_8UC1);//存储每个分割后的手写数字实际值0，1，2...
     //创建截取训练数据集中的数字的截取矩形
     int index=0;
     cv::Rect rox;
     rox.x=0;
     rox.height=1;
     rox.width=400;
     //然后通过遍历的方式对5000个数字实现遍历截取
     for (int i=0;i<50;i++) {//有50行
         int label=i/5;//代表当前数字
         int offsety=i*20;//为了跳转到下一行y轴的偏移，因为是20*20的图像嘛，所以每一步向下偏移20个像素
         for (int j=0;j<100;j++) {//有100列
             int offsetx=j*20;//为了跳转到下一列x轴的偏移
             //创建一个截取数字的Mat对象
             cv::Mat digit=cv::Mat(20,20,CV_8UC1);
             for (int k=0;k<20;k++ ){
                 for (int t=0;t<20;t++){
                     digit.at<uchar>(k,t)=Sampledata.at<uchar>(k+offsety,t+offsetx);
                 }
             }
             //我们把截取出来的每个Mat对象，也就是每个数字的像素值对象，重新整形成一行400列的数据
             cv::Mat one_row=digit.reshape(1,1);//单通道且1行
             rox.y=index;//索引值赋值给截取的矩形的y坐标，因为马上我要把矩形的数据放入images中去
             one_row.copyTo(images(rox));//将截取出来的像素数据放入训练数据中去
             //将截取出来的像素区域中的数据放入训练数据中的实际存储区中
             labels.at<uchar>(index,0)=label;
             index++;
         }
     }
     //状态信息输出
     ui->textEdit_MachineLearnInforShow->append(QString("%1:手动加载训练数据并开始训练...").arg(formattedTime));
     /*二、训练并将训练模型保存至输出文件*/
     //将处理好的数据集转换为浮点数
     images.convertTo(images,CV_32F);//特征矩阵
     labels.convertTo(labels,CV_32S);//目标标签
     //开始训练
     cv::Ptr<cv::ml::KNearest>knn=cv::ml::KNearest::create();//创建实例Knn实例对象的智能指针
     knn->setDefaultK(ui->spinBox_KnnNumber->value());//设置默认最近邻数目
     knn->setIsClassifier(true);//将 KNN 设置为分类器模式。这意味着算法将用于分类任务而不是回归任务。
     //参数1:将标签数据给入 参数2:代表每一行是一个样本数据 参数3:目标标签
     cv::Ptr<cv::ml::TrainData>tdata=cv::ml::TrainData::create(images,cv::ml::ROW_SAMPLE,labels);
     //设置训练分类器
     knn->train(tdata);
     //保存训练好的knn模型
     knn->save("E:/DeskTop/SongXinFile/Qt Design/SongXinDemo/trainData/knn_knowledge.yml");
     ui->textEdit_MachineLearnInforShow->append(QString("%1:训练完成准备加载测试数据...").arg(formattedTime));
     /*三、加载KNN分类器，并处理测试数据得出结果*/
     cv::Ptr<cv::ml::KNearest> knnload=cv::Algorithm::load<cv::ml::KNearest>("E:/DeskTop/SongXinFile/Qt Design/SongXinDemo/trainData/knn_knowledge.yml");
     //加载测试数据
     if(MachineLerrnSpace::VideoImageFrame.empty()){
         //加载失败，向信息区中写入信息
         QDateTime currentDateTime = QDateTime::currentDateTime();
         QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
         ui->textEdit_MachineLearnInforShow->append(QString("测试数据采集:%1，测试数据采集失败，加载目标为空").arg(formattedTime));
         return;
       }
       //处理测试数据，将测试数据转换为和训练标签数据同维度的标签
       cv::Mat TrainData,predictResult,Gray;
       cv::cvtColor(MachineLerrnSpace::VideoImageFrame,Gray,cv::COLOR_BGR2GRAY);
       cv::resize(Gray, TrainData, cv::Size(20, 20));//将测试数据大小设置为20x20
       TrainData.convertTo(TrainData, CV_32F);//拿到和训练数据通规格的测试数据
       cv::Mat data=TrainData.reshape(1,1);
       ui->textEdit_MachineLearnInforShow->append(QString("%1:加载测试数据成功，即将预测结果...").arg(formattedTime));
       //预测分类
       knn->findNearest(data,ui->spinBox_KnnNumber->value(),predictResult);
       cv::Mat Result=MachineLerrnSpace::VideoImageFrame.clone();
       for (int i=0;i<predictResult.rows;i++) {
            int predict=predictResult.at<float>(i,0);
            cv::putText(Result,cv::format("%d",predict),cv::Point(Result.rows/2,Result.cols/4),cv::FONT_HERSHEY_SIMPLEX,0.1,cv::Scalar(0,255,0),1,8,0);
            ui->textEdit_MachineLearnInforShow->append(QString("%1:KNN分类预测结果%2").arg(formattedTime).arg(predict));
       }
       //将Mat对象转换为QpixMap对象,然后将图像大小设置成Qlabel大小并赋值到Qlabel中去
       std::unique_ptr<ImbaProcess> imageChange=std::make_unique<ImbaProcess>();
       QPixmap pix = imageChange->matToQPixmap(Result);
       pix.scaled(ui->label_MachineLearnDataShow->size(),Qt::KeepAspectRatio);
       ui->label_MachineLearnDataShow->setScaledContents(true);
       ui->label_MachineLearnDataShow->setPixmap(pix);
}
//SVM分类
void machinelearn::on_toolButton_SVM_clicked(){
    /*一、首先读入样本图像digits.png,构建样本数据与标签数据*/
    QDateTime currentDateTime = QDateTime::currentDateTime();
    QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
    ui->textEdit_MachineLearnInforShow->append(QString("%1:SVM分类开始，准备数据集中...").arg(formattedTime));
    cv::Mat Sampledata=cv::imread("E:/DeskTop/SongXinFile/Qt Design/SongXinDemo/trainData/digits.png",cv::IMREAD_GRAYSCALE);
    //分割为5000个单独的数字图像
    cv::Mat images=cv::Mat::zeros(5000,400,CV_8UC1);//存储每个分割后的手写数字像素值
    cv::Mat labels=cv::Mat::zeros(5000,1  ,CV_8UC1);//存储每个分割后的手写数字实际值0，1，2...
    //创建截取训练数据集中的数字的截取矩形
    int index=0;
    cv::Rect rox;
    rox.x=0;
    rox.height=1;
    rox.width=400;
    //然后通过遍历的方式对5000个数字实现遍历截取
    for (int i=0;i<50;i++) {//有50行
        int label=i/5;//代表当前数字
        int offsety=i*20;//为了跳转到下一行y轴的偏移，因为是20*20的图像嘛，所以每一步向下偏移20个像素
        for (int j=0;j<100;j++) {//有100列
            int offsetx=j*20;//为了跳转到下一列x轴的偏移
            //创建一个截取数字的Mat对象
            cv::Mat digit=cv::Mat(20,20,CV_8UC1);
            for (int k=0;k<20;k++ ){
                for (int t=0;t<20;t++){
                    digit.at<uchar>(k,t)=Sampledata.at<uchar>(k+offsety,t+offsetx);
                }
            }
            //我们把截取出来的每个Mat对象，也就是每个数字的像素值对象，重新整形成一行400列的数据
            cv::Mat one_row=digit.reshape(1,1);//单通道且1行
            rox.y=index;//索引值赋值给截取的矩形的y坐标，因为马上我要把矩形的数据放入images中去
            one_row.copyTo(images(rox));//将截取出来的像素数据放入训练数据中去
            //将截取出来的像素区域中的数据放入训练数据中的实际存储区中
            labels.at<uchar>(index,0)=label;
            index++;
        }
    }
    //状态信息输出
    ui->textEdit_MachineLearnInforShow->append(QString("%1:手动加载训练数据并开始训练...").arg(formattedTime));
    /*二、训练并将训练模型保存至输出文件*/
    //将处理好的数据集转换为浮点数
    images.convertTo(images,CV_32F);//特征矩阵
    labels.convertTo(labels,CV_32S);//目标标签
    //开始训练
    cv::Ptr<cv::ml::SVM>svm=cv::ml::SVM::create();//创建实例SVM实例对象的智能指针
    svm->setGamma(ui->doubleSpinBox_GamaValue->value());//设置影响范围
    svm->setC(ui->doubleSpinBox_CValue->value());//设置惩罚参数
    svm->setKernel(cv::ml::SVM::LINEAR);//SVM核函数的类型为线性函数
    svm->setType(cv::ml::SVM::C_SVC);//支持向量分类

    //参数1:将标签数据给入 参数2:代表每一行是一个样本数据 参数3:目标标签
    cv::Ptr<cv::ml::TrainData>tdata=cv::ml::TrainData::create(images,cv::ml::ROW_SAMPLE,labels);
    //设置训练分类器
    svm->train(tdata);
    //保存训练好的svm模型
    svm->save("E:/DeskTop/SongXinFile/Qt Design/SongXinDemo/trainData/svm_knowledge.yml");
    ui->textEdit_MachineLearnInforShow->append(QString("%1:训练完成准备加载测试数据...").arg(formattedTime));

    /*三、加载SVM分类器，并处理测试数据得出结果*/
    cv::Ptr<cv::ml::SVM> svmload=cv::ml::StatModel::load<cv::ml::SVM>("E:/DeskTop/SongXinFile/Qt Design/SongXinDemo/trainData/svm_knowledge.yml");
    //加载测试数据
    if(MachineLerrnSpace::VideoImageFrame.empty()){
        //加载失败，向信息区中写入信息
        QDateTime currentDateTime = QDateTime::currentDateTime();
        QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
        ui->textEdit_MachineLearnInforShow->append(QString("测试数据采集:%1，测试数据采集失败，加载目标为空").arg(formattedTime));
        return;
      }
      //处理测试数据，将测试数据转换为和训练标签数据同维度的标签
      cv::Mat TrainData,predictResult,Gray;
      cv::cvtColor(MachineLerrnSpace::VideoImageFrame,Gray,cv::COLOR_BGR2GRAY);
      cv::resize(Gray, TrainData, cv::Size(20, 20));//将测试数据大小设置为20x20
      TrainData.convertTo(TrainData, CV_32F);//拿到和训练数据通规格的测试数据
      cv::Mat data=TrainData.reshape(1,1);
      ui->textEdit_MachineLearnInforShow->append(QString("%1:加载测试数据成功，即将预测结果...").arg(formattedTime));
      //预测分类
     svm->predict(data,predictResult);
      cv::Mat Result=MachineLerrnSpace::VideoImageFrame.clone();
      for (int i=0;i<predictResult.rows;i++) {
           int predict=predictResult.at<float>(i,0);
           cv::putText(Result,cv::format("%d",predict),cv::Point(Result.rows/2,Result.cols/4),cv::FONT_HERSHEY_SIMPLEX,0.1,cv::Scalar(0,255,0),1,8,0);
           ui->textEdit_MachineLearnInforShow->append(QString("%1:SVM分类预测结果%2").arg(formattedTime).arg(predict));
      }
      //将Mat对象转换为QpixMap对象,然后将图像大小设置成Qlabel大小并赋值到Qlabel中去
      std::unique_ptr<ImbaProcess> imageChange=std::make_unique<ImbaProcess>();
      QPixmap pix = imageChange->matToQPixmap(Result);
      pix.scaled(ui->label_MachineLearnDataShow->size(),Qt::KeepAspectRatio);
      ui->label_MachineLearnDataShow->setScaledContents(true);
      ui->label_MachineLearnDataShow->setPixmap(pix);

}
//SVM+HOG对象检测
void get_hog_descriptor(cv::Mat &image, std::vector<float> &desc) {//获取图像的HOG描述子
    cv::HOGDescriptor hog;//创建HOG描述子
    //保证图像宽和高比例不变的前提下将图像的宽设置为64，然后计算高的比率，按比率缩放，保证图像不变形
    int h = image.rows;//150
    int w = image.cols;//70
    float rate = 64.0 / w;//
    cv::Mat img, gray;
    resize(image, img, cv::Size(64, int(rate*h)));
    //将缩放的图像转换为灰度图，创建一个单通道背景颜色为灰度127
    cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Mat result = cv::Mat::zeros(cv::Size(64, 128), CV_8UC1);
    result = cv::Scalar(127);
    //目的是将我们缩放后的图像的灰度图垂直居中地拷贝到目标背景图像(64x128)中
    cv::Rect roi;
    roi.x = 0;
    roi.width = 64;
    roi.y = (128 - gray.rows) / 2;
    roi.height = gray.rows;
    gray.copyTo(result(roi));
    //计算HOG描述子
    hog.compute(result, desc, cv::Size(8, 8), cv::Size(0, 0));
}

void train_ele_watch(std::string positive_dir, std::string negative_dir) {//训练电表数据
    // 创建变量
    cv::Mat trainData = cv::Mat::zeros(cv::Size(3780, 26), CV_32FC1);//建立HOG描述子特征数据，每张64x128的图像有3780个描述子，每一行代表一个图像描述子
    cv::Mat labels = cv::Mat::zeros(cv::Size(1, 26), CV_32SC1);//建立标签数据，每一行代表一个图像数据标签
    std::vector<std::string> images;//批量文件路径
    cv::glob(positive_dir, images);//将路径下的所有的正向文件存储到images中
    int pos_num = images.size();//计算这个路径下的文件有多少文件，方便后续存描述子的时候，再这个数量后面添加负向数据的描述子特征向量和标签向量

    //正向数据的特征与标签的存入
    for (int i = 0; i < images.size(); i++) {
        cv::Mat image = cv::imread(images[i].c_str());//循环读取正向数据
        std::vector<float> fv;
        get_hog_descriptor(image, fv);//循环计算正向描述子
        for (int j = 0; j < fv.size(); j++) {
            trainData.at<float>(i, j) = fv[j];//将正向数据的描述子放入特征数据中
        }
        labels.at<int>(i, 0) = 1;//将正向数据的标签标记为1
    }

    images.clear();//批量文件路径情况准备加载负向数据
    cv::glob(negative_dir, images);//将路径下的所有的负向文件存储到images中
    for (int i = 0; i < images.size(); i++) {
        cv::Mat image = cv::imread(images[i].c_str());//循环读取负向数据
        std::vector<float> fv;
        get_hog_descriptor(image, fv);//循环计算负向描述子
        for (int j = 0; j < fv.size(); j++) {
            trainData.at<float>(i + pos_num, j) = fv[j];//将负向数据的描述子放入特征数据中
        }
        labels.at<int>(i + pos_num, 0) = -1;//将负向数据的标签标记为-1
    }

    // 训练SVM仪表分类器
    cv::Ptr<cv::ml::SVM > svm = cv::ml::SVM::create();
    svm->setKernel(cv::ml::SVM::LINEAR);
    svm->setC(2.0);
    svm->setType(cv::ml::SVM::C_SVC);
    svm->train(trainData, cv::ml::ROW_SAMPLE, labels);
    svm->save("E:/DeskTop/SongXinFile/Qt Design/SongXinDemo/trainData/svm_hog_elec.yml");
}

cv::Mat& hog_svm_detector_demo(cv::Mat &image,double Confidlevel=0.1,int MinSize=8,int MaxSize=32) {
    // 创建HOG与加载SVM训练数据
    cv::HOGDescriptor hog;//创建HOG描述子，用于提取图像中的HOG特征
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load("E:/DeskTop/SongXinFile/Qt Design/SongXinDemo/trainData/svm_hog_elec.yml");
    cv::Mat sv = svm->getSupportVectors();//取出关键样本点
    cv::Mat alpha, svidx;
    double rho = svm->getDecisionFunction(0, alpha, svidx);//获取权重alpha与偏移量rho

    // 构建detector 这部分代码是将从训练好的 SVM 模型中提取出来的参数转化为一个可以用来检测目标的格式，然后将这个格式传递给 HOG 检测器。
    std::vector<float> svmDetector;//创建一个空的检测器: svmDetector 是一个存储参数的列表
    svmDetector.clear();
    svmDetector.resize(sv.cols + 1);//调整大小: svmDetector 的大小被调整为支持向量的数量加上一个位置偏移量。
    for (int j = 0; j < sv.cols; j++) {
        svmDetector[j] = -sv.at<float>(0, j);//填充参数: 将支持向量的参数填入 svmDetector 列表中.
    }
    svmDetector[sv.cols] = (float)rho;//同时加上一个偏移量（rho）
    hog.setSVMDetector(svmDetector);//设置检测器: 把填充好的 svmDetector 传递给 HOG 检测器，使其使用这些参数进行目标检测。

    std::vector<cv::Rect> objects;//用于存储检测到的目标区域
    //参数1：待检测图像 参数2：存储检测到的目标区域图像 参数3：阈值用于过滤检测结果的置信度
    //参数4：窗口的最小大小 参数5：窗口的最大大小  参数6：图像的缩放因子，用于多尺度检测
    hog.detectMultiScale(image, objects, Confidlevel, cv::Size(MinSize, MinSize), cv::Size(MaxSize, MaxSize), 1.25);
    //绘制检测结果
    for (int i = 0; i < objects.size(); i++) {
        rectangle(image, objects[i], cv::Scalar(0, 0, 255), 4, 8, 0);
    }
    return image;
}
void machinelearn::on_toolButton_SVMHOGDection_clicked(){

   //一、构建样本数据
   QDateTime currentDateTime = QDateTime::currentDateTime();
   QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
   ui->textEdit_MachineLearnInforShow->append(QString("%1:SVM分类开始，准备数据集中...").arg(formattedTime));
   std::string positive="E:/DeskTop/SongXinFile/Qt Design/SongXinDemo/ElectricWatchimage/positive";
   std::string negative="E:/DeskTop/SongXinFile/Qt Design/SongXinDemo/ElectricWatchimage/negative";
   train_ele_watch(positive,negative);
   //二、加载样本数据
   if(MachineLerrnSpace::VideoImageFrame.empty()){
       //加载失败，向信息区中写入信息
       QDateTime currentDateTime = QDateTime::currentDateTime();
       QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
       ui->textEdit_MachineLearnInforShow->append(QString("测试数据采集:%1，测试数据采集失败，加载目标为空").arg(formattedTime));
       return;
     }
    ui->textEdit_MachineLearnInforShow->append(QString("%1:模型文件训练成功，准备加载样本数据...").arg(formattedTime));
    cv::Mat TestData=MachineLerrnSpace::VideoImageFrame.clone();
    cv::Mat Result= hog_svm_detector_demo(TestData,ui->doubleSpinBox_ConfidenceLevel->value(),ui->spinBox_MinFrame->value(),ui->spinBox_MaxFrame->value());
    //将Mat对象转换为QpixMap对象,然后将图像大小设置成Qlabel大小并赋值到Qlabel中去
    std::unique_ptr<ImbaProcess> imageChange=std::make_unique<ImbaProcess>();
    QPixmap pix = imageChange->matToQPixmap(Result);
    pix.scaled(ui->label_MachineLearnDataShow->size(),Qt::KeepAspectRatio);
    ui->label_MachineLearnDataShow->setScaledContents(true);
    ui->label_MachineLearnDataShow->setPixmap(pix);
    ui->textEdit_MachineLearnInforShow->append(QString("%1:检测完成...").arg(formattedTime));
}
//基于Caffe框架图像分类
std::vector<std::string> readClassNames(std::string filePatch)
{
    std::vector<std::string> classNames;
    std::ifstream fp(filePatch);
    if (!fp.is_open())
    {
        //文件打开失败
        printf("could not open file...\n");
        exit(-1);
    }
    std::string name;
    //未到文件结尾
    while (!fp.eof())
    {   //按行读取文件中的数据,并存入name中
        std::getline(fp, name);
        //如果name中的数据不为空，那么将数据存到容器中
        if (name.length())
            classNames.push_back(name);
    }
    //读取完成关闭文件，并返回容器
    fp.close();
    return classNames;
}
void machinelearn::on_checkBox_ImgaeClassify_stateChanged(int state){
  //选中则计算
  if(state==Qt::Checked){
      if(MachineLerrnSpace::VideoImageFrame.empty()){
          //加载失败，向信息区中写入信息
          QDateTime currentDateTime = QDateTime::currentDateTime();
          QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
          ui->textEdit_MachineLearnInforShow->append(QString("%1:，测试数据采集失败，加载目标为空").arg(formattedTime));
          return;
        }
       cv::Mat image=MachineLerrnSpace::VideoImageFrame.clone();
       //模型文件,模型二进制文件,以及标签文件路径
       std::string model_dir = "E:/DeskTop/SongXinFile/Qt Design/SongXinDemo/dnn/GoogLeNet/";
       std::string weight_path = model_dir + "bvlc_googlenet.caffemodel";//模型权重文件
       std::string config_path = model_dir + "bvlc_googlenet.prototxt"  ;//模型框架文件
       //加载模型
       cv::dnn::Net net = cv::dnn::readNetFromCaffe(config_path, weight_path);
       if (net.empty()) {
           printf("read caffe model data failure...\n");
           return;
       }
       //读取标签文件，并将输入图像转换为模型输入要求的格式
       std::vector<std::string> labels = readClassNames(model_dir+ "classification_classes_ILSVRC2012.txt");
       cv::Mat inputBlob = cv::dnn::blobFromImage(image, 1.0, cv::Size(224, 224), cv::Scalar(104, 117, 123), false, false);

       // 执行图像分类
       cv::Mat prob;
       //设置模型输入,并前向执行推理预测，并返回输出层标签预测结果
       net.setInput(inputBlob);
       prob = net.forward();
       //计算推理时间
       std::vector<double> times;
       double time = net.getPerfProfile(times);//将每一层的推理时间填充到 times 向量中，以便进行更细粒度的性能分析,并返回推理总时间
       float ms = (time * 1000) / cv::getTickFrequency();//用于将s转换为ms
       // 得到最可能分类输出
       cv::Mat probMat = prob.reshape(1, 1);//将预测结果转换为单通道1行格式的数据
       cv::Point classNumber;
       double classProb;
       //找到最大值的预测得分和位置
       cv::minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber);
       int classidx = classNumber.x;//这个就是标签文件中的行数
       // 显示文本
       putText(image, labels.at(classidx), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2, 8);
       ui->textEdit_MachineLearnInforShow->append(QString("预测结果%1,结果位置%2,结果得分%3,推理时间%4ms").arg(QString::fromStdString(labels.at(classidx))).arg(classidx).arg(classProb).arg(ms));
       //将Mat对象转换为QpixMap对象,然后将图像大小设置成Qlabel大小并赋值到Qlabel中去
       QDateTime currentDateTime = QDateTime::currentDateTime();
       QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
       std::unique_ptr<ImbaProcess> imageChange=std::make_unique<ImbaProcess>();
       QPixmap pix = imageChange->matToQPixmap(image);
       pix.scaled(ui->label_MachineLearnDataShow->size(),Qt::KeepAspectRatio);
       ui->label_MachineLearnDataShow->setScaledContents(true);
       ui->label_MachineLearnDataShow->setPixmap(pix);
       ui->textEdit_MachineLearnInforShow->append(QString("%1:检测完成...").arg(formattedTime));
   }
}
//SSD对象检测
void machinelearn::on_checkBox_SSDobjectDection_stateChanged(int state){
    //SSD对象检测预训练模型
    std::string objNames[] = { "background",
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor" };
    //选中则计算
    if(state==Qt::Checked){
        if(MachineLerrnSpace::VideoImageFrame.empty()){
            //加载失败，向信息区中写入信息
            QDateTime currentDateTime = QDateTime::currentDateTime();
            QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
            ui->textEdit_MachineLearnInforShow->append(QString("%1:，测试数据采集失败，加载目标为空").arg(formattedTime));
            return;
          }
         cv::Mat image=MachineLerrnSpace::VideoImageFrame.clone();
         //首先读取权重文件和描述文件
         std::string ssd_config = "E:/DeskTop/SongXinFile/Qt Design/SongXinDemo/dnn/SSD/MobileNetSSD_deploy.prototxt";
         std::string ssd_weight = "E:/DeskTop/SongXinFile/Qt Design/SongXinDemo/dnn/SSD/MobileNetSSD_deploy.caffemodel";
         cv::dnn::Net net = cv::dnn::readNetFromCaffe(ssd_config, ssd_weight);
         //用于选择深度学习模型的计算后端和用于选择模型计算的目标硬件设备。
         net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
         net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
         //将图像转换为blob的格式并返回blob(NCHW)格式
         /*
            参数2:0.007843：缩放因子，将像素值归一化。1/127.5 约等于 0.007843，用于将像素值从 [0, 255] 归一化到 [-1, 1]。
            参数3:Size(300, 300)：输入图像的尺寸，网络需要的输入尺寸。
            参数4:Scalar(127.5, 127.5, 127.5)：均值减去偏移量，用于标准化。由于将像素值范围缩放到 [-1, 1]，需要将均值设置为 127.5。
            参数5:交换 R 和 B 通道。
         */
         cv::Mat blobImage = cv::dnn::blobFromImage(image, 0.007843,cv::Size(300, 300),cv::Scalar(127.5, 127.5, 127.5), true, false);
         //设置模型的输入, MobileNetSSD 模型，通常输入层名称是 "data",，进行前向计算设置输出层,输出层的结果通常是detection_out
         net.setInput(blobImage, "data");
         //detection[0]:批量大小（Batch size）：通常是1，因为我们通常一次处理一张图像。
         //detection[1]:通道数（Channels）：通常是1，因为每个检测结果的属性不是按通道组织的。
         //detection[2]:高度（Height）：代表检测到的对象数量。
         //detection[3]:宽度（Width）：代表每个检测结果的属性数量。
         cv::Mat detection = net.forward("detection_out");
         //计算推理时间
         std::vector<double> layersTimings;
         double freq = cv::getTickFrequency() / 1000;//获取时钟频率，并转换为ms
         double time = net.getPerfProfile(layersTimings) / freq;//获取网络总时间，并以ms输出
         //将检测到的每个对象的属性创建成一个Mat对象
         cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
         //置信度阈值
         float confidence_threshold = ui->doubleSpinBox_CidenceThresHoldValue->value();
         for (int i = 0; i < detectionMat.rows; i++) {
             float confidence = detectionMat.at<float>(i, 2);//置信度
             if (confidence > confidence_threshold) {
                 size_t objIndex = (size_t)(detectionMat.at<float>(i, 1));
                 float tl_x = detectionMat.at<float>(i, 3) * image.cols;//左上x
                 float tl_y = detectionMat.at<float>(i, 4) * image.rows;//左上y
                 float br_x = detectionMat.at<float>(i, 5) * image.cols;//右下x
                 float br_y = detectionMat.at<float>(i, 6) * image.rows;//右下y
                 //位置矩形
                 cv::Rect object_box((int)tl_x, (int)tl_y, (int)(br_x - tl_x), (int)(br_y - tl_y));
                 rectangle(image, object_box, cv::Scalar(0, 0, 255), 2, 8, 0);
                 putText(image, cv::format(" confidence %.2f, %s", confidence, objNames[objIndex].c_str()),
                 cv::Point(tl_x - 10, tl_y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2, 8);
             }
         }

         //将Mat对象转换为QpixMap对象,然后将图像大小设置成Qlabel大小并赋值到Qlabel中去
         QDateTime currentDateTime = QDateTime::currentDateTime();
         QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
         std::unique_ptr<ImbaProcess> imageChange=std::make_unique<ImbaProcess>();
         QPixmap pix = imageChange->matToQPixmap(image);
         pix.scaled(ui->label_MachineLearnDataShow->size(),Qt::KeepAspectRatio);
         ui->label_MachineLearnDataShow->setScaledContents(true);
         ui->label_MachineLearnDataShow->setPixmap(pix);
         ui->textEdit_MachineLearnInforShow->append(QString("%1:检测完成...").arg(formattedTime));
     }
}
//FastRCNN对象检测
void machinelearn::on_checkBox_FastRcnnDection_stateChanged(int state){
    //选中则计算
    if(state==Qt::Checked){
        if(MachineLerrnSpace::VideoImageFrame.empty()){
            //加载失败，向信息区中写入信息
            QDateTime currentDateTime = QDateTime::currentDateTime();
            QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
            ui->textEdit_MachineLearnInforShow->append(QString("%1:，测试数据采集失败，加载目标为空").arg(formattedTime));
            return;
          }
         cv::Mat image=MachineLerrnSpace::VideoImageFrame.clone();
         //首先读取权重文件和描述文件
         std::string path="E:/DeskTop/SongXinFile/Qt Design/SongXinDemo/dnn/FasterRcnn/faster_rcnn_resnet50_coco_2018_01_28";
         std::string rcnn_config = path+"/frozen_inference_graph.pbtxt";
         std::string rcnn_weight = path+"/frozen_inference_graph.pb";
         cv::dnn::Net net = cv::dnn::readNetFromTensorflow(rcnn_weight,rcnn_config);
         //用于选择深度学习模型的计算后端和用于选择模型计算的目标硬件设备。
         net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
         net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
         std::vector<std::string>objectname= readClassNames(path+"/classes.txt");
         //将图像转换为blob的格式
         /*
            参数2:1.0：缩放因子，不缩放
            参数3:Size(800, 600)：输入图像的尺寸，网络需要的输入尺寸。
            参数4:Scalar(0, 0, 0)：均值减去偏移量，用于标准化。
            参数5:交换 R 和 B 通道。
         */
         cv::Mat blobImage = cv::dnn::blobFromImage(image, 1.0,cv::Size(800, 600),cv::Scalar(0, 0, 0), true, false);
         //设置模型的输入, MobileNetSSD 模型，通常输入层名称是 "data",，进行前向计算设置输出层,输出层的结果通常是detection_out
         net.setInput(blobImage);
         //detection[0]:批量大小（Batch size）：通常是1，因为我们通常一次处理一张图像。
         //detection[1]:通道数（Channels）：通常是1，因为每个检测结果的属性不是按通道组织的。
         //detection[2]:高度（Height）：代表检测到的对象数量。
         //detection[3]:宽度（Width）：代表每个检测结果的属性数量。
         cv::Mat detection = net.forward();
         //计算推理时间
         std::vector<double> layersTimings;
         double freq = cv::getTickFrequency() / 1000;//获取时钟频率，并转换为ms
         double time = net.getPerfProfile(layersTimings) / freq;//获取网络总时间，并以ms输出
         //将检测到的每个对象的属性创建成一个Mat对象
         cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
         //置信度阈值
         float confidence_threshold = ui->doubleSpinBo_FasterRcnnThreshold->value();
         for (int i = 0; i < detectionMat.rows; i++) {
             float confidence = detectionMat.at<float>(i, 2);//置信度
             if (confidence > confidence_threshold) {
                 size_t objIndex = (size_t)(detectionMat.at<float>(i, 1));
                 float tl_x = detectionMat.at<float>(i, 3) * image.cols;//左上x
                 float tl_y = detectionMat.at<float>(i, 4) * image.rows;//左上y
                 float br_x = detectionMat.at<float>(i, 5) * image.cols;//右下x
                 float br_y = detectionMat.at<float>(i, 6) * image.rows;//右下y
                 //位置矩形
                 cv::Rect object_box((int)tl_x, (int)tl_y, (int)(br_x - tl_x), (int)(br_y - tl_y));
                 rectangle(image, object_box, cv::Scalar(0, 0, 255), 2, 8, 0);
                 putText(image, cv::format(" confidence %.2f, %s", confidence, objectname[objIndex].c_str()),
                 cv::Point(tl_x - 10, tl_y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1, 8);
             }
         }

         //将Mat对象转换为QpixMap对象,然后将图像大小设置成Qlabel大小并赋值到Qlabel中去
         QDateTime currentDateTime = QDateTime::currentDateTime();
         QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
         std::unique_ptr<ImbaProcess> imageChange=std::make_unique<ImbaProcess>();
         QPixmap pix = imageChange->matToQPixmap(image);
         pix.scaled(ui->label_MachineLearnDataShow->size(),Qt::KeepAspectRatio);
         ui->label_MachineLearnDataShow->setScaledContents(true);
         ui->label_MachineLearnDataShow->setPixmap(pix);
         ui->textEdit_MachineLearnInforShow->append(QString("%1:检测完成...").arg(formattedTime));
     }

}
//YOLO对象检测
void machinelearn::on_checkBox_YOLODection_stateChanged(int state){
//    //选中则计算
//    if(state==Qt::Checked){
//        if(MachineLerrnSpace::VideoImageFrame.empty()){
//            //加载失败，向信息区中写入信息
//            QDateTime currentDateTime = QDateTime::currentDateTime();
//            QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
//            ui->textEdit_MachineLearnInforShow->append(QString("%1:，测试数据采集失败，加载目标为空").arg(formattedTime));
//            return;
//          }
//         cv::Mat image=MachineLerrnSpace::VideoImageFrame.clone();
//         //加载模型文件和配置文件
//         std::string yolov4_model  = "E:/DeskTop/SongXinFile/Qt Design/SongXinDemo/dnn/Yolo/yolov3-tiny.weights";
//         std::string yolov4_config = "E:/DeskTop/SongXinFile/Qt Design/SongXinDemo/dnn/Yolo/yolov3-tiny.cfg";
//         //读取标签文件，放入容器中
//         std::vector<std::string> classNamesVec;
//         std::ifstream classNamesFile("E:/DeskTop/SongXinFile/Qt Design/SongXinDemo/dnn/Yolo/object_detection_classes_yolov4.txt");
//         if (classNamesFile.is_open()){
//             std::string className = "";
//             while (std::getline(classNamesFile, className))
//                 classNamesVec.push_back(className);
//         }
//         classNamesFile.close();//关闭文件
//         // 加载YOLOv4
//         cv::dnn::Net net = cv::dnn::readNetFromDarknet(yolov4_config, yolov4_model);
//         std::vector<std::string> outNames = net.getUnconnectedOutLayersNames();
//         //获取输出层的名称
//         for (int i = 0; i < outNames.size(); i++){
//             ui->textEdit_MachineLearnInforShow->append(QString("输出层名称:%1").arg(outNames[i].c_str()));
//         }
//         // 设置输入
//         cv::Mat inputBlob = cv::dnn::blobFromImage(image, 1 / 255.F, cv::Size(416, 416), cv::Scalar(), true, false);
//         net.setInput(inputBlob);
//         //推理预测
//         std::vector<cv::Mat> outs;
//         net.forward(outs, outNames);

//         std::vector<cv::Rect> boxes;//检测框
//         std::vector<int> classIds;//名称ID
//         std::vector<float> confidences;//置信度
//         for (size_t i = 0; i<outs.size(); ++i){
//             // 解析与合并各输出层的预测结果
//             float* data = (float*)outs[i].data;
//             for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols){
//                 cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
//                 cv::Point classIdPoint;
//                 double confidence;
//                 minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
//                 if (confidence > 0.5){
//                     int centerX = (int)(data[0] * image.cols);
//                     int centerY = (int)(data[1] * image.rows);
//                     int width = (int)(data[2] * image.cols);
//                     int height = (int)(data[3] * image.rows);
//                     int left = centerX - width / 2;
//                     int top = centerY - height / 2;

//                     classIds.push_back(classIdPoint.x);
//                     confidences.push_back((float)confidence);
//                     boxes.push_back(cv::Rect(left, top, width, height));
//                 }
//             }
//         }

//         // 非最大抑制与输出
//         std::vector<int> indices;
//         cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.2, indices);
//         for (size_t i = 0; i < indices.size(); ++i){
//             int idx = indices[i];
//             cv::Rect box = boxes[idx];
//             std::string className = classNamesVec[classIds[idx]];
//             putText(image, className.c_str(), box.tl(), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0), 2, 8);
//             rectangle(image, box, cv::Scalar(0, 0, 255), 2, 8, 0);
//         }
//         //将Mat对象转换为QpixMap对象,然后将图像大小设置成Qlabel大小并赋值到Qlabel中去
//         QDateTime currentDateTime = QDateTime::currentDateTime();
//         QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
//         std::unique_ptr<ImbaProcess> imageChange=std::make_unique<ImbaProcess>();
//         QPixmap pix = imageChange->matToQPixmap(image);
//         pix.scaled(ui->label_MachineLearnDataShow->size(),Qt::KeepAspectRatio);
//         ui->label_MachineLearnDataShow->setScaledContents(true);
//         ui->label_MachineLearnDataShow->setPixmap(pix);
//         ui->textEdit_MachineLearnInforShow->append(QString("%1:检测完成...").arg(formattedTime));
//  }

}

//ENET图像语义分割
void postENetProcess(cv::Mat &score, cv::Mat &mask) {
    const int rows = score.size[2];//高
    const int cols = score.size[3];//宽
    const int chns = score.size[1];//类别通道
    cv::Mat maxVal = cv::Mat::zeros(rows, cols, CV_32FC1);//创建一个纯黑值都为0的最大值得分图像
    for (int ch = 1; ch < chns; ch++)//遍历类别
    {
        for (int row = 0; row < rows; row++)//遍历行
        {
            const float *ptrScore = score.ptr<float>(0, ch, row);//得分的每个类的行指针
            uchar *ptrMaxCl = mask.ptr<uchar>(row);//结果行的的指针
            float *ptrMaxVal = maxVal.ptr<float>(row);//最大值行的指针
            for (int col = 0; col < cols; col++)//遍历列
            {
                if (ptrScore[col] > ptrMaxVal[col])//如果得分矩阵行的每个像素值大于maxVal,那么就直接赋值给
                {
                    ptrMaxVal[col] = ptrScore[col];//记录最高分
                    ptrMaxCl[col] = (uchar)ch;//最高分的行
                }
            }
        }
    }
    normalize(mask, mask, 0, 255, cv::NORM_MINMAX);//将结果归一化到0~255
    applyColorMap(mask, mask, cv::COLORMAP_HSV);//将图像转换为HSV色彩空间，以便结果更加直观
}
void machinelearn::on_checkBox_EnetSemanticOrientation_stateChanged(int state){
    //选中则计算
    if(state==Qt::Checked){
        if(MachineLerrnSpace::VideoImageFrame.empty()){
            //加载失败，向信息区中写入信息
            QDateTime currentDateTime = QDateTime::currentDateTime();
            QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
            ui->textEdit_MachineLearnInforShow->append(QString("%1:，测试数据采集失败，加载目标为空").arg(formattedTime));
            return;
          }
         cv::Mat image=MachineLerrnSpace::VideoImageFrame.clone();
         //首先读取权重文件和描述文件
         std::string path="E:/DeskTop/SongXinFile/Qt Design/SongXinDemo/dnn/ENet";
         std::string rcnn_config = path+"/Enet-model-best.net";
         cv::dnn::Net net = cv::dnn::readNetFromTorch(rcnn_config);
         //用于选择深度学习模型的计算后端和用于选择模型计算的目标硬件设备。
         net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
         net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
         std::vector<std::string>objectname= readClassNames(path+"/enet-classes.txt");
         //将图像转换为blob的格式
         cv::Mat blobImage = cv::dnn::blobFromImage(image, 0.00392,cv::Size(512, 256),cv::Scalar(0, 0, 0), true, false);//scaler:0.00392是将结果归一化为0~1
         //设置模型的输入
         net.setInput(blobImage);
         //推理运行
         cv::Mat score=net.forward();
         std::vector<double> layersTimes;
         double freq = cv::getTickFrequency() / 1000;
         double t = net.getPerfProfile(layersTimes) / freq;
         std::string label = cv::format("Inference time: %.2f ms", t);
         //解析输出与显示
         cv::Mat mask = cv::Mat::zeros(256, 512, CV_8UC1);
         postENetProcess(score, mask);
         cv::resize(mask, mask, image.size());
         cv::Mat dst;
         //将得分图形与原图加权组合
         addWeighted(image, 0.8, mask, 0.2, 0, dst);//0.8是image的权重 0.2是Mask的权重，0:beta,没有偏移，最终生成dst结果
         //将Mat对象转换为QpixMap对象,然后将图像大小设置成Qlabel大小并赋值到Qlabel中去
         QDateTime currentDateTime = QDateTime::currentDateTime();
         QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
         std::unique_ptr<ImbaProcess> imageChange=std::make_unique<ImbaProcess>();
         QPixmap pix = imageChange->matToQPixmap(dst);
         pix.scaled(ui->label_MachineLearnDataShow->size(),Qt::KeepAspectRatio);
         ui->label_MachineLearnDataShow->setScaledContents(true);
         ui->label_MachineLearnDataShow->setPixmap(pix);
         ui->textEdit_MachineLearnInforShow->append(QString("推理时间:%1ms").arg(t));
         ui->textEdit_MachineLearnInforShow->append(QString("%1:检测完成...").arg(formattedTime));
     }

}
//风格迁移
void machinelearn::on_checkBox_StyleTransfer_stateChanged(int state){
    //选中则计算
    if(state==Qt::Checked){
        if(MachineLerrnSpace::VideoImageFrame.empty()){
            //加载失败，向信息区中写入信息
            QDateTime currentDateTime = QDateTime::currentDateTime();
            QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
            ui->textEdit_MachineLearnInforShow->append(QString("%1:，测试数据采集失败，加载目标为空").arg(formattedTime));
            return;
          }
         cv::Mat image=MachineLerrnSpace::VideoImageFrame.clone();
         //权重文件向量空间
         std::vector<std::string>WeightFile={"candy.t7","composition_vii.t7","feathers.t7","la_muse.t7","mosaic.t7","starry_night.t7","the_scream.t7","the_wave.t7","udnie.t7"};
         //首先读取权重文件和描述文件
         std::string path="E:/DeskTop/SongXinFile/Qt Design/SongXinDemo/dnn/style/";
         cv::dnn::Net net = cv::dnn::readNetFromTorch(path+WeightFile[ui->spinBox_Flage->value()]);
         //用于选择深度学习模型的计算后端和用于选择模型计算的目标硬件设备。
         net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
         net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

         //将图像转换为blob的格式
         cv::Mat blobImage = cv::dnn::blobFromImage(image, 1.0,image.size(),cv::Scalar(103.939, 116.779, 123.68), false, false);
         //设置模型的输入
         net.setInput(blobImage);
         //推理运行
         cv::Mat out=net.forward();//推理输出一个3维的数据，刚好就是一张图像
         std::vector<double> layersTimes;
         double freq = cv::getTickFrequency() / 1000;
         double t = net.getPerfProfile(layersTimes) / freq;
         std::string label = cv::format("Inference time: %.2f ms", t);
         //解析输出与显示
         int ch = out.size[1];//通道
         int h  = out.size[2];//高度
         int w  = out.size[3];//宽度
         cv::Mat result = cv::Mat::zeros(cv::Size(w, h), CV_32FC3);
         float* data = out.ptr<float>();
         for (int c = 0; c < ch; c++) {
             for (int row = 0; row < h; row++) {
                 for (int col = 0; col < w; col++) {
                     result.at<cv::Vec3f>(row, col)[c] = *data++;
                 }
             }
         }
         // 整合结果输出
         add(result, cv::Scalar(103.939, 116.779, 123.68), result);//网络中减去的均值（mean subtraction）添加回去，从而恢复图像的颜色信息。
         normalize(result, result, 0, 255.0, cv::NORM_MINMAX);//归一化到0~255
         // 中值滤波
         medianBlur(result, result, 5);
         result.convertTo(result,CV_8UC3);//转换为8U的数据类型,否则Qpix无法识别
         //将Mat对象转换为QpixMap对象,然后将图像大小设置成Qlabel大小并赋值到Qlabel中去
         QDateTime currentDateTime = QDateTime::currentDateTime();
         QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
         std::unique_ptr<ImbaProcess> imageChange=std::make_unique<ImbaProcess>();
         QPixmap pix = imageChange->matToQPixmap(result);
         pix.scaled(ui->label_MachineLearnDataShow->size(),Qt::KeepAspectRatio);
         ui->label_MachineLearnDataShow->setScaledContents(true);
         ui->label_MachineLearnDataShow->setPixmap(pix);
         ui->textEdit_MachineLearnInforShow->append(QString("推理时间:%1ms").arg(t));
         ui->textEdit_MachineLearnInforShow->append(QString("%1:检测完成...").arg(formattedTime));
     }

}
//场景文字识别
/*参数1:Scores:输出模型的得分矩阵,表示每个字符概率
 *参数2:geometry模型输出的几何信息矩阵,提供每个位置的几何特征，用于计算字符的旋转矩形
 *参数3:scoreThresh最小得分阈值，仅保留得分高于此阈值的检测
 *参数4:detections: 检测框的输出结果，存储为 cv::RotatedRect 对象的向量。
 *参数5:confidences: 每个检测框的置信度，存储为浮点数的向量。
 */
void decode(const cv::Mat& scores, const cv::Mat& geometry, float scoreThresh,
    std::vector<cv::RotatedRect>& detections, std::vector<float>& confidences){
    detections.clear();
    //确保 scores 和 geometry 矩阵的维度和大小符合预期。得分矩阵和几何信息矩阵都应该是 4 维的，具体维度应符合模型输出的格式。
    CV_Assert(scores.dims == 4); CV_Assert(geometry.dims == 4); CV_Assert(scores.size[0] == 1);
    CV_Assert(geometry.size[0] == 1); CV_Assert(scores.size[1] == 1); CV_Assert(geometry.size[1] == 5);
    CV_Assert(scores.size[2] == geometry.size[2]); CV_Assert(scores.size[3] == geometry.size[3]);
    //height 和 width 表示模型输出的特征图的尺寸。
    const int height = scores.size[2];
    const int width = scores.size[3];
    //遍历每个位置 (x, y)。如果得分小于阈值 scoreThresh，则跳过。
    for (int y = 0; y < height; ++y){
        const float* scoresData = scores.ptr<float>(0, 0, y);
        const float* x0_data = geometry.ptr<float>(0, 0, y);
        const float* x1_data = geometry.ptr<float>(0, 1, y);
        const float* x2_data = geometry.ptr<float>(0, 2, y);
        const float* x3_data = geometry.ptr<float>(0, 3, y);
        const float* anglesData = geometry.ptr<float>(0, 4, y);
        for (int x = 0; x < width; ++x){
            float score = scoresData[x];
            if (score < scoreThresh)
                continue;
            float offsetX = x * 4.0f, offsetY = y * 4.0f;
            float angle = anglesData[x];
            float cosA = std::cos(angle);
            float sinA = std::sin(angle);
            float h = x0_data[x] + x2_data[x];
            float w = x1_data[x] + x3_data[x];
            //计算旋转矩形的中心点 offset，以及两个对角点 p1 和 p3。
            cv::Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
            cv::Point2f p1 = cv::Point2f(-sinA * h, -cosA * h) + offset;
            cv::Point2f p3 = cv::Point2f(-cosA * w, sinA * w) + offset;
            //使用中心点、宽度、高度和角度创建 cv::RotatedRect 对象。
            //将旋转矩形 r 和其置信度 score 存储到输出向量 detections 和 confidences 中。
            cv::RotatedRect r(0.5f * (p1 + p3), cv::Size2f(w, h), -angle * 180.0f / (float)CV_PI);
            detections.push_back(r);
            confidences.push_back(score);
        }
    }
}
void machinelearn::on_checkBox_CharacterRecognition_stateChanged(int state){
    //选中则计算
    if(state==Qt::Checked){
        if(MachineLerrnSpace::VideoImageFrame.empty()){
            //加载失败，向信息区中写入信息
            QDateTime currentDateTime = QDateTime::currentDateTime();
            QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
            ui->textEdit_MachineLearnInforShow->append(QString("%1:，测试数据采集失败，加载目标为空").arg(formattedTime));
            return;
          }
         cv::Mat image=MachineLerrnSpace::VideoImageFrame.clone();
         float confThreshold = ui->doubleSpinBox_ConfidenceThreshold->value();//置信度阈值
         float nmsThreshold = 0.4;//非最大抑制的阈值
         int inpWidth = 320;//输入图像宽
         int inpHeight = 320;//输入图像高
         //首先读取权重文件和描述文件
         std::string path="E:/DeskTop/SongXinFile/Qt Design/SongXinDemo/dnn/EAST/";
         cv::dnn::Net net = cv::dnn::readNet(path+"frozen_east_text_detection.pb");
         //用于选择深度学习模型的计算后端和用于选择模型计算的目标硬件设备。
         net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
         net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
         //获取输入层的名称后续推理使用
         std::vector<std::string> outNames = net.getUnconnectedOutLayersNames();
         //将图像转换为blob的格式
         cv::Mat blobImage = cv::dnn::blobFromImage(image, 1.0,cv::Size(inpWidth,inpHeight),cv::Scalar(123.68, 116.78, 103.94), true, false);
         //设置模型的输入
         net.setInput(blobImage);
         //推理运行
         std::vector<cv::Mat> outs;
         net.forward(outs,outNames);//推理输出一个2维的数据
         //解析输出与显示
         cv::Mat geometry  = outs[0];//几何信息
         cv::Mat scores    = outs[1];//置信度分数
         // 解析输出
         std::vector<cv::RotatedRect> boxes;
         std::vector<float> confidences;
         decode(scores, geometry, confThreshold, boxes, confidences);
         // 非最大抑制
         std::vector<int> indices;
         /*
            参数1:输入的检测框列表。
            参数2:对应每个检测框的置信度分数列表
            参数3:置信度阈值
            参数4:NMS阈值，用于决定哪些检测框需要被抑制，也就是两个框的重叠度，如果超过这个阈值，其中一个就会被抑制掉
            参数5:输出参数,经过非最大抑制后的检测框索引列表，这个列表在Boxs中被保留的框的索引
         */
         cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
         // 绘制检测框
         cv::Point2f ratio((float)image.cols / inpWidth, (float)image.rows / inpHeight);//计算图像缩放比例
         for (size_t i = 0; i < indices.size(); ++i){//遍历所有保留的检测框。
             //找到旋转矩形框
             cv::RotatedRect& box = boxes[indices[i]];
            //矩形框的四个坐标点
             cv::Point2f vertices[4];
             box.points(vertices);
             for (int j = 0; j < 4; ++j){//遍历检测框的四个顶点。
                 vertices[j].x *= ratio.x;//根据缩放比例 ratio 将顶点的坐标从模型输入图像的坐标系转换到原始图像的坐标系。
                 vertices[j].y *= ratio.y;
             }
             for (int j = 0; j < 4; ++j)//遍历四个顶点,并绘制直线。
                 line(image, vertices[j], vertices[(j + 1) % 4], cv::Scalar(255, 0, 0), 2);
         }

         // 显示信息
         std::vector<double> layersTimes;
         double freq = cv::getTickFrequency() / 1000;
         double t = net.getPerfProfile(layersTimes) / freq;
         std::string label = cv::format("Inference time: %.2f ms", t);         
         //将Mat对象转换为QpixMap对象,然后将图像大小设置成Qlabel大小并赋值到Qlabel中去
         QDateTime currentDateTime = QDateTime::currentDateTime();
         QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
         std::unique_ptr<ImbaProcess> imageChange=std::make_unique<ImbaProcess>();
         QPixmap pix = imageChange->matToQPixmap(image);
         pix.scaled(ui->label_MachineLearnDataShow->size(),Qt::KeepAspectRatio);
         ui->label_MachineLearnDataShow->setScaledContents(true);
         ui->label_MachineLearnDataShow->setPixmap(pix);
         ui->textEdit_MachineLearnInforShow->append(QString("推理时间:%1ms").arg(t));
         ui->textEdit_MachineLearnInforShow->append(QString("%1:检测完成...").arg(formattedTime));
     }

}
//实时人脸检测
void machinelearn::on_checkBox_RealFaceDection_stateChanged(int state){
    //选中则计算
    if(state==Qt::Checked){
        bool onlyonce=1;
        if(MachineLerrnSpace::VideoImageFrame.empty()){
            //加载失败，向信息区中写入信息
            QDateTime currentDateTime = QDateTime::currentDateTime();
            QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
            ui->textEdit_MachineLearnInforShow->append(QString("%1:，测试数据采集失败，加载目标为空").arg(formattedTime));
            return;
          }
         cv::Mat image=MachineLerrnSpace::VideoImageFrame.clone();
         //首先读取权重文件和描述文件
         std::string model ="E:/DeskTop/SongXinFile/Qt Design/SongXinDemo/dnn/FaceDection/res10_300x300_ssd_iter_140000.caffemodel";
         std::string config="E:/DeskTop/SongXinFile/Qt Design/SongXinDemo/dnn/FaceDection/deploy.prototxt";
         //加载模型与配置文件
         cv::dnn::Net net=cv::dnn::readNetFromCaffe(config,model);
         //用于选择深度学习模型的计算后端和用于选择模型计算的目标硬件设备。
         net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
         net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
         //将图像转换为blob的格式
         cv::Mat blobImage = cv::dnn::blobFromImage(image, 1.0,cv::Size(300,300),cv::Scalar(104.0, 177.0, 123.0), false, false);
         //设置模型的输入
         net.setInput(blobImage,"data");
         //推理运行
         cv::Mat detection =net.forward("detection_out");//推理输出一个2维的数据
         cv::Mat dectionMat(detection.size[2],detection.size[3],CV_32F,detection.ptr<float>());//NCHM
         //遍历每个检测到的对象
         for (int i = 0; i < dectionMat.rows; i++){
             float confidence = dectionMat.at<float>(i, 2);
             //置信度大于设定值则绘制
             if (confidence > ui->doubleSpinBox_facedectionThreshold->value()){
                 int x1 = static_cast<int>(dectionMat.at<float>(i, 3) * image.cols);//0:Batch 1:Class 2:confidence 3:Left 4:Top 5:Right 6:Bottom 并调整比例
                 int y1 = static_cast<int>(dectionMat.at<float>(i, 4) * image.rows);
                 int x2 = static_cast<int>(dectionMat.at<float>(i, 5) * image.cols);
                 int y2 = static_cast<int>(dectionMat.at<float>(i, 6) * image.rows);
                 //绘制检测结果
                 cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),2, 8);
                 cv::putText(image,cv::format("%f",confidence),cv::Point(x1, y1-3),cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 1, 8);
             }
         }
         // 显示信息
         std::vector<double> layersTimes;
         double freq = cv::getTickFrequency() / 1000;
         double t = net.getPerfProfile(layersTimes) / freq;
         std::string label = cv::format("Inference time: %.2f ms", t);
         //将Mat对象转换为QpixMap对象,然后将图像大小设置成Qlabel大小并赋值到Qlabel中去
         QDateTime currentDateTime = QDateTime::currentDateTime();
         QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
         std::unique_ptr<ImbaProcess> imageChange=std::make_unique<ImbaProcess>();
         QPixmap pix = imageChange->matToQPixmap(image);
         pix.scaled(ui->label_MachineLearnDataShow->size(),Qt::KeepAspectRatio);
         ui->label_MachineLearnDataShow->setScaledContents(true);
         ui->label_MachineLearnDataShow->setPixmap(pix);
         ui->textEdit_MachineLearnInforShow->append(QString("推理时间:%1ms").arg(t));
         ui->textEdit_MachineLearnInforShow->append(QString("%1:检测完成...").arg(formattedTime));
    }
}

//对象检测
void machinelearn::on_checkBox_ObjectDetection_stateChanged(int state){
    //选中则计算
    if(state==Qt::Checked){
        if(MachineLerrnSpace::VideoImageFrame.empty()){
            //加载失败，向信息区中写入信息
            QDateTime currentDateTime = QDateTime::currentDateTime();
            QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
            ui->textEdit_MachineLearnInforShow->append(QString("%1:，测试数据采集失败，加载目标为空").arg(formattedTime));
            return;
          }
         cv::Mat image=MachineLerrnSpace::VideoImageFrame.clone();
         // 颜色表
         std::vector<cv::Scalar> colors;
         colors.push_back(cv::Scalar(0, 255, 0));
         colors.push_back(cv::Scalar(0, 255, 255));
         colors.push_back(cv::Scalar(255, 255, 0));
         colors.push_back(cv::Scalar(255, 0, 0));
         colors.push_back(cv::Scalar(0, 0, 255));
         int64 start = cv::getTickCount();
         //模型文件以及标签文件
         std::string model="E:/DeskTop/Learn File/SoftWare/yolov5-6.1/yolov5s.onnx";
         std::vector<std::string>className=readClassNames("E:/DeskTop/Learn File/SoftWare/yolov5-6.1/CocoClassName.txt");
         //模型加载,与设置加载硬件
         auto net=cv::dnn::readNet(model);
         net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
         net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
         //图像的预处理,将图像防放置在一个正方形块中，防止因为缩放图像导致图像畸变
         int width =image.cols;
         int height=image.rows;
         int max=std::max(width,height);//取出长宽中的最长边,建立模板
         cv::Mat black = cv::Mat::zeros(cv::Size(max, max), CV_8UC3);
         cv::Rect roi(0,0,width,height);//建立ROI区域
         image.copyTo(black(roi));//将ROI区域拷贝到模板上
         //计算缩放比率,后面将要进行还原
         float x_factor = image.cols / 640.0f;
         float y_factor = image.rows / 640.0f;
         //图像做输入处理
         cv::Mat blob=cv::dnn::blobFromImage(image,1 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);
         net.setInput(blob);
         cv::Mat preds = net.forward();
         // 后处理, 1x25200x85
         cv::Mat det_output(preds.size[1], preds.size[2], CV_32F, preds.ptr<float>());//将数据整合到一个Mat对象中
         float confidence_threshold = 0.5;//置信度阈值
         std::vector<cv::Rect> boxes;//检测框
         std::vector<int> classIds;//类别ID容器
         std::vector<float> confidences;//置信度容器
         for (int i = 0; i < det_output.rows; i++) {//det_output.rows等价于preds.size[1]等价于25200
             float confidence = det_output.at<float>(i, 4);
             qDebug()<<confidence;
             if (confidence < 0) {//小于置信度阈值的检测框直接忽略
                 continue;
             }
             cv::Mat classes_scores = det_output.row(i).colRange(5, preds.size[2]);//当前检测框的类别分数向量,从第 5 列到最后一列,也就是80个类别的分别得分也就是将后80列的数据存储到新的Mat中
             cv::Point classIdPoint;
             double score;
             minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);//函数找到类别分数中的最大值 score 以及对应的类别 ID classIdPoint.x,这个classIdPoint.x就是对应类别ID
             //类别得分
             if (score > 0){
                 //检测结果Box坐标
                 float cx = det_output.at<float>(i, 0);//中心坐标x
                 float cy = det_output.at<float>(i, 1);//中心坐标y
                 float ow = det_output.at<float>(i, 2);
                 float oh = det_output.at<float>(i, 3);
                 //检测框的中心坐标 cx, cy转换为左上角坐标 宽度 ow 和高度 oh。这些值转换为像素坐标，乘以 x_factor 和 y_factor（用于将模型预测的坐标转换为实际图像坐标）。
                 int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
                 int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
                 int width = static_cast<int>(ow * x_factor);
                 int height = static_cast<int>(oh * y_factor);
                 cv::Rect box;
                 box.x = x;
                 box.y = y;
                 box.width  = width;
                 box.height = height;
                 //检测框、类别ID、得分放置到容器当中
                 boxes.push_back(box);//检测框
                 classIds.push_back(classIdPoint.x);//类别ID
                 confidences.push_back(score);//得分
             }
         }
         //NMS非最大抑制()
         std::vector<int> indexes;
         cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.50, indexes);//执行非最大抑制，得出最终的检测框容器
         for (size_t i = 0; i < indexes.size(); i++) {//遍历绘制检测框
             //类别索引
             int index = indexes[i];
             int idx = classIds[index];
             //每5个类别绘制一个颜色
             cv::rectangle(image, boxes[index], colors[idx % 5], 2, 8);
             cv::rectangle(image, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
             cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(255, 255, 255), -1);
             cv::putText(image, className[idx], cv::Point(boxes[index].tl().x, boxes[index].tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
         }

         float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
         putText(image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);

         //将Mat对象转换为QpixMap对象,然后将图像大小设置成Qlabel大小并赋值到Qlabel中去
         QDateTime currentDateTime = QDateTime::currentDateTime();
         QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
         std::unique_ptr<ImbaProcess> imageChange=std::make_unique<ImbaProcess>();
         QPixmap pix = imageChange->matToQPixmap(image);
         pix.scaled(ui->label_MachineLearnDataShow->size(),Qt::KeepAspectRatio);
         ui->label_MachineLearnDataShow->setScaledContents(true);
         ui->label_MachineLearnDataShow->setPixmap(pix);
         ui->textEdit_MachineLearnInforShow->append(QString("推理时间:%1ms").arg(1));
         ui->textEdit_MachineLearnInforShow->append(QString("%1:检测完成...").arg(formattedTime));
    }
}
//自定义对象检测
void machinelearn::on_checkBox_CustmerDetection_stateChanged(int state){

}
//onnxruntime推理部署
void machinelearn::on_checkBox_onnxruntiome_stateChanged(int state){

}
//openvino推理部署
void machinelearn::on_checkBox_openvino_stateChanged(int state){

}
//TrnsorRT推理部署
void machinelearn::on_checkBox_TensorRt_stateChanged(int state){

}

//停止数据分析
void machinelearn::on_toolButton_StopVideo_clicked(){
   MachineLerrnSpace::StarBit=0;//停止加载视频
}
