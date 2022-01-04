import sys 
import os
import shutil
from os.path import exists
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QIcon  
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from load_render import *

class Canvas(QtWidgets.QLabel):

    def __init__(self):
        super().__init__()
        self.pixmap_ = QtGui.QPixmap(600,600) 
        self.pixmap_.fill() 
        self.setPixmap(self.pixmap_)


        self.last_x, self.last_y = None, None
        self.pen_color = QtGui.QColor('#003f00')
        self.pen_width = 70

    def set_pen_color(self, c):
        self.pen_color = QtGui.QColor(c)
        if(c=="#0000ff" or c=="#ff0000"):
            self.pen_width = 4


    def mouseMoveEvent(self, e):
        if self.last_x is None: # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return # Ignore the first time.

        painter = QtGui.QPainter(self.pixmap())
        p = painter.pen()
        p.setWidth(self.pen_width)
        p.setColor(self.pen_color)
        painter.setPen(p)
        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.pixmap_.fill() 
        self.setPixmap(self.pixmap_)

    def save_canvas(self):
        if(exists('temp')==False):
            os.mkdir('temp')
        self.pixmap().save('temp/test.png')
        

        print('started saving')
        img = cv2.imread("temp/test.png") 
        img = cv2.resize(img,(256,256))
        test = np.zeros((256,512,3))
        test[:,:256,:] = img

        cv2.imwrite("temp/test.png",test)
        print('loading')
        dataset = TerrainDataset(root = "./",#config['exp_params']['data_path'],
                                train=False,
                                hide_green=config['exp_params']['hide_green'],
                                norm=config['exp_params']['norm'])

        sample_dataloader = DataLoader(dataset,
                                batch_size= 1,
                                num_workers=config['exp_params']['n_workers'],
                                shuffle = True,
                                drop_last=False)
        print('predicting')
        for ip, op,_ in sample_dataloader:
            display(ip,op)
        
            break
        shutil.rmtree('temp')
        print('done')
        # cv2.imwrite("test2.png",test)




COLORS = [
'#003f00', '#007f00','#00bf00','#00ff00','#0000ff','#ff0000', '#ffffff',
]


class QPaletteButton(QtWidgets.QPushButton):

    def __init__(self, color):
        super().__init__()
        self.setFixedSize(QtCore.QSize(24,24))
        self.color = color
        self.setStyleSheet("background-color: %s;" % color)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.canvas = Canvas()
        w = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout()
        w.setLayout(l)
        
        hbox = QHBoxLayout()
        self.initUI(hbox)
        l.addLayout(hbox)

        
        l.addWidget(self.canvas)

        palette = QtWidgets.QHBoxLayout()
        self.add_palette_buttons(palette)
        l.addLayout(palette)

        self.setCentralWidget(w)

    def add_palette_buttons(self, layout):
        for c in COLORS:
            b = QPaletteButton(c)
            b.pressed.connect(lambda c=c: self.canvas.set_pen_color(c))
            layout.addWidget(b)

    def initUI(self,hbox):

        btn1 = QPushButton(QIcon('clear.png'), 'Clear', self)
        btn2 = QPushButton(QIcon('save.png'), 'Save', self)
        btn3 = QPushButton(QIcon('exit.png'), 'Exit', self)

        hbox.addWidget(btn1)
        hbox.addWidget(btn2)
        hbox.addWidget(btn3)
        hbox.addStretch(0.1)

        self.setLayout(hbox)
        btn1.clicked.connect(self.canvas.clear_canvas)
        btn2.clicked.connect(self.canvas.save_canvas)
        btn3.clicked.connect(self.exit_app)

        label = QLabel("Thickness")
        label.setFont(QtGui.QFont("Sanserif", 12))
        hbox.addWidget(label)


        mySlider = QSlider(Qt.Horizontal, self)
        mySlider.setGeometry(300, 300, 100, 250) 
        mySlider.setMinimum(4)
        mySlider.setMaximum(70)

        mySlider.valueChanged[int].connect(self.change_pen_width)

        hbox.addWidget(mySlider)

        self.move(300, 300)
        self.setWindowTitle('Interface')
        self.show()

    def change_pen_width(self,value): 
        self.canvas.pen_width = value


    def exit_app(self):
        QCoreApplication.instance().quit() 



if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
