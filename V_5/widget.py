# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 15:20:31 2021

@author: SEREF
"""
from PyQt5.QtWidgets import*
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

class widget(QWidget):
    def __init__(self, parent= None):
        
        QWidget.__init__(self,parent)
        
        self.canvas=FigureCanvas(Figure())
        
        vertical_layout=QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)