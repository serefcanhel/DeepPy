# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 18:56:49 2021

@author: SEREF
"""
from PyQt5.QtWidgets import*
from reset_password_python import Ui_MainWindow

class reset(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("RESET PASSWORD")
        
        