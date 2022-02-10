# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 18:41:31 2021

@author: SEREF
"""
import smtplib, ssl
from PyQt5.QtWidgets import*
from forget_password_python import Ui_MainWindow
from random import randint
from reset_password import reset
from PyQt5 import QtCore
import time
class forget_password(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("FORGET PASSWORD")
        
        self.ui.pushButton_2.clicked.connect(self.send_code)
        self.ui.pushButton_3.clicked.connect(self.verify)
        
        self.reset=reset()
        
    def send_code(self):
        port = 465  # For SSL
        smtp_server = "smtp.gmail.com"
        sender_email = "serefcanhel@gmail.com"  # Enter your address
        receiver_email = self.ui.lineEdit_2.text()  # Enter receiver address
        password = "cancan2122646"
        self.num=randint(100000,999999)
        self.message ="Verification code is "+ str(self.num)
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, self.message)
    def verify(self):
        if self.num == int(self.ui.lineEdit_3.text()):
            self.reset.show()
        else:
            text="Verification code is wrong. Please try again"
            self.error(text)
        
    def error(self,text):
        msg = QMessageBox()#CREATE MESSAGE BOX
        msg.setIcon(QMessageBox.Critical)#GIVE AN ICON TO MESSAGE BOX
        msg.setText(text)#SET TEXT OF MESSAGE BOX
        msg.setWindowTitle("Error")#GIVE TITLE TO MESSAGE BOX
        msg.exec_()