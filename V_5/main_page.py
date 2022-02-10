# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 23:48:22 2021

@author: SEREF
"""
from login_python import Ui_Form
from PyQt5.QtWidgets import*

from register import register
import sqlite3
from forget_password import forget_password
from design import design
import webbrowser

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.ui=Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle("Main Menu")
        
        #PAGE DEFINITIONS
        self.register=register()
        self.design=design()
        self.forget_password=forget_password()
        
        #PSUHBUTTONS DEFINITIONS
        self.ui.pushButton.clicked.connect(self.login)
        self.ui.pushButton_2.clicked.connect(self.register_page)
        self.ui.pushButton_4.clicked.connect(self.password_page)
        self.ui.pushButton_3.clicked.connect(self.design_page)
        self.ui.pushButton_5.clicked.connect(self.change)
        self.ui.pushButton_6.clicked.connect(self.web)
        
    def web(self):
        webbrowser.open("https://www.ikcu.edu.tr/") #OPEN SCHOOLS WEB PAGE
    def change(self,state):
        self.ui.lineEdit_2.setEchoMode(QLineEdit.EchoMode.Normal) #CHANGES MODE OF PASSWORD SECTION

    def design_page(self):
        self.design.show() #SHOWS DESIGN PAGE
    def password_page(self):
        self.forget_password.show() #SHOWS FORGET PASSWORD PAGE
    def register_page(self):
        self.register.show() # SHOWS REGISTER PAGE
    def login(self):
        self.baglanti_olustur() #CREATES CONNECTION WITH SQLITE DATABASE
        kullanici_adi = self.ui.lineEdit.text() #GET USERNAME FROM USER
        parola = self.ui.lineEdit_2.text()#GET PASSWORD FROM USER
        self.cursor.execute("Select *From üyeler where kullanici_adi = ? and parola = ?", (kullanici_adi, parola)) #CHECKS USERNAME AND PASSWORD FROM DATABASE
        data = self.cursor.fetchall()
        if len(data) != 0:
            self.design.show() #SHOWS DESIGN PAGE
        else:
            text="Your username or password is wrong"
            self.error(text)
        self.baglanti.close()
    def baglanti_olustur(self):
        self.baglanti = sqlite3.connect("users.db") # CONNECT SQLITE DATABASE
        self.cursor = self.baglanti.cursor()
        self.cursor.execute("Create Table If not exists üyeler (email TEXT,kullanici_adi TEXT,parola TEXT)")
        self.baglanti.commit()   
    def error(self,text):
        msg = QMessageBox()#CREATE MESSAGE BOX
        msg.setIcon(QMessageBox.Critical)#GIVE AN ICON TO MESSAGE BOX
        msg.setText(text)#SET TEXT OF MESSAGE BOX
        msg.setWindowTitle("Error")#GIVE TITLE TO MESSAGE BOX
        msg.exec_()