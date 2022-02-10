# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 23:47:51 2021

@author: SEREF
"""
from PyQt5.QtWidgets import*
from main_page import Main


    

app=QApplication([])
window=Main()
window.show()

app.exec_()
