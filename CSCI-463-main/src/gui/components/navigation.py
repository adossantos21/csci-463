"""
File Name: navigation.py
Purpose: Constrains action buttons in the GUI such as Back, Next, Open Input File, or Save Generated Output
Author: Arteom Katkov
Documented: 05/10/2024
"""
import enum

from loguru import logger as log
from PyQt5.QtWidgets import QPushButton, QHBoxLayout

class ButtonTypes(enum.Enum):
    BACK_BUTTON = 0
    ACTION_BUTTON = 1
    NEXT_BUTTON = 2

class NavigationBar:
    def __init__(self, buttonCb = None) -> None:
        self.buttonCb = buttonCb

        self._layout = QHBoxLayout()
        
        self._btn_back = QPushButton('Back')
        self._btn_back.clicked.connect(self._onBackButtonClick)
        self._layout.addWidget(self._btn_back)

        self._btn_action = QPushButton('Open PDF')
        self._btn_action.clicked.connect(self._onActionButtonClick)
        self._layout.addWidget(self._btn_action)

        self._btn_next = QPushButton('Next')
        self._btn_next.clicked.connect(self._onNextButtonClick)
        self._layout.addWidget(self._btn_next)


    def getLayout(self):
        return self._layout
    
    def getActionButton(self):
        return self._btn_action

    def setButtonClickCb(self, cb):
        self.buttonCb = cb

    def _onBackButtonClick(self):
        if self.buttonCb is not None:
            self.buttonCb(ButtonTypes.BACK_BUTTON)

    def _onActionButtonClick(self):
        if self.buttonCb is not None:
            self.buttonCb(ButtonTypes.ACTION_BUTTON)
    
    def _onNextButtonClick(self):
        if self.buttonCb is not None:
            self.buttonCb(ButtonTypes.NEXT_BUTTON)

    def _onHide(self):
        self._btn_action.hide()
        self._btn_back.hide()
        self._btn_next.hide()

    def _onShow(self):
        self._btn_action.show()
        self._btn_back.show()
        self._btn_next.show()