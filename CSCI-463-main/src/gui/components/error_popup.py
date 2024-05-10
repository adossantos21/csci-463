"""
File Name: error.popup.py
Purpose: Displays error for incorrect screen transitions
Author: Arteom Katkov
Documented: 05/10/2024
"""
from PyQt5.QtWidgets import QDialogButtonBox, QVBoxLayout, QLabel, QDialog

class ErrorPopup(QDialog):
    def __init__(self, title: str, message: str, parent=None) -> None:
        super().__init__(parent)

        self.setWindowTitle(title)

        QBtn = QDialogButtonBox.Ok # | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)

        self.layout = QVBoxLayout()
        message = QLabel(message)
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
