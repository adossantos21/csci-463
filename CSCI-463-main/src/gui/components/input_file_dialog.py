import os
import PyPDF2
from typing import Optional, Union
from loguru import logger as log
from PyQt5.QtWidgets import QFileDialog, QDialog, QWidget

class InputFileDialog(QDialog):
    def __init__(self, parent: Optional[QWidget], caption: str, file_types: str) -> None:
        super().__init__(parent)

        self.file_dialog = QFileDialog()
        self._file_name, _ = self.file_dialog.getOpenFileName(self, 
            caption=caption, 
            directory='', 
            filter=file_types)
        self._file_text = ""
        
        _, file_type = os.path.splitext(self._file_name)
        if file_type == ".txt":
            self._parseTextInput()
        elif file_type == ".pdf":
            self._parsePDFInput()
        else:
            log.error("Unknown file type was opened! Not parsing...")

    def getOpenedFileName(self) -> str:
        return self._file_name
    
    def getOpenedFileContents(self) -> str:
        return self._file_text
    
    def _parsePDFInput(self):
        try:
            with open(self._file_name, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    self._file_text += page.extract_text()
                
        except Exception as e:
            log.error(f"Error Parsing PDF File: {str(e)}")
            self._file_text = ""

    def _parseTextInput(self):
        try:
            with open(self._file_name, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    self._file_text += line

        except Exception as e:
            log.error(f"Error Parsing Text File: {str(e)}")
            self._file_text = ""
