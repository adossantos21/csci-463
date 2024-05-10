import os
import PyPDF2
from typing import Optional, Union

from loguru import logger as log
from PyQt5.QtWidgets import QFileDialog, QDialog, QWidget

class OutputFileDialog(QDialog):
    def __init__(self, parent: Optional[QWidget], caption: str, file_types: str, file_text: str) -> None:
        super().__init__(parent)

        self.file_dialog = QFileDialog(self)
        self._file_name, _ = self.file_dialog.getSaveFileName(self, 
            caption=caption, 
            directory='', 
            filter=file_types)
        self._file_text = file_text
        
        _, file_type = os.path.splitext(self._file_name)
        if file_type == ".txt":
            self._saveToTextFile()
        elif file_type == ".pdf":
            self._saveToPDFFile()
        else:
            log.error("Unknown file type was wanted to be saved! Not saving...")

    def getSaveFileName(self) -> str:
        return self._file_name
    
    def _saveToTextFile(self):
        try:
            with open(self._file_name, 'w') as file:
                file.write(self._file_text)

        except Exception as e:
            log.error(f"Error Save to Text File: {str(e)}")

    def _saveToPDFFile(self):
        try: # TODO: Inprove PDF saving algorithm
            writer = PyPDF2.PdfWriter()
            # Need to specify page size since there is no prior page to
            # draw size from.
            writer.add_blank_page(800.0, 800.0)

            annotation = PyPDF2.generic.AnnotationBuilder.free_text(
                self._file_text,
                rect=(50, 550, 200, 650),
                font="Arial",
                bold=True,
                italic=True,
                font_size="20pt",
                font_color="00ff00",
                border_color="0000ff",
                background_color="cdcdcd",
            )
            writer.add_annotation(page_number=0, annotation=annotation)

            # Write the PDF content to a file
            with open(self._file_name, 'wb') as output_file:
                writer.write(output_file)

        except Exception as e:
            log.error(f"Error Save to PDF File: {str(e)}")
