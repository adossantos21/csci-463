"""
File Name: gui_main.py
Purpose: Formats GUI
Author: Arteom Katkov and Alessandro Dos Santos
Documented: 05/10/2024
"""
import enum
from loguru import logger as log
from PyQt5.QtWidgets import QMainWindow, QTextEdit, QVBoxLayout, QWidget, QLabel, QComboBox, QPushButton, QTableWidget, QHeaderView, QTableWidgetItem
from PyQt5.QtGui import QFont
from PyQt5 import QtCore
import PyPDF2

#from neural_net.neural_net import NeuralNet
from neural_net.scratch import NeuralNet
from gui.components.error_popup import ErrorPopup
from gui.components.input_file_dialog import InputFileDialog
from gui.components.navigation import ButtonTypes, NavigationBar
from gui.components.output_file_dialog import OutputFileDialog
from gui.threading.nn_compute_thread import NNComputeThread

class GuiDisplayStates(enum.Enum):
    MODEL_SELECTION = 0
    GET_INPUT_TEXT = 1
    SAVE_OUTPUT_TEXT = 2
    LOG = 5
    DISPLAY_ERROR = 3

class GUIApp(QMainWindow):
    APPLICATION_NAME = "Nutshell Summarizer Application"
    
    def __init__(self, nn: NeuralNet):
        super().__init__()
        
        self._display_state = GuiDisplayStates.MODEL_SELECTION
        self._nn = nn
        self._log = {}
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 600, 400)
        self.setWindowTitle(GUIApp.APPLICATION_NAME)

        # Initialize GUI Display Variables
        self._layout = QVBoxLayout()
        self._screen_title = QLabel()
        self._screen_title.setText("Nutshell Summarizer Application")
        newFont = QFont()
        newFont.setFamily("Arial")
        newFont.setPointSize(20)
        self._screen_title.setFont(newFont)
        self._screen_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        self._layout.addWidget(self._screen_title)

        self._input_text_display = QTextEdit(self)
        self._layout.addWidget(self._input_text_display)

        self._output_text_display = QTextEdit(self)
        self._output_text_display.setReadOnly(True)
        self._layout.addWidget(self._output_text_display)

        self._model_combobox = QComboBox()
        self._model_combobox.addItem("Finetuned Llama-v2-7B")
        self._model_combobox.addItem("LongT5")
        self._layout.addWidget(self._model_combobox)
        
        self._nav_bar = NavigationBar()
        self._layout.addLayout(self._nav_bar.getLayout())
        self._nav_bar.setButtonClickCb(self._onNavBarButtonClick)

        self._btn_log = QPushButton('Logs')
        self._btn_log.clicked.connect(self._onLogButtonClick)
        self._layout.addWidget(self._btn_log)

        self._btn_home = QPushButton('Home')
        self._btn_home.clicked.connect(self._onLogHomeButtonClick)
        self._layout.addWidget(self._btn_home)

        # Initialize the QTableWidget
        self._logTable = QTableWidget()
        self._logTable.setColumnCount(2)
        self._logTable.setHorizontalHeaderLabels(["Query", "Summary"])
        self._logTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._layout.addWidget(self._logTable)


        self.widget = QWidget()
        self.widget.setLayout(self._layout)
        self.setCentralWidget(self.widget)

        #self._grid_layout = QGridLayout()
        #self._layout.addLayout(self._grid_layout)

        # Need to create a new class in scratch.py for loading a model from HuggingFace for multiple models using summarizer pipeline
        # Then probably use if statements to for Basic Model or HuggingFace Model to specify which model is used by NNComputeThread
        self._nn_compute = NNComputeThread(self._input_text_display.toPlainText(), self._nn)
        self._nn_compute.text_result.connect(self._onNNComputeComplete)
        self._updateUIView() # Setup the initial UI view
            
    def _onNNComputeComplete(self, data: str):
        self._query = self._input_text_display.toPlainText()
        self._input_text_display.clear()
        self._output_text_display.setText(data) 

    def _writeToLog(self, query: str, summary: str):
        self._log[query] = summary

    def _onLogButtonClick(self) -> None: 
        self._display_state = GuiDisplayStates(GuiDisplayStates.LOG.value)
        self._logTable.setRowCount(0)
        items = self._log
        if items is not None:
            self._logTable.setRowCount(len(items))

            for index, (key, value) in enumerate(items.items()):
                self._logTable.setItem(index, 0, QTableWidgetItem(str(key)))
                self._logTable.setItem(index, 1, QTableWidgetItem(str(value)))
        else:
            self._logTable.setRowCount(1)
            self._logTable.setItem(0, 0, QTableWidgetItem("No summaries requested yet"))
            self._logTable.setItem(0, 1, QTableWidgetItem(""))

        self._updateUIView()
        
    def _onLogHomeButtonClick(self):
        self._display_state = GuiDisplayStates(GuiDisplayStates.MODEL_SELECTION.value)
        self._updateUIView()

    def _onNavBarButtonClick(self, btnType: ButtonTypes):
        if btnType == ButtonTypes.BACK_BUTTON and self._display_state != GuiDisplayStates.DISPLAY_ERROR and self._isValidReverseTransition():
            self._display_state = GuiDisplayStates((self._display_state.value - 1) % GuiDisplayStates.DISPLAY_ERROR.value)
            self._updateUIView()

        elif btnType == ButtonTypes.NEXT_BUTTON and self._display_state != GuiDisplayStates.DISPLAY_ERROR:
            if self._isValidForwardTransition():
                self._display_state = GuiDisplayStates((self._display_state.value + 1) % GuiDisplayStates.DISPLAY_ERROR.value)
                self._updateUIView()

            else:
                tempDialog = ErrorPopup("Invalid Input", "Select correct input to proceed", self)
                tempDialog.exec()

        elif btnType == ButtonTypes.ACTION_BUTTON: # TODO: Add actions in certain states
            if self._display_state == GuiDisplayStates.GET_INPUT_TEXT:
                file_dialog = InputFileDialog(self, "Open Input Text File", "Input Text (*.pdf *.txt)")
                if file_dialog.getOpenedFileContents() != "":
                    self._input_text_display.setText(file_dialog.getOpenedFileContents())

            elif self._display_state == GuiDisplayStates.SAVE_OUTPUT_TEXT:
                OutputFileDialog(self, "Save Output Model Text", "Text File(*.txt);;PDF File(*.pdf)", self._output_text_display.toPlainText())

    '''
    def _isValidReverseTransition(self) -> bool:
         match self._display_state:
            case GuiDisplayStates.MODEL_SELECTION: # Don't go back from the first page
                return False
             
            case _:
                return True
    '''

    def _isValidReverseTransition(self) -> bool:
        if self._display_state == GuiDisplayStates.MODEL_SELECTION:  # Don't go back from the first page
            return False
        else:
            return True

    def _isValidForwardTransition(self) -> bool:
        if self._display_state == GuiDisplayStates.MODEL_SELECTION:
            return True if self._model_combobox.currentIndex() < 1 else False
        elif self._display_state == GuiDisplayStates.GET_INPUT_TEXT:
            return True if self._input_text_display.toPlainText() != "" else False
        else:
            return True

    def _updateUIView(self) -> None:
        if self._display_state == GuiDisplayStates.MODEL_SELECTION:
            self._nav_bar._onShow()
            self._input_text_display.hide()
            self._output_text_display.hide()
            self._logTable.hide()
            self._btn_home.hide()
            self._model_combobox.show()
            self._btn_log.show()
            self._nav_bar.getActionButton().hide()
            self._screen_title.setText("Model Selection")

        elif self._display_state == GuiDisplayStates.GET_INPUT_TEXT:
            self._nav_bar._onShow()
            self._model_combobox.hide()
            self._logTable.hide()
            self._btn_log.hide()
            self._btn_home.hide()
            self._screen_title.setText("Select the input text")
            self._nav_bar.getActionButton().show()
            self._nav_bar.getActionButton().setText("Open Input File")
            self._input_text_display.show()

        elif self._display_state == GuiDisplayStates.SAVE_OUTPUT_TEXT:
            self._nav_bar._onShow()
            self._screen_title.setText("Save Output Text")
            self._nav_bar.getActionButton().setText("Save Generated Output")
            self._input_text_display.hide()
            self._logTable.hide()
            self._btn_log.hide()
            self._btn_home.hide()
            self._output_text_display.show()
            # TODO: Create state for generating output
            self._nn_compute.setInputText(self._input_text_display.toPlainText())
            IN, OUT = self._nn_compute.run()
            self._writeToLog(IN, OUT)#self._nn_compute.text_result)

        elif self._display_state == GuiDisplayStates.LOG:
            self._screen_title.setText("Logs")
            self._input_text_display.hide()
            self._output_text_display.hide()
            self._model_combobox.hide()
            self._nav_bar.getActionButton().hide()
            self._btn_log.hide()
            self._nav_bar._onHide()
            self._logTable.show()
            self._btn_home.show()
        elif self._display_state == GuiDisplayStates.DISPLAY_ERROR:
            self._screen_title.setText("Error!!!! Should not be here....")

        else:
            log.critical("Unknown GUI View!!")
            self._display_state = GuiDisplayStates.DISPLAY_ERROR
        