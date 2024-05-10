"""
File Name: nn_compute_thread.py
Purpose: Configures neural network forward propagation for GUI
Author: Arteom Katkov
Documented: 05/10/2024
"""
from loguru import logger as log
from PyQt5 import QtCore

#from neural_net.neural_net import NeuralNet
from neural_net.scratch import NeuralNet

class NNComputeThread(QtCore.QThread):
    text_result = QtCore.pyqtSignal(object)
    
    def __init__(self, text_array, nn: NeuralNet) -> None:
        super().__init__()
        self._text_array = text_array
        self._nn = nn
        
    def run(self) -> None:
        try:
            INPUT = self._text_array
            OUTPUT = self._nn.summarize_text(str(INPUT))
            self.text_result.emit(OUTPUT)
            return INPUT, OUTPUT
        except Exception as e:
            log.error(f"Error While Running Neural Net: {str(e)}")
            self.text_result.emit("???")

    
    def run_HF(self) -> None:
        try:
            length = len(str(self._text_array).split(" ", -1))
            if length > 220:# Should be 1024 but I'm not sure what the tokenizer splits on
                self._text_array = str(self._text_array).split(" ", -1)[0:220]
            
            # TODO: Hopefully not need this parsing?
            out_data = self._nn.summarize_text(str(self._text_array))
            word_list = []
            for word in out_data[0]["summary_text"].split():
                if word[0] == "'" or word[-1] == ',':
                    if len(word) > 2 and word[-2] == "'":
                        word_list.append(word[1:-2])
                    else:
                        word_list.append(word[1:-1])
                else:
                    word_list.append(word)

            self.text_result.emit((" ").join(word_list))

        except Exception as e:
            log.error(f"Error While Running Neural Net: {str(e)}")
            self.text_result.emit("???")

            
    def setInputText(self, text: str) -> None:
        self._text_array = text
