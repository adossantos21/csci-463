"""
File: main.py
Description: Main application entrypoint
Date: 01/29/24
"""
import sys

from PyQt5.QtWidgets import QApplication

#from neural_net.neural_net import NeuralNet
from neural_net.scratch import NeuralNet
from gui.gui_main import GUIApp

def main() -> None:
    try:
        my_net = NeuralNet()
        app = QApplication(sys.argv)
        my_gui = GUIApp(my_net)
        my_gui.show()
        sys.exit(app.exec_())
        
    except KeyboardInterrupt:
        sys.exit(-1)

if __name__ == '__main__':
    main()
