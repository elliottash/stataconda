import sys
from PyQt5.QtWidgets import QApplication
from main import StatacondaGUI

def test_reghdfe():
    # Create QApplication instance
    app = QApplication(sys.argv)
    
    # Create GUI instance
    gui = StatacondaGUI()
    
    # Test basic reghdfe
    cmd = "reghdfe invest value, absorb(firm)"
    result = gui.translate_reghdfe(cmd)
    print("\nTest 1: Basic reghdfe with absorb()")
    print(result)
    
    # Test reghdfe without absorb to verify constant is included
    cmd = "reghdfe invest value"
    result = gui.translate_reghdfe(cmd)
    print("\nTest 2: reghdfe without absorb()")
    print(result)
    
    # Test reghdfe with multiple absorb variables
    cmd = "reghdfe invest value, absorb(firm year)"
    result = gui.translate_reghdfe(cmd)
    print("\nTest 3: reghdfe with multiple absorb variables")
    print(result)
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_reghdfe() 