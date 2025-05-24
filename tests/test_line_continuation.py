import sys
from PyQt5.QtWidgets import QApplication
from main import StatacondaGUI, set_test_environment, show_and_close_figures
import pandas as pd

def test_line_continuation():
    # Set test environment to handle graph closing
    set_test_environment(True)
    
    # Create application instance
    app = QApplication(sys.argv)
    gui = StatacondaGUI()
    
    # Test case 1: Simple line continuation
    test_command1 = """binscatter invest capital, \\
title("Binned Scatter: Invest vs Capital")"""
    gui.command_prompt.setPlainText(test_command1)
    gui.execute_command()
    show_and_close_figures()
    
    # Test case 2: Multiple line continuation
    test_command2 = """binscatter invest capital, \\
title("Binned Scatter: Invest vs Capital, \\
controlling for Value")"""
    gui.command_prompt.setPlainText(test_command2)
    gui.execute_command()
    show_and_close_figures()
    
    # Test case 3: Multiple commands with line continuation
    test_command3 = """binscatter invest capital, \\
title("First Plot")

binscatter value capital, \\
title("Second Plot")"""
    gui.command_prompt.setPlainText(test_command3)
    gui.execute_command()
    show_and_close_figures()
    
    print("All line continuation tests completed!")

def test_panel_lag_variable():
    # Set test environment to handle graph closing
    set_test_environment(True)
    
    app = QApplication(sys.argv)
    gui = StatacondaGUI()
    # Create a simple panel dataset
    df = pd.DataFrame({
        'firm': [1, 1, 1, 2, 2, 2],
        'year': [2000, 2001, 2002, 2000, 2001, 2002],
        'capital': [10, 20, 30, 40, 50, 60]
    })
    gui._df = df.copy()
    # Set panel structure
    result_tsset = gui.cmd_tsset('tsset firm year')
    assert 'Panel data set' in result_tsset
    # Check that firm_idx and year_idx are in index
    assert 'firm_idx' in gui._df.index.names
    assert 'year_idx' in gui._df.index.names
    # Generate lagged variable
    result_gen = gui.cmd_generate('gen L1_capital = L.capital')
    assert 'Generated new variable' in result_gen
    # Check lag values
    lagged = gui._df['L1_capital']
    # For firm 1, years 2000, 2001, 2002: lag should be [NaN, 10, 20]
    firm1 = lagged.loc[1]
    assert pd.isna(firm1.loc[2000])
    assert firm1.loc[2001] == 10
    assert firm1.loc[2002] == 20
    # For firm 2, years 2000, 2001, 2002: lag should be [NaN, 40, 50]
    firm2 = lagged.loc[2]
    assert pd.isna(firm2.loc[2000])
    assert firm2.loc[2001] == 40
    assert firm2.loc[2002] == 50
    print('Panel lag variable test passed!')

if __name__ == "__main__":
    test_line_continuation()
    test_panel_lag_variable() 