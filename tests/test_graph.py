import pandas as pd
import sys
from PyQt5.QtWidgets import QApplication
from main import StatacondaGUI, set_test_environment

def test_lgraph():
    """Test line graph command"""
    # Create test data
    df = pd.DataFrame({
        'year': [2000, 2000, 2001, 2001, 2002, 2002],
        'value': [1, 2, 3, 4, 5, 6],
        'group': ['A', 'B', 'A', 'B', 'A', 'B']
    })
    
    # Create GUI instance
    app = QApplication(sys.argv)
    set_test_environment(True)
    gui = StatacondaGUI()
    gui._df = df
    
    # Test single series (should show mean for each year)
    result = gui.cmd_lgraph('lgraph value year')
    assert 'Line graph of value vs year displayed' in result
    
    # Test grouped series using direct group variable
    result = gui.cmd_lgraph('lgraph value year group')
    assert 'Line graph of value vs year displayed' in result
    
    # Test grouped series using by() option
    result = gui.cmd_lgraph('lgraph value year, by(group)')
    assert 'Line graph of value vs year displayed' in result
    
    # Test with titles
    result = gui.cmd_lgraph('lgraph value year, title("Test") xtitle("Year") ytitle("Value")')
    assert 'Line graph of value vs year displayed' in result
    
    # Test error handling
    result = gui.cmd_lgraph('lgraph nonexistent year')
    assert 'Error' in result
    
    result = gui.cmd_lgraph('lgraph value nonexistent')
    assert 'Error' in result
    
    print("All lgraph tests passed!")
    set_test_environment(False)

def test_multiple_commands():
    """Test executing multiple commands on separate lines"""
    # Create test data
    df = pd.DataFrame({
        'year': [2000, 2000, 2001, 2001, 2002, 2002],
        'value': [1, 2, 3, 4, 5, 6],
        'group': ['A', 'B', 'A', 'B', 'A', 'B']
    })
    
    # Create GUI instance
    app = QApplication(sys.argv)
    set_test_environment(True)
    gui = StatacondaGUI()
    gui._df = df
    
    # Test multiple commands
    commands = """describe
summarize
lgraph value year"""
    
    # Set the commands in the command prompt
    gui.command_prompt.setPlainText(commands)
    
    # Execute the commands
    gui.execute_command()
    
    # Get the results
    results = gui.results_window.toPlainText()
    
    # Verify that all commands were executed
    assert "Variable Overview" in results  # from describe
    assert "Summary statistics" in results  # from summarize
    assert "Line graph of value vs year displayed" in results  # from lgraph
    
    print("Multiple commands test passed!")
    set_test_environment(False)

if __name__ == "__main__":
    test_lgraph()
    test_multiple_commands() 