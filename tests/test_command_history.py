import sys
from PyQt5.QtWidgets import QApplication
from main import StatacondaGUI

def test_command_history():
    # Create QApplication instance
    app = QApplication(sys.argv)
    
    # Create GUI instance
    gui = StatacondaGUI()
    
    # Test 1: Basic command logging
    test_commands = [
        "summarize invest",
        "describe",
        "browse"
    ]
    
    for cmd in test_commands:
        gui.command_prompt.setPlainText(cmd)
        gui.execute_command()
    
    # Verify commands were added to history
    assert len(gui.command_history) == len(test_commands), "Not all commands were added to history"
    for i, cmd in enumerate(test_commands):
        assert gui.history_list.item(i).text() == cmd, f"Command {i} not properly logged"
    
    # Test 2: Command filtering
    gui.history_filter.setText("sum")
    gui.filter_history()
    
    # Verify filtered results
    filtered_count = gui.history_list.count()
    assert filtered_count == 1, f"Filter should show 1 command, got {filtered_count}"
    assert gui.history_list.item(0).text() == "summarize invest", "Filter did not find correct command"
    
    # Test 3: Clear filter
    gui.history_filter.clear()
    gui.filter_history()
    assert gui.history_list.count() == len(test_commands), "Filter clear did not restore all commands"
    
    print("All command history tests passed!")

if __name__ == "__main__":
    test_command_history() 