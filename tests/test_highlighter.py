import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QTextDocument, QTextCharFormat, QFont, QColor
from main import StataHighlighter

def test_stata_highlighter():
    # Create QApplication instance (required for Qt)
    app = QApplication(sys.argv)
    
    # Create a test document
    doc = QTextDocument()
    doc.setPlainText("""Command: reg invest value
Linear regression                                      Number of obs = 50
                                                      F(2.0, 47.0) = 0.15
                                                      Prob > F      = 0.8575
                                                      R-squared     = 0.0065
                                                      Adj R-squared = -0.0358
                                                      Root MSE      = 1.1571

------------------------------------------------------------------------------
y            |      Coef.   Std. Err.      t    P>|t|     [95% Conf. Interval]
-------------+----------------------------------------------------------------
_cons        |     0.1236     0.1686    0.73  0.4672    -0.2156     0.4629
x1           |    -0.0655     0.1911   -0.34  0.7335    -0.4499     0.3190
x2           |     0.0607     0.1605    0.38  0.7071    -0.2623     0.3836
------------------------------------------------------------------------------

Error: Variable not found: nonexistent_var""")
    
    # Create highlighter and apply it to the document
    highlighter = StataHighlighter(doc)
    
    # Get the format at different positions to verify highlighting
    def get_format_at_line(line_number):
        cursor = doc.findBlockByLineNumber(line_number)
        if cursor.isValid():
            return highlighter.format(cursor.position())
        return None
    
    # Test command line (should be bold)
    command_format = get_format_at_line(0)
    assert command_format.fontWeight() == QFont.Bold
    assert command_format.foreground().color() == QColor("black")
    
    # Test output line (should be normal weight)
    output_format = get_format_at_line(1)
    assert output_format.fontWeight() == QFont.Normal
    assert output_format.foreground().color() == QColor("black")
    
    # Test error line (should be red)
    error_format = get_format_at_line(15)
    assert error_format.fontWeight() == QFont.Normal
    assert error_format.foreground().color() == QColor("red")
    
    print("All highlighter tests passed!")

if __name__ == "__main__":
    test_stata_highlighter() 