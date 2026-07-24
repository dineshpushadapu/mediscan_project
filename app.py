import os
import sys

# Ensure User_interface directory is in python path
ui_dir = os.path.join(os.path.dirname(__file__), "User_interface")
if ui_dir not in sys.path:
    sys.path.insert(0, ui_dir)

# Import and execute main Streamlit UI
from User_interface import main
