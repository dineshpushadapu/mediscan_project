import os
import sys
import runpy

# Ensure root and User_interface directories are in Python path
root_dir = os.path.dirname(os.path.abspath(__file__))
ui_dir = os.path.join(root_dir, "User_interface")

if ui_dir not in sys.path:
    sys.path.insert(0, ui_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Execute main Streamlit UI script
main_path = os.path.join(ui_dir, "main.py")
runpy.run_path(main_path, run_name="__main__")
