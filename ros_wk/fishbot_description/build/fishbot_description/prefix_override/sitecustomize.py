import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/elpco/code/lerobot/lerobot_rdt/ros_wk/fishbot_description/install/fishbot_description'
