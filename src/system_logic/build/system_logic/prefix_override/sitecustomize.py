import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ieerasufcg/arena_ws/src/system_logic/install/system_logic'
