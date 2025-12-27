import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 尝试导入 from_urdf，如果不存在则显示相关信息
try:
    from urdf2dh import from_urdf
    result = from_urdf("so101_new_calib.urdf")
    print(result)
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("\n注意: urdf2dh 包的 from_urdf 函数未在 __init__.py 中导出")
    print("请检查 urdf2dh 包的完整性，或确保 from_urdf 函数已被正确定义和导出")
except FileNotFoundError:
    print("❌ 错误: 找不到 so101_new_calib.urdf 文件")
    print(f"当前目录: {os.getcwd()}")
    print(f"脚本目录: {os.path.dirname(os.path.abspath(__file__))}")
except Exception as e:
    print(f"❌ 运行错误: {e}")