#!/bin/bash
# SO-101 机械臂 ROS2 仿真启动脚本

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SO-101 机械臂 ROS2 仿真启动脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查 ROS2 环境
if [ -z "$ROS_DISTRO" ]; then
    echo -e "${YELLOW}⚠️  ROS2 环境未配置，尝试自动加载...${NC}"
    if [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
        echo -e "${GREEN}✅ 已加载 ROS2 Humble${NC}"
    elif [ -f "/opt/ros/foxy/setup.bash" ]; then
        source /opt/ros/foxy/setup.bash
        echo -e "${GREEN}✅ 已加载 ROS2 Foxy${NC}"
    else
        echo -e "${RED}❌ 找不到 ROS2 安装，请先安装 ROS2${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ ROS2 环境: $ROS_DISTRO${NC}"
fi

# 进入 sim 目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检查必要文件
if [ ! -f "so101_new_calib.urdf" ]; then
    echo -e "${RED}❌ 找不到 URDF 文件: so101_new_calib.urdf${NC}"
    exit 1
fi

echo -e "${GREEN}✅ URDF 文件就绪${NC}"

# 检查是否安装了 joint_state_publisher_gui
if ros2 pkg list | grep -q "joint_state_publisher_gui"; then
    echo -e "${GREEN}✅ 检测到 joint_state_publisher_gui${NC}"
    HAS_GUI=true
else
    echo -e "${YELLOW}⚠️  未安装 joint_state_publisher_gui${NC}"
    echo -e "${YELLOW}   将使用程序化控制模式${NC}"
    HAS_GUI=false
fi

echo ""
echo -e "${GREEN}启动方式:${NC}"
echo "  1) 仅启动 RViz2 和 robot_state_publisher"
echo "  2) 启动并运行正弦波轨迹演示"
echo "  3) 安装 joint_state_publisher_gui"
echo ""
read -p "请选择 (1/2/3): " choice

case $choice in
    1)
        echo -e "${GREEN}启动基础可视化...${NC}"
        echo -e "${YELLOW}请在另一终端运行: python3 joint_state_publisher.py${NC}"
        python3 display.launch
        ;;
    2)
        echo -e "${GREEN}启动完整仿真（带轨迹演示）...${NC}"
        # 在后台启动 launch 文件
        python3 display.launch &
        LAUNCH_PID=$!
        
        # 等待节点启动
        sleep 3
        
        # 启动关节控制
        echo -e "${GREEN}启动关节控制脚本...${NC}"
        python3 joint_state_publisher.py
        
        # 清理
        kill $LAUNCH_PID 2>/dev/null
        ;;
    3)
        echo -e "${GREEN}安装 joint_state_publisher_gui...${NC}"
        sudo apt update
        sudo apt install ros-$ROS_DISTRO-joint-state-publisher-gui -y
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ 安装成功！请重新运行此脚本${NC}"
        else
            echo -e "${RED}❌ 安装失败${NC}"
        fi
        ;;
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac
