import argparse
import os
import time

from ftservo_controller import ServoController


def arc_span(cfg: dict) -> int:
    """Return arc length on 12-bit circle (always from range_min to range_max)."""
    span = (cfg["range_max"] - cfg["range_min"]) % 4096
    return span or 4096


def clamp_on_arc(value: int, cfg: dict) -> int:
    """Clamp value to the configured arc (from range_min to range_max)."""
    rng_min = cfg["range_min"] % 4096
    span = arc_span(cfg)
    value = value % 4096
    offset = (value - rng_min) % 4096
    
    if offset <= span:
        return value
    
    # 超出范围，选最近端点
    rng_max = (rng_min + span) % 4096
    dist_to_min = (rng_min - value) % 4096
    dist_to_max = (value - rng_max) % 4096
    return rng_min if dist_to_min < dist_to_max else rng_max


def progress_from_position(value: int, cfg: dict) -> float:
    """将位置转换为 0-1 的进度值 (range_min=0, range_max=1)"""
    clamped = clamp_on_arc(value, cfg)
    rng_min = cfg["range_min"] % 4096
    span = arc_span(cfg)
    offset = (clamped - rng_min) % 4096
    return offset / span if span > 0 else 0.0


def position_from_progress(progress: float, cfg: dict) -> int:
    """将 0-1 的进度值转换为位置"""
    rng_min = cfg["range_min"] % 4096
    span = arc_span(cfg)
    progress = min(max(progress, 0.0), 1.0)
    return int(round((rng_min + progress * span) % 4096))


def shortest_delta(target: int, current: int, modulo: int = 4096) -> int:
    """计算环形空间上从 current 到 target 的最短有符号位移"""
    diff = (target - current) % modulo
    if diff > modulo // 2:
        diff -= modulo
    return diff


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(__file__), path)


class SingleArmTeleop:
    def __init__(self, name: str, leader_controller: ServoController, follower_controller: ServoController,
                 joint_names: list, alpha: float, speed: int):
        self.name = name
        self.leader = leader_controller
        self.follower = follower_controller
        self.joint_names = joint_names
        self.alpha = alpha
        self.speed = speed

        self.filtered = {}  # 存储平滑后的从臂目标位置
        self.prev_leader_pos = {}  # 上一帧的引导臂位置
        self.unwrapped_leader = {}  # 展开的引导臂位置（处理跨0）
        self.unwrapped_follower_current = {}  # 展开的从臂当前位置

    def step(self, debug=False) -> str:
        leader_pos = self.leader.read_servo_positions(self.joint_names)
        follower_pos_raw = self.follower.read_servo_positions(self.joint_names)
        targets = {}
        debug_info = []

        for name in self.joint_names:
            src_cfg = self.leader.config[name]
            dst_cfg = self.follower.config[name]

            # --- 1. 引导臂展开逻辑 ---
            raw_leader = leader_pos[name]
            if name not in self.prev_leader_pos:
                self.unwrapped_leader[name] = raw_leader
            else:
                delta = shortest_delta(raw_leader, self.prev_leader_pos[name])
                self.unwrapped_leader[name] += delta
            self.prev_leader_pos[name] = raw_leader

            # --- 2. 映射逻辑 ---
            progress = progress_from_position(self.unwrapped_leader[name], src_cfg)
            mapped_pos_0_4096 = position_from_progress(progress, dst_cfg)

            # --- 3. 从臂最短路径逻辑 (Multi-turn Support) ---
            current_raw = follower_pos_raw[name]
            if name not in self.unwrapped_follower_current:
                self.unwrapped_follower_current[name] = current_raw
            else:
                last_raw = self.unwrapped_follower_current[name] % 4096
                delta_feedback = shortest_delta(current_raw, last_raw)
                self.unwrapped_follower_current[name] += delta_feedback

            current_unwrapped = self.unwrapped_follower_current[name]

            # 计算从当前虚拟位置到映射目标的最近增量
            diff_to_target = shortest_delta(mapped_pos_0_4096, current_unwrapped % 4096)
            target_unwrapped = current_unwrapped + diff_to_target

            # --- 4. 低通滤波 ---
            if name in self.filtered:
                smoothed = self.filtered[name] + self.alpha * (target_unwrapped - self.filtered[name])
            else:
                smoothed = float(target_unwrapped)

            self.filtered[name] = smoothed

            # 发送给舵机的值不取模，允许发送 >4095 或 <0 的值
            targets[name] = int(round(smoothed))
            
            if debug:
                debug_info.append(f"{name}:L{raw_leader}->T{targets[name]}")

        self.follower.fast_move_to_pose(targets, speed=self.speed)
        
        return f"[{self.name}] " + " | ".join(debug_info) if debug else ""


def run(args: argparse.Namespace) -> None:
    # 1. Initialize Paths
    left_leader_cfg_path = resolve_path(args.left_leader_config)
    left_follower_cfg_path = resolve_path(args.left_follower_config)
    right_leader_cfg_path = resolve_path(args.right_leader_config)
    right_follower_cfg_path = resolve_path(args.right_follower_config)

    # 2. Initialize Controllers
    print("Connecting to Left Arm...")
    left_leader = ServoController(args.left_leader_port, args.baudrate, left_leader_cfg_path)
    left_follower = ServoController(args.left_follower_port, args.baudrate, left_follower_cfg_path)
    
    print("Connecting to Right Arm...")
    right_leader = ServoController(args.right_leader_port, args.baudrate, right_leader_cfg_path)
    right_follower = ServoController(args.right_follower_port, args.baudrate, right_follower_cfg_path)

    # 3. Determine Joints
    left_joints = args.joints.split(",") if args.joints else [
        n for n in left_leader.config.keys() if n in left_follower.config
    ]
    right_joints = args.joints.split(",") if args.joints else [
        n for n in right_leader.config.keys() if n in right_follower.config
    ] 
    
    # 4. Create Teleop Instances
    alpha = min(max(args.alpha, 0.0), 1.0)
    left_teleop = SingleArmTeleop("Left", left_leader, left_follower, left_joints, alpha, args.speed)
    right_teleop = SingleArmTeleop("Right", right_leader, right_follower, right_joints, alpha, args.speed)

    # 5. Print Configs (Debug)
    if args.debug:
        print("\n=== Left Arm Config ===")
        for name in left_joints:
             print(f"{name} Span: {arc_span(left_leader.config[name])} -> {arc_span(left_follower.config[name])}")
        print("\n=== Right Arm Config ===")
        for name in right_joints:
             print(f"{name} Span: {arc_span(right_leader.config[name])} -> {arc_span(right_follower.config[name])}")
        print("="*30)

    print("Starting Dual Arm Teleop. Ctrl+C to exit.")
    period = 1.0 / args.rate
    debug_counter = 0

    try:
        while True:
            start_time = time.time()
            
            # Step both arms
            left_debug = left_teleop.step(debug=(args.debug and debug_counter % 10 == 0))
            right_debug = right_teleop.step(debug=(args.debug and debug_counter % 10 == 0))

            if args.debug and debug_counter % 10 == 0:
                print(f"\r{left_debug} || {right_debug}", end="", flush=True)

            debug_counter += 1
            
            # Maintain loop rate
            elapsed = time.time() - start_time
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        print("\nDual Arm Teleop stopped by user.")
    finally:
        print("Closing connections...")
        left_leader.close()
        left_follower.close()
        right_leader.close()
        right_follower.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual Arm Leader-follower teleoperation")
    
    # Left Arm
    parser.add_argument("--left-leader-port", default="/dev/left_leader", help="Port for Left Leader")
    parser.add_argument("--left-follower-port", default="/dev/left_arm", help="Port for Left Follower")
    parser.add_argument("--left-leader-config", default="left_arm_leader.json", help="Config for Left Leader")
    parser.add_argument("--left-follower-config", default="left_arm.json", help="Config for Left Follower")
    
    # Right Arm
    parser.add_argument("--right-leader-port", default="/dev/right_leader", help="Port for Right Leader")
    parser.add_argument("--right-follower-port", default="/dev/right_arm", help="Port for Right Follower")
    parser.add_argument("--right-leader-config", default="right_arm_leader.json", help="Config for Right Leader")
    parser.add_argument("--right-follower-config", default="right_arm.json", help="Config for Right Follower")

    # Common
    parser.add_argument("--baudrate", type=int, default=1_000_000, help="Serial baudrate")
    parser.add_argument("--rate", type=float, default=30.0, help="Control loop rate (Hz)")
    parser.add_argument("--speed", type=int, default=1200, help="Servo speed for followers")
    parser.add_argument("--alpha", type=float, default=0.35, help="Low-pass smoothing factor 0-1")
    parser.add_argument("--joints", default="", help="Comma-separated joint names (if empty, matches all shared)")
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
