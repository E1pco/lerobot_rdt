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


def run(args: argparse.Namespace) -> None:
    leader_cfg_path = resolve_path(args.leader_config)
    follower_cfg_path = resolve_path(args.follower_config)

    leader = ServoController(args.leader_port, args.baudrate, leader_cfg_path)
    follower = ServoController(args.follower_port, args.baudrate, follower_cfg_path)

    joint_names = args.joints.split(",") if args.joints else [
        n for n in leader.config.keys() if n in follower.config
    ]
    joint_names = [n.strip() for n in joint_names if n.strip()]
    if not joint_names:
        raise ValueError("No shared joints between leader and follower configs")

    alpha = min(max(args.alpha, 0.0), 1.0)
    period = 1.0 / args.rate
    filtered = {}  # 存储平滑后的从臂目标位置
    prev_leader_pos = {}  # 上一帧的引导臂位置
    unwrapped_leader = {}  # 展开的引导臂位置（处理跨0）
    unwrapped_follower_current = {} # 展开的从臂当前位置

    # 调试：打印各关节的弧配置
    if args.debug:
        print("\n=== Joint Arc Configuration ===")
        for name in joint_names:
            src_cfg = leader.config[name]
            dst_cfg = follower.config[name]
            print(f"{name}:")
            print(f"  Leader:   min={src_cfg['range_min']:4d}, max={src_cfg['range_max']:4d}, "
                  f"span={arc_span(src_cfg):4d}")
            print(f"  Follower: min={dst_cfg['range_min']:4d}, max={dst_cfg['range_max']:4d}, "
                  f"span={arc_span(dst_cfg):4d}")
        print("=" * 35 + "\n")

    print("Starting teleop. Ctrl+C to exit.")
    debug_counter = 0
    try:
        while True:
            leader_pos = leader.read_servo_positions(joint_names)
            follower_pos_raw = follower.read_servo_positions(joint_names)
            targets = {}

            for name in joint_names:
                src_cfg = leader.config[name]
                dst_cfg = follower.config[name]
                
                # --- 1. 引导臂展开逻辑 ---
                raw_leader = leader_pos[name]
                if name not in prev_leader_pos:
                    unwrapped_leader[name] = raw_leader
                else:
                    delta = shortest_delta(raw_leader, prev_leader_pos[name])
                    unwrapped_leader[name] += delta
                prev_leader_pos[name] = raw_leader
                
                # --- 2. 映射逻辑 ---
                progress = progress_from_position(unwrapped_leader[name], src_cfg)
                mapped_pos_0_4096 = position_from_progress(progress, dst_cfg)

                current_raw = follower_pos_raw[name]
                if name not in unwrapped_follower_current:
                    unwrapped_follower_current[name] = current_raw
                else:
                    last_raw = unwrapped_follower_current[name] % 4096
                    delta_feedback = shortest_delta(current_raw, last_raw)
                    unwrapped_follower_current[name] += delta_feedback

                current_unwrapped = unwrapped_follower_current[name]
                
                # 计算从当前虚拟位置到映射目标的最近增量
                diff_to_target = shortest_delta(mapped_pos_0_4096, current_unwrapped % 4096)
                target_unwrapped = current_unwrapped + diff_to_target

                # --- 4. 低通滤波 ---
                if name in filtered:
                     # 直接在展开空间滤波，无需再调 shortest_delta
                    smoothed = filtered[name] + alpha * (target_unwrapped - filtered[name])
                else:
                    smoothed = float(target_unwrapped)
                
                filtered[name] = smoothed  # 保持展开值用于滤波连续性
                
                # 发送给舵机的值不取模，允许发送 >4095 或 <0 的值（舵机需处于多圈模式或支持绝对位置）
                # 原始代码：targets[name] = int(round(smoothed)) % 4096
                targets[name] = int(round(smoothed))

            # 调试输出
            if args.debug:
                debug_counter += 1
                if debug_counter % 10 == 0:  # 每10帧输出一次
                    debug_line = " | ".join(
                        f"{name}: L{leader_pos[name]:4d}->T{targets[name]:5d}"
                        for name in joint_names
                    )
                    print(f"\r{debug_line}", end="", flush=True)

            follower.fast_move_to_pose(targets, speed=args.speed)
            time.sleep(period)
    except KeyboardInterrupt:
        if args.debug:
            print()  # 换行
        print("\nTeleop stopped by user.")
    finally:
        leader.close()
        follower.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leader-follower teleoperation with offset mapping")
    parser.add_argument("--leader-port", default="/dev/right_leader", help="Serial port of leader arm")
    parser.add_argument("--follower-port", default="/dev/right_arm", help="Serial port of follower arm")
    parser.add_argument("--leader-config", default="right_arm_leader.json", help="Leader config JSON")
    parser.add_argument("--follower-config", default="right_arm.json", help="Follower config JSON")
    parser.add_argument("--baudrate", type=int, default=1_000_000, help="Serial baudrate")
    parser.add_argument("--rate", type=float, default=30.0, help="Control loop rate (Hz)")
    parser.add_argument("--speed", type=int, default=1200, help="Servo speed for follower")
    parser.add_argument("--alpha", type=float, default=0.35, help="Low-pass smoothing factor 0-1")
    parser.add_argument("--joints", default="", help="Comma-separated joint names to mirror")
    parser.add_argument("--debug", action="store_true", help="Print debug info for mapping")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
