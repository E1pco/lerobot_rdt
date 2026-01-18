import argparse
import json
import os
import select
import sys
import time

from ftservo_driver import FTServo


# 默认关节定义，避免依赖外部 servo_config.json
DEFAULT_JOINTS = [
    {"name": "shoulder_pan", "id": 1, "drive_mode": 0},
    {"name": "shoulder_lift", "id": 2, "drive_mode": 0},
    {"name": "elbow_flex", "id": 3, "drive_mode": 0},
    {"name": "wrist_flex", "id": 4, "drive_mode": 0},
    {"name": "wrist_yaw", "id": 5, "drive_mode": 0},
    {"name": "wrist_roll", "id": 6, "drive_mode": 0},
    {"name": "gripper", "id": 7, "drive_mode": 0},
]


def shortest_delta(cur: int, prev: int, modulo: int = 4096) -> int:
    """计算环形空间上从 prev 到 cur 的最短有符号位移（-modulo/2, modulo/2]）"""
    diff = (cur - prev) % modulo
    if diff > modulo // 2:
        diff -= modulo
    return diff


def read_positions(servo: FTServo, joints, prev=None):
    ids = [j["id"] for j in joints]
    resp = servo.sync_read(0x38, 2, ids)
    positions = {}
    for j in joints:
        sid = j["id"]
        name = j["name"]
        if resp and sid in resp:
            params = resp[sid]
            # 解析为无符号16位后取12位有效位，确保在0-4095范围内
            positions[name] = ((params[0] & 0xFF) | ((params[1] & 0xFF) << 8)) & 0x0FFF
        else:
            positions[name] = prev.get(name, 0) if prev else 0
    return positions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively record servo limits and dump a config JSON."
    )
    parser.add_argument("--port", default="/dev/right_leader", help="Serial port of the arm")
    parser.add_argument("--baudrate", type=int, default=1_000_000, help="Baudrate")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "right_arm_leader.json"),
        help="Where to write the calibrated JSON",
    )
    parser.add_argument("--interval", type=float, default=0.2, help="Polling interval seconds")
    args = parser.parse_args()

    servo = FTServo(args.port, args.baudrate)
    try:
        joint_names = [j["name"] for j in DEFAULT_JOINTS]

        input("Place the arm at your desired neutral pose, then press Enter to start recording ...")
        mid_positions = read_positions(servo, DEFAULT_JOINTS, prev=None)
        for name in joint_names:
            print(f"{name:15s}: {mid_positions[name]:4d}")

        print("\nRecording ranges. Manually move every joint to its extremes.")
        print("Press Enter again to stop and write the JSON.\n")

        # 使用累积偏移量来追踪范围，正确处理跨0问题
        prev_positions = {k: v for k, v in mid_positions.items()}
        cumulative_offset = {k: 0 for k in mid_positions.keys()}  # 相对于中位的累积偏移
        min_offset = {k: 0 for k in mid_positions.keys()}
        max_offset = {k: 0 for k in mid_positions.keys()}

        last_print = time.time()
        while True:
            positions = read_positions(servo, DEFAULT_JOINTS, prev=prev_positions)
            for name, pos in positions.items():
                # 计算与上次位置的最短路径位移
                delta = shortest_delta(pos, prev_positions[name])
                cumulative_offset[name] += delta
                prev_positions[name] = pos

                # 更新相对于中位的最小/最大偏移
                if cumulative_offset[name] < min_offset[name]:
                    min_offset[name] = cumulative_offset[name]
                if cumulative_offset[name] > max_offset[name]:
                    max_offset[name] = cumulative_offset[name]

            now = time.time()
            if now - last_print >= 0.1:  # 更高频率更新以获得流畅视觉反馈
                summary = " | ".join(
                    f"{name}: {min_offset[name]:+5d}~{max_offset[name]:+5d}" for name in joint_names
                )
                print(f"\r{summary}", end="", flush=True)
                last_print = now

            if select.select([sys.stdin], [], [], args.interval)[0]:
                _ = sys.stdin.readline()
                break

        print()  # 换行，避免结果覆盖最后一行统计
        result = {}
        for item in DEFAULT_JOINTS:
            name = item["name"]
            mid = mid_positions[name]
            # 根据中位和偏移计算实际的 range_min 和 range_max（mod 4096）
            rng_min = (mid + min_offset[name]) % 4096
            rng_max = (mid + max_offset[name]) % 4096
            
            # 重新计算几何中位（范围弧段的中点）
            span = (rng_max - rng_min) % 4096
            if span == 0:
                computed_home = rng_min
            else:
                computed_home = (rng_min + span // 2) % 4096
            
            result[name] = {
                "id": item["id"],
                "drive_mode": item.get("drive_mode", 0),
                "homing_offset": int(computed_home),
                "range_min": int(rng_min),
                "range_max": int(rng_max),
            }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        print(f"\nDone. Saved calibrated config to {args.output}")
    finally:
        try:
            servo.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
