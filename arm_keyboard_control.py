from pynput import keyboard
import threading
import time
from ftservo_controller import ServoController


class ArmKeyboardController:
    def __init__(self, port="/dev/ttyACM0", baudrate=1000000, config_path="servo_config.json", step=100):
        self.controller = ServoController(port, baudrate, config_path)
        self.step = step
        self.running = True

        # 初始化当前舵机位置
        ids = [cfg["id"] for cfg in self.controller.config.values()]
        responses = self.controller.servo.sync_read(0x38, 2, ids)
        self.current_pos = {}
        for name, cfg in self.controller.config.items():
            sid = cfg["id"]
            if sid in responses:
                params = responses[sid]
                pos = params[0] + (params[1] << 8)
            else:
                pos = self.controller.get_home_position(name)
            self.current_pos[name] = pos

        print("✅ 已初始化机械臂键盘控制器")
        self.print_controls()

    def print_controls(self):
        print("\n🎮 控制说明：")
        print("  shoulder_pan:   q(+), a(-)")
        print("  shoulder_lift:  w(+), s(-)")
        print("  elbow_flex:     e(+), d(-)")
        print("  wrist_flex:     r(+), f(-)")
        print("  wrist_roll:     t(+), g(-)")
        print("  gripper:        y(+), h(-)")
        print("  回中位:         m")
        print("  退出:           ESC\n")

    def update_joint(self, name, delta):
        new_pos = self.current_pos[name] + delta
        new_pos = self.controller.limit_position(name, new_pos)
        self.current_pos[name] = new_pos
        print(f"→ {name}: {new_pos}")

        sid = self.controller.config[name]["id"]
        data = [
            new_pos & 0xFF, (new_pos >> 8) & 0xFF,
            0x00, 0x00,
            0xE8, 0x03
        ]
        self.controller.servo.sync_write(0x2A, 6, {sid: data})

    def reset_to_home(self):
        print("🏠 回中位中...")
        self.controller.soft_move_to_home(step_count=10, interval=0.15)
        for name in self.current_pos.keys():
            self.current_pos[name] = self.controller.get_home_position(name)

    # ------------------------
    # 键盘监听部分
    # ------------------------
    def on_press(self, key):
        try:
            k = key.char.lower()  # 转为小写字符
        except AttributeError:
            if key == keyboard.Key.esc:
                print("\n🛑 已退出控制")
                self.running = False
            return

        mapping = {
            "q": ("shoulder_pan", +self.step),
            "a": ("shoulder_pan", -self.step),
            "w": ("shoulder_lift", +self.step),
            "s": ("shoulder_lift", -self.step),
            "e": ("elbow_flex", +self.step),
            "d": ("elbow_flex", -self.step),
            "r": ("wrist_flex", +self.step),
            "f": ("wrist_flex", -self.step),
            "t": ("wrist_roll", +self.step),
            "g": ("wrist_roll", -self.step),
            "y": ("gripper", +self.step),
            "h": ("gripper", -self.step),
        }

        if k in mapping:
            joint, delta = mapping[k]
            self.update_joint(joint, delta)
        elif k == "m":
            self.reset_to_home()

    def run(self):
        print("🕹️ 开始键盘控制（按 ESC 退出）")
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

        while self.running:
            time.sleep(0.05)

        listener.stop()
        self.controller.close()


if __name__ == "__main__":
    arm_ctrl = ArmKeyboardController("/dev/ttyACM0", 1000000, "servo_config.json", step=50)
    arm_ctrl.run()
