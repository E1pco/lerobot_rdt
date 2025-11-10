#!/usr/bin/env python3
"""
JoyCon IK Control - Test Script (No Hardware Required)
=======================================================

Tests the JoyCon to IK pipeline without real hardware.
Simulates Joy-Con input and validates IK solving.

Usage:
    python test_joycon_ik.py
"""

import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

# Add path for joyconrobotics
sys.path.insert(0, '/home/elpco/code/lerobot/joycon-robotics')

# Import robot kinematics
from lerobot_kinematics.ET import ET


def create_so101_5dof():
    """Create SO-101 5DOF robot model"""
    E1 = ET.Rz()
    E2 = ET.tx(0.0612)
    E3 = ET.tz(0.0598)
    E4 = ET.tx(0.02943)
    E5 = ET.tz(0.05504)
    E6 = ET.Ry()
    E7 = ET.tz(0.1127)
    E8 = ET.tx(0.02798)
    E9 = ET.Ry()
    E10 = ET.tx(0.13504)
    E11 = ET.tz(0.00519)
    E12 = ET.Ry()
    E13 = ET.tx(0.0593)
    E14 = ET.tz(0.00996)
    E15 = ET.Rx()
    
    robot = E1 * E2 * E3 * E4 * E5 * E6 * E7 * E8 * E9 * E10 * E11 * E12 * E13 * E14 * E15
    robot.qlim = np.array([
        [-1.91986, -1.74533, -1.69, -1.65806, -2.74385],
        [ 1.91986,  1.74533,  1.69,  1.65806,  2.84121]
    ])
    return robot


def build_target_pose(x, y, z, roll, pitch, yaw):
    """Build 4x4 transformation matrix"""
    T = np.eye(4)
    T[:3, :3] = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def test_joycon_data_extraction():
    """Test 1: Simulate JoyCon data extraction"""
    print("\n" + "=" * 70)
    print("Test 1: JoyCon Data Extraction")
    print("=" * 70)
    
    # Simulated JoyCon pose data (from get_control())
    # Format: [x, y, z, roll, pitch, yaw]
    simulated_pose = [0.15, 0.05, 0.20, 0.0, np.pi/6, np.pi/12]
    
    print(f"\n📱 Simulated Joy-Con data:")
    print(f"   Position: [{simulated_pose[0]:.3f}, {simulated_pose[1]:.3f}, {simulated_pose[2]:.3f}] m")
    print(f"   Rotation: [{np.degrees(simulated_pose[3]):.1f}°, "
          f"{np.degrees(simulated_pose[4]):.1f}°, {np.degrees(simulated_pose[5]):.1f}°]")
    
    # Extract position and orientation
    x, y, z = simulated_pose[0:3]
    roll, pitch, yaw = simulated_pose[3:6]
    
    # Build target pose matrix
    T_target = build_target_pose(x, y, z, roll, pitch, yaw)
    
    print(f"\n🎯 Target transformation matrix:")
    print(np.round(T_target, 4))
    print(f"\n✓ Test 1 passed: Successfully extracted pose data")
    
    return T_target


def test_ik_solving(T_target):
    """Test 2: Solve IK for target pose"""
    print("\n" + "=" * 70)
    print("Test 2: Inverse Kinematics Solving")
    print("=" * 70)
    
    # Create robot model
    robot = create_so101_5dof()
    print(f"\n🤖 Robot model created:")
    print(f"   DOF: 5")
    print(f"   Joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll")
    
    # Initial joint configuration (home position = all zeros)
    q0 = np.zeros(5)
    
    print(f"\n📐 Initial joint angles:")
    print(f"   {np.round(np.degrees(q0), 2)} deg")
    
    # Solve IK
    print(f"\n⚙️ Solving IK...")
    sol = robot.ikine_LM(
        Tep=T_target,
        q0=q0,
        ilimit=50,
        slimit=3,
        tol=1e-3,
        mask=np.array([1, 1, 1, 0, 0.8, 0.8]),
        k=0.1,
        method="sugihara"
    )
    
    if sol.success:
        print(f"✓ IK solution found!")
        print(f"\n📊 Solution joint angles:")
        print(f"   {np.round(np.degrees(sol.q), 2)} deg")
        print(f"   {np.round(sol.q, 4)} rad")
        
        # Verify solution with forward kinematics
        T_result = robot.fkine(sol.q).A
        pos_result = T_result[:3, 3]
        rpy_result = R.from_matrix(T_result[:3, :3]).as_euler('xyz')
        
        print(f"\n🔍 Verification (Forward Kinematics):")
        print(f"   Position: {np.round(pos_result, 4)} m")
        print(f"   RPY: {np.round(np.degrees(rpy_result), 2)} deg")
        
        # Calculate error
        pos_error = np.linalg.norm(T_target[:3, 3] - pos_result)
        print(f"\n📏 Position error: {pos_error:.6f} m")
        
        if pos_error < 0.001:
            print(f"✓ Test 2 passed: IK solution verified")
        else:
            print(f"⚠ Test 2 warning: Position error is high")
        
        return sol.q
    else:
        print(f"❌ IK solution failed: {sol.reason}")
        print(f"❌ Test 2 failed")
        return None


def test_servo_conversion(q_rad):
    """Test 3: Convert joint angles to servo positions"""
    print("\n" + "=" * 70)
    print("Test 3: Joint Angle to Servo Position Conversion")
    print("=" * 70)
    
    if q_rad is None:
        print("⚠ Skipping test 3 (no IK solution)")
        return
    
    # Home position map
    home_pose = {
        "shoulder_pan": 2096,
        "shoulder_lift": 1983,
        "elbow_flex": 2100,
        "wrist_flex": 1954,
        "wrist_roll": 2048,
    }
    
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", 
                   "wrist_flex", "wrist_roll"]
    
    # Conversion parameters
    counts_per_rev = 4096
    counts_per_rad = counts_per_rev / (2 * np.pi)
    
    print(f"\n🔧 Conversion parameters:")
    print(f"   Counts per revolution: {counts_per_rev}")
    print(f"   Counts per radian: {counts_per_rad:.4f}")
    
    print(f"\n📋 Servo target positions:")
    print(f"   {'Joint':<20} {'Angle (deg)':<15} {'Angle (rad)':<15} {'Home (steps)':<15} {'Target (steps)'}")
    print("   " + "-" * 90)
    
    for i, name in enumerate(joint_names):
        angle_deg = np.degrees(q_rad[i])
        angle_rad = q_rad[i]
        home_pos = home_pose[name]
        target_pos = int(round(home_pos + angle_rad * counts_per_rad))
        delta = target_pos - home_pos
        
        print(f"   {name:<20} {angle_deg:>13.2f}° {angle_rad:>14.4f} {home_pos:>14d} "
              f"{target_pos:>14d} (Δ{delta:+d})")
    
    print(f"\n✓ Test 3 passed: Servo positions calculated")


def test_multiple_poses():
    """Test 4: Test multiple target poses"""
    print("\n" + "=" * 70)
    print("Test 4: Multiple Target Poses")
    print("=" * 70)
    
    robot = create_so101_5dof()
    
    # Test poses simulating Joy-Con movements
    test_cases = [
        ("Home position", 0.15, 0.00, 0.20, 0.0, np.pi/4, 0.0),
        ("Left reach", 0.12, 0.08, 0.18, 0.0, np.pi/6, np.pi/6),
        ("Right reach", 0.12, -0.08, 0.18, 0.0, np.pi/6, -np.pi/6),
        ("High position", 0.10, 0.00, 0.25, 0.0, -np.pi/6, 0.0),
        ("Forward reach", 0.20, 0.00, 0.15, 0.0, np.pi/3, 0.0),
    ]
    
    success_count = 0
    
    for name, x, y, z, roll, pitch, yaw in test_cases:
        print(f"\n  Testing: {name}")
        print(f"    Target: pos=[{x:.2f}, {y:.2f}, {z:.2f}], "
              f"rpy=[{np.degrees(roll):.1f}°, {np.degrees(pitch):.1f}°, {np.degrees(yaw):.1f}°]")
        
        T_target = build_target_pose(x, y, z, roll, pitch, yaw)
        sol = robot.ikine_LM(
            Tep=T_target,
            q0=np.zeros(5),
            ilimit=50,
            slimit=3,
            tol=1e-3,
            mask=np.array([1, 1, 1, 0, 0.8, 0.8]),
            k=0.1,
            method="sugihara"
        )
        
        if sol.success:
            # Verify
            T_result = robot.fkine(sol.q).A
            pos_error = np.linalg.norm(T_target[:3, 3] - T_result[:3, 3])
            print(f"    ✓ Solution found (error: {pos_error:.6f} m)")
            success_count += 1
        else:
            print(f"    ❌ Failed: {sol.reason}")
    
    print(f"\n📊 Results: {success_count}/{len(test_cases)} poses solved successfully")
    
    if success_count == len(test_cases):
        print(f"✓ Test 4 passed: All poses reachable")
    else:
        print(f"⚠ Test 4 warning: Some poses unreachable")


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("JoyCon-IK Integration Test Suite")
    print("=" * 70)
    print("\nTesting the pipeline without hardware:")
    print("  JoyCon Pose Data → IK Solver → Servo Commands")
    
    try:
        # Test 1: Extract pose data
        T_target = test_joycon_data_extraction()
        
        # Test 2: Solve IK
        q_solution = test_ik_solving(T_target)
        
        # Test 3: Convert to servo positions
        test_servo_conversion(q_solution)
        
        # Test 4: Multiple poses
        test_multiple_poses()
        
        print("\n" + "=" * 70)
        print("✓ All Tests Complete")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Connect Joy-Con controller")
        print("  2. Connect robot servos to /dev/ttyACM0")
        print("  3. Run: python joycon_ik_control.py --device right")
        print("=" * 70 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
