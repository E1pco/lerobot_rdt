

import numpy as np
from typing import List, Optional, Union
from enum import Enum


class ETType(Enum):
    """基本变换类型"""
    RX, RY, RZ = 'Rx', 'Ry', 'Rz'  # 旋转
    TX, TY, TZ = 'tx', 'ty', 'tz'  # 平移


class ET:
    """
    基本变换（Elementary Transform）
    
    表示单个刚体变换，可以是平移或旋转。
    """
    
    def __init__(self, et_type: ETType, value: Optional[float] = None, 
                 joint: bool = False, flip: bool = False):
        self.et_type = et_type
        self._value = value
        self.joint = joint if value is None else False
        self.flip = flip
        
    @property
    def isjoint(self) -> bool:
        return self.joint
    
    @property
    def value(self) -> Optional[float]:
        return self._value
    
    def __str__(self) -> str:
        if self.joint:
            return f"{self.et_type.value}(q)"
        return f"{self.et_type.value}({self._value or 0:.4f})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __mul__(self, other):
        if isinstance(other, ET):
            return ETS([self, other])
        elif isinstance(other, ETS):
            return ETS([self] + other.ets)
        raise TypeError(f"Cannot multiply ET with {type(other)}")
    
    def T(self, q: Optional[float] = None) -> np.ndarray:
        """计算齐次变换矩阵"""
        if self.joint:
            if q is None:
                raise ValueError("Joint variable requires a value")
            theta = -q if self.flip else q
        else:
            theta = self._value or 0.0
            if self.flip:
                theta = -theta
        
        T = np.eye(4)
        c, s = np.cos(theta), np.sin(theta)
        
        if self.et_type == ETType.RX:
            T[1:3, 1:3] = [[c, -s], [s, c]]
        elif self.et_type == ETType.RY:
            T[0, 0], T[0, 2], T[2, 0], T[2, 2] = c, s, -s, c
        elif self.et_type == ETType.RZ:
            T[0:2, 0:2] = [[c, -s], [s, c]]
        elif self.et_type == ETType.TX:
            T[0, 3] = theta
        elif self.et_type == ETType.TY:
            T[1, 3] = theta
        elif self.et_type == ETType.TZ:
            T[2, 3] = theta
        return T
    
    # 工厂方法
    @staticmethod
    def Rx(q: Optional[float] = None, flip: bool = False):
        return ET(ETType.RX, value=q, joint=(q is None), flip=flip)
    
    @staticmethod
    def Ry(q: Optional[float] = None, flip: bool = False):
        return ET(ETType.RY, value=q, joint=(q is None), flip=flip)
    
    @staticmethod
    def Rz(q: Optional[float] = None, flip: bool = False):
        return ET(ETType.RZ, value=q, joint=(q is None), flip=flip)
    
    @staticmethod
    def tx(d: float = 0.0, flip: bool = False):
        return ET(ETType.TX, value=d, joint=False, flip=flip)
    
    @staticmethod
    def ty(d: float = 0.0, flip: bool = False):
        return ET(ETType.TY, value=d, joint=False, flip=flip)
    
    @staticmethod
    def tz(d: float = 0.0, flip: bool = False):
        return ET(ETType.TZ, value=d, joint=False, flip=flip)


class ETS:
    """
    基本变换序列（Elementary Transform Sequence）
    
    表示一系列基本变换的组合，用于描述机器人的运动学链。
    """
    
    def __init__(self, ets: List[ET]):
        self.ets = ets
        self._qlim = None
        self._n = sum(1 for et in ets if et.isjoint)
        self._joint_indices = [i for i, et in enumerate(ets) if et.isjoint]
    
    @property
    def n(self) -> int:
        return self._n
    
    @property
    def qlim(self) -> Optional[np.ndarray]:
        return self._qlim
    
    @qlim.setter
    def qlim(self, limits: np.ndarray):
        if limits.shape != (2, self.n):
            raise ValueError(f"qlim must be 2 x {self.n}, got {limits.shape}")
        self._qlim = limits
    
    def __str__(self) -> str:
        return " * ".join(str(et) for et in self.ets)
    
    def __repr__(self) -> str:
        return f"ETS({self.__str__()})"
    
    def __mul__(self, other):
        if isinstance(other, ETS):
            return ETS(self.ets + other.ets)
        elif isinstance(other, ET):
            return ETS(self.ets + [other])
        raise TypeError(f"Cannot multiply ETS with {type(other)}")
    
    def __len__(self) -> int:
        return len(self.ets)
    
    def __getitem__(self, idx) -> ET:
        return self.ets[idx]
    
    def fkine(self, q: np.ndarray) -> np.ndarray:
        """计算正运动学"""
        if len(q) != self.n:
            raise ValueError(f"q must have length {self.n}, got {len(q)}")
        
        T, q_idx = np.eye(4), 0
        for et in self.ets:
            if et.isjoint:
                T = T @ et.T(q[q_idx])
                q_idx += 1
            else:
                T = T @ et.T()
        return T
    
    def eval(self, q: np.ndarray) -> np.ndarray:
        """计算正运动学（fkine 的别名）"""
        return self.fkine(q)
    
    def jacob0(self, q: np.ndarray, tool: Optional[np.ndarray] = None) -> np.ndarray:
        """计算基座标系下的雅可比矩阵"""
        if len(q) != self.n:
            raise ValueError(f"q must have length {self.n}, got {len(q)}")
        
        J = np.zeros((6, self.n))
        T_end = self.fkine(q)
        if tool is not None:
            T_end = T_end @ tool
        p_end = T_end[:3, 3]
        
        T_current, q_idx = np.eye(4), 0
        axis_map = {ETType.RX: 0, ETType.RY: 1, ETType.RZ: 2,
                    ETType.TX: 0, ETType.TY: 1, ETType.TZ: 2}
        
        for et in self.ets:
            if et.isjoint:
                p_joint = T_current[:3, 3]
                axis_idx = axis_map[et.et_type]
                z = T_current[:3, axis_idx]
                
                if et.et_type in (ETType.RX, ETType.RY, ETType.RZ):
                    J[:3, q_idx] = np.cross(z, p_end - p_joint)
                    J[3:, q_idx] = z
                else:
                    J[:3, q_idx] = z
                
                T_current = T_current @ et.T(q[q_idx])
                q_idx += 1
            else:
                T_current = T_current @ et.T()
        return J
    
    def jacobe(self, q: np.ndarray, tool: Optional[np.ndarray] = None) -> np.ndarray:
        """计算末端坐标系下的雅可比矩阵"""
        J0 = self.jacob0(q, tool)
        T = self.fkine(q)
        if tool is not None:
            T = T @ tool
        
        R = T[:3, :3].T
        R_block = np.zeros((6, 6))
        R_block[:3, :3] = R_block[3:, 3:] = R
        return R_block @ J0
    
    def hessian0(self, q: np.ndarray, J0: Optional[np.ndarray] = None) -> np.ndarray:
        """计算基座标系下的Hessian矩阵"""
        H, eps = np.zeros((6, self.n, self.n)), 1e-6
        
        for i in range(self.n):
            q_plus, q_minus = q.copy(), q.copy()
            q_plus[i] += eps
            q_minus[i] -= eps
            H[:, :, i] = (self.jacob0(q_plus) - self.jacob0(q_minus)) / (2 * eps)
        return H
    
    def manipulability(self, q: np.ndarray, method: str = 'yoshikawa') -> float:
        """计算可操作度"""
        J = self.jacob0(q)
        
        if method == 'yoshikawa':
            return np.sqrt(np.linalg.det(J @ J.T))
        elif method in ('asada', 'minsingular'):
            return np.min(np.linalg.svd(J, compute_uv=False))
        raise ValueError(f"Unknown method: {method}")


# 辅助函数
def SE3_to_xyzrpy(T: np.ndarray) -> np.ndarray:
    """将齐次变换矩阵转换为 [x, y, z, roll, pitch, yaw]"""
    x, y, z = T[:3, 3]
    R = T[:3, :3]
    
    beta = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    cb = np.cos(beta)
    
    if np.abs(cb) > 1e-6:
        alpha = np.arctan2(R[1, 0] / cb, R[0, 0] / cb)
        gamma = np.arctan2(R[2, 1] / cb, R[2, 2] / cb)
    else:
        alpha, gamma = 0, np.arctan2(R[0, 1], R[1, 1])
    
    return np.array([x, y, z, gamma, beta, alpha])


def xyzrpy_to_SE3(pose: np.ndarray) -> np.ndarray:
    """将 [x, y, z, roll, pitch, yaw] 转换为齐次变换矩阵"""
    x, y, z, roll, pitch, yaw = pose
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    T = np.eye(4)
    T[:3, :3] = [
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr           ]
    ]
    T[:3, 3] = [x, y, z]
    return T
