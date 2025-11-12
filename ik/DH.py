"""
纯 Python 实现的 Denavit-Hartenberg (DH) 参数建模

提供基于标准 DH 和修正 DH 参数的机器人运动学建模。

DH 参数说明：
- 标准 DH (Standard DH): theta, d, a, alpha
- 修正 DH (Modified DH): theta, d, a, alpha (不同的坐标系定义)

变换顺序：
- 标准 DH: Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)
- 修正 DH: Rx(alpha_{i-1}) * Tx(a_{i-1}) * Rz(theta_i) * Tz(d_i)
"""

import numpy as np
from typing import List, Union, Optional
from .et import ET, ETS


class DHRobot:
    """
    基于 DH 参数的机器人模型
    
    使用 Denavit-Hartenberg 参数定义机器人运动学链。
    
    Parameters
    ----------
    dh_params : np.ndarray
        DH 参数矩阵，每行为 [theta, d, a, alpha]
        - theta: 关节角度（变量）或固定角度
        - d: 连杆偏移（变量）或固定偏移
        - a: 连杆长度
        - alpha: 连杆扭转角
    convention : str
        DH 约定，'standard' 或 'modified'
    joint_type : list of str, optional
        每个关节的类型，'R' (旋转) 或 'P' (移动)
        如果未指定，默认所有关节为旋转关节
    qlim : np.ndarray, optional
        关节限位 [min_values; max_values]
        
    Examples
    --------
    >>> # 标准 DH 参数定义 2R 机械臂
    >>> dh = np.array([
    ...     [0,    0,   0.5, 0],      # 关节1: theta=q1
    ...     [0,    0,   0.3, 0]       # 关节2: theta=q2
    ... ])
    >>> robot = DHRobot(dh, convention='standard')
    >>> 
    >>> # 计算正运动学
    >>> q = np.array([0, np.pi/4])
    >>> T = robot.fkine(q)
    """
    
    def __init__(self, dh_params: np.ndarray, 
                 convention: str = 'standard',
                 joint_type: Optional[List[str]] = None,
                 qlim: Optional[np.ndarray] = None):
        """初始化 DH 机器人模型"""
        self.dh_params = np.array(dh_params)
        self.convention = convention.lower()
        
        if self.convention not in ['standard', 'modified']:
            raise ValueError("convention must be 'standard' or 'modified'")
        
        # 关节数量
        self.n = len(dh_params)
        
        # 关节类型（默认全部为旋转关节）
        if joint_type is None:
            self.joint_type = ['R'] * self.n
        else:
            if len(joint_type) != self.n:
                raise ValueError(f"joint_type length ({len(joint_type)}) must match number of joints ({self.n})")
            self.joint_type = joint_type
        
        # 关节限位
        self.qlim = qlim
        
        # 构建 ETS 表示（用于雅可比和逆运动学）
        self.ets = self._build_ets()
        
    def _build_ets(self) -> ETS:
        """
        将 DH 参数转换为 ETS 表示
        
        Returns
        -------
        ETS
            等效的基本变换序列
        """
        et_list = []
        
        if self.convention == 'standard':
            # 标准 DH: Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)
            for i, (theta, d, a, alpha) in enumerate(self.dh_params):
                jtype = self.joint_type[i]
                
                if jtype == 'R':
                    # 旋转关节：theta 是变量
                    et_list.append(ET.Rz())  # Rz(q)
                    if d != 0:
                        et_list.append(ET.tz(d))
                    if a != 0:
                        et_list.append(ET.tx(a))
                    if alpha != 0:
                        et_list.append(ET.Rx(alpha))
                else:  # 'P'
                    # 移动关节：d 是变量
                    if theta != 0:
                        et_list.append(ET.Rz(theta))
                    et_list.append(ET.tz())  # Tz(q)
                    if a != 0:
                        et_list.append(ET.tx(a))
                    if alpha != 0:
                        et_list.append(ET.Rx(alpha))
                        
        else:  # modified
            # 修正 DH: Rx(alpha_{i-1}) * Tx(a_{i-1}) * Rz(theta_i) * Tz(d_i)
            for i, (theta, d, a, alpha) in enumerate(self.dh_params):
                jtype = self.joint_type[i]
                
                if i > 0:
                    # 添加前一个连杆的 alpha 和 a
                    prev_alpha = self.dh_params[i-1, 3]
                    prev_a = self.dh_params[i-1, 2]
                    if prev_alpha != 0:
                        et_list.append(ET.Rx(prev_alpha))
                    if prev_a != 0:
                        et_list.append(ET.tx(prev_a))
                
                if jtype == 'R':
                    # 旋转关节
                    et_list.append(ET.Rz())  # Rz(q)
                    if d != 0:
                        et_list.append(ET.tz(d))
                else:  # 'P'
                    # 移动关节
                    if theta != 0:
                        et_list.append(ET.Rz(theta))
                    et_list.append(ET.tz())  # Tz(q)
            
            # 添加最后一个连杆的 alpha 和 a
            last_alpha = self.dh_params[-1, 3]
            last_a = self.dh_params[-1, 2]
            if last_alpha != 0:
                et_list.append(ET.Rx(last_alpha))
            if last_a != 0:
                et_list.append(ET.tx(last_a))
        
        ets = ETS(et_list)
        if self.qlim is not None:
            ets.qlim = self.qlim
        
        return ets
    
    def fkine(self, q: np.ndarray) -> np.ndarray:
        """
        计算正运动学
        
        Parameters
        ----------
        q : np.ndarray
            关节角度/位移向量
            
        Returns
        -------
        np.ndarray
            4x4 齐次变换矩阵
        """
        return self.ets.fkine(q)
    
    def jacob0(self, q: np.ndarray) -> np.ndarray:
        """
        计算基坐标系雅可比矩阵
        
        Parameters
        ----------
        q : np.ndarray
            关节角度/位移向量
            
        Returns
        -------
        np.ndarray
            6xn 雅可比矩阵
        """
        return self.ets.jacob0(q)
    
    def jacobe(self, q: np.ndarray) -> np.ndarray:
        """
        计算末端坐标系雅可比矩阵
        
        Parameters
        ----------
        q : np.ndarray
            关节角度/位移向量
            
        Returns
        -------
        np.ndarray
            6xn 雅可比矩阵
        """
        return self.ets.jacobe(q)
    
    def manipulability(self, q: np.ndarray, 
                      method: str = 'yoshikawa') -> float:
        """
        计算可操作度
        
        Parameters
        ----------
        q : np.ndarray
            关节角度/位移向量
        method : str
            计算方法：'yoshikawa', 'asada', 'minsingular'
            
        Returns
        -------
        float
            可操作度指标
        """
        return self.ets.manipulability(q, method=method)
    
    def T(self, i: int, q: np.ndarray) -> np.ndarray:
        """
        计算从基座到第 i 个关节的变换矩阵
        
        Parameters
        ----------
        i : int
            关节索引（0 到 n-1）
        q : np.ndarray
            关节角度/位移向量
            
        Returns
        -------
        np.ndarray
            4x4 齐次变换矩阵
        """
        if i < 0 or i >= self.n:
            raise ValueError(f"Joint index must be between 0 and {self.n-1}")
        
        T = np.eye(4)
        
        if self.convention == 'standard':
            for j in range(i + 1):
                theta, d, a, alpha = self.dh_params[j]
                jtype = self.joint_type[j]
                
                if jtype == 'R':
                    theta = theta + q[j]
                else:  # 'P'
                    d = d + q[j]
                
                # 标准 DH 变换
                T = T @ self._standard_dh_matrix(theta, d, a, alpha)
        else:  # modified
            for j in range(i + 1):
                theta, d, a, alpha = self.dh_params[j]
                jtype = self.joint_type[j]
                
                if jtype == 'R':
                    theta = theta + q[j]
                else:  # 'P'
                    d = d + q[j]
                
                # 修正 DH 变换
                T = T @ self._modified_dh_matrix(theta, d, a, alpha, 
                                                 j > 0 and self.dh_params[j-1] or None)
        
        return T
    
    @staticmethod
    def _standard_dh_matrix(theta: float, d: float, 
                           a: float, alpha: float) -> np.ndarray:
        """
        标准 DH 变换矩阵
        
        T = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        return np.array([
            [ct,    -st*ca,   st*sa,   a*ct],
            [st,     ct*ca,  -ct*sa,   a*st],
            [0,      sa,      ca,      d   ],
            [0,      0,       0,       1   ]
        ])
    
    @staticmethod
    def _modified_dh_matrix(theta: float, d: float, 
                           a: float, alpha: float,
                           prev_params: Optional[np.ndarray] = None) -> np.ndarray:
        """
        修正 DH 变换矩阵
        
        T = Rx(alpha_{i-1}) * Tx(a_{i-1}) * Rz(theta_i) * Tz(d_i)
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        if prev_params is not None:
            prev_alpha = prev_params[3]
            prev_a = prev_params[2]
            cpa = np.cos(prev_alpha)
            spa = np.sin(prev_alpha)
        else:
            prev_alpha = 0
            prev_a = 0
            cpa = 1
            spa = 0
        
        return np.array([
            [ct,     -st,     0,      prev_a  ],
            [st*cpa,  ct*cpa, -spa,  -d*spa  ],
            [st*spa,  ct*spa,  cpa,   d*cpa  ],
            [0,       0,       0,      1      ]
        ])
    
    def __str__(self) -> str:
        """字符串表示"""
        s = f"DHRobot ({self.convention} DH, {self.n} joints)\n"
        s += "DH Parameters:\n"
        s += "  theta      d        a        alpha    type\n"
        for i, (theta, d, a, alpha) in enumerate(self.dh_params):
            jtype = self.joint_type[i]
            s += f"  {theta:7.4f}  {d:7.4f}  {a:7.4f}  {alpha:7.4f}  {jtype}\n"
        return s
    
    def __repr__(self) -> str:
        return self.__str__()


def create_dh_robot(dh_params: Union[np.ndarray, List[List[float]]],
                   convention: str = 'standard',
                   joint_type: Optional[List[str]] = None,
                   qlim: Optional[np.ndarray] = None) -> DHRobot:
    """
    创建 DH 机器人的便捷函数
    
    Parameters
    ----------
    dh_params : array-like
        DH 参数矩阵，每行为 [theta, d, a, alpha]
    convention : str
        DH 约定，'standard' 或 'modified'
    joint_type : list of str, optional
        关节类型列表，'R' 或 'P'
    qlim : np.ndarray, optional
        关节限位
        
    Returns
    -------
    DHRobot
        DH 机器人模型
        
    Examples
    --------
    >>> # 创建 PUMA 560 机械臂（标准 DH）
    >>> dh_puma = [
    ...     [0,        0,       0,       np.pi/2],
    ...     [0,        0,       0.4318,  0      ],
    ...     [0,        0.15005, 0.0203,  -np.pi/2],
    ...     [0,        0.4318,  0,       np.pi/2],
    ...     [0,        0,       0,       -np.pi/2],
    ...     [0,        0,       0,       0      ]
    ... ]
    >>> puma = create_dh_robot(dh_puma, convention='standard')
    """
    return DHRobot(dh_params, convention=convention, 
                  joint_type=joint_type, qlim=qlim)


# 预定义的经典机器人模型
def create_puma560() -> DHRobot:
    """
    创建 PUMA 560 机械臂模型（标准 DH）
    
    Returns
    -------
    DHRobot
        PUMA 560 机器人模型
    """
    dh_params = np.array([
        [0,        0,       0,       np.pi/2],
        [0,        0,       0.4318,  0      ],
        [0,        0.15005, 0.0203,  -np.pi/2],
        [0,        0.4318,  0,       np.pi/2],
        [0,        0,       0,       -np.pi/2],
        [0,        0,       0,       0      ]
    ])
    
    qlim = np.array([
        [-160, -110, -135, -266, -100, -266],
        [ 160,  110,  135,  266,  100,  266]
    ]) * np.pi / 180  # 转换为弧度
    
    return DHRobot(dh_params, convention='standard', qlim=qlim)


def create_stanford_arm() -> DHRobot:
    """
    创建 Stanford 机械臂模型（包含移动关节）
    
    Returns
    -------
    DHRobot
        Stanford 机器人模型
    """
    dh_params = np.array([
        [0,        0.412,   0,       -np.pi/2],
        [0,        0.154,   0,        np.pi/2],
        [0,        0,       0,        0      ],  # 移动关节
        [0,        0,       0,       -np.pi/2],
        [0,        0,       0,        np.pi/2],
        [0,        0.263,   0,        0      ]
    ])
    
    joint_type = ['R', 'R', 'P', 'R', 'R', 'R']
    
    qlim = np.array([
        [-170, -170, 0,    -170, -120, -170],
        [ 170,  170, 0.5,  170,  120,  170]
    ])
    qlim[:2] = qlim[:2] * np.pi / 180  # 旋转关节转换为弧度
    
    return DHRobot(dh_params, convention='standard', 
                  joint_type=joint_type, qlim=qlim)


if __name__ == "__main__":
    """测试 DH 机器人建模"""
    print("=" * 60)
    print("测试 DH 机器人建模")
    print("=" * 60)
    
    # 1. 测试简单 2R 机械臂
    print("\n1. 简单 2R 平面机械臂")
    dh_2r = np.array([
        [0, 0, 0.5, 0],
        [0, 0, 0.3, 0]
    ])
    robot_2r = DHRobot(dh_2r, convention='standard')
    print(robot_2r)
    
    q = np.array([0, np.pi/4])
    T = robot_2r.fkine(q)
    print(f"q = {q}")
    print("正运动学结果:")
    print(np.round(T, 4))
    
    # 2. 测试 PUMA 560
    print("\n" + "=" * 60)
    print("2. PUMA 560 机械臂")
    puma = create_puma560()
    print(puma)
    
    q_puma = np.zeros(6)
    T_puma = puma.fkine(q_puma)
    print(f"q = {q_puma}")
    print("正运动学结果:")
    print(np.round(T_puma, 4))
    
    # 3. 测试雅可比矩阵
    print("\n" + "=" * 60)
    print("3. 雅可比矩阵计算")
    J0 = robot_2r.jacob0(q)
    print(f"基坐标系雅可比矩阵 (q={q}):")
    print(np.round(J0, 4))
    
    # 4. 测试可操作度
    print("\n" + "=" * 60)
    print("4. 可操作度计算")
    m = robot_2r.manipulability(q)
    print(f"可操作度 (q={q}): {m:.6f}")
    
    print("\n✅ 所有测试完成！")
