"""
简化的逆运动学求解器

提供 LM, GN, NR, QP 四种数值逆运动学求解方法
"""

import numpy as np


class IKResult:
    """IK 求解结果封装类"""
    def __init__(self, success, q, reason=""):
        self.success = success
        self.q = q
        self.reason = reason


def _angle_axis(Te, Tep):
    """计算两个位姿之间的角轴误差"""
    e = np.zeros(6)
    e[:3] = Tep[:3, 3] - Te[:3, 3]  # 位置误差
    
    # 旋转误差
    R_err = Tep[:3, :3] @ Te[:3, :3].T
    li = np.array([R_err[2,1] - R_err[1,2], 
                   R_err[0,2] - R_err[2,0], 
                   R_err[1,0] - R_err[0,1]])
    li_norm = np.linalg.norm(li)
    
    if li_norm < 1e-6:
        if np.trace(R_err) > 0:
            e[3:] = np.zeros(3)
        else:
            e[3:] = np.pi / 2 * (np.diag(R_err) + 1)
    else:
        angle = np.arctan2(li_norm, np.trace(R_err) - 1)
        e[3:] = angle * li / li_norm
    
    return e


def ikine_LM(ets, Tep, q0=None, ilimit=100, slimit=10, tol=1e-3, mask=None, 
             k=1.0, method='chan'):
    """
    使用 Levenberg-Marquardt 方法求解逆运动学
    
    Parameters
    ----------
    ets : ETS
        机器人运动学链
    Tep : np.ndarray
        目标位姿 (4x4 齐次变换矩阵)
    q0 : array_like, optional
        初始关节角度，默认为零向量
    ilimit : int
        最大迭代次数
    slimit : int
        搜索次数限制
    tol : float
        收敛容差
    mask : array_like, optional
        位姿权重 [x, y, z, roll, pitch, yaw]，0 表示忽略该维度
    k : float
        LM 阻尼系数
    method : str
        LM 更新方法 ('chan', 'wampler', 'sugihara')
        
    Returns
    -------
    IKResult
        求解结果，包含 .success, .q, .reason 属性
    """
    n = ets.n
    
    if q0 is None:
        q0 = np.zeros(n)
    else:
        q0 = np.asarray(q0)
    
    if mask is None:
        mask = np.ones(6)
    else:
        mask = np.asarray(mask)
    
    We = np.diag(mask)
    method = method.lower()
    
    # 多次搜索尝试
    for search in range(slimit):
        # 初始化 (第一次用 q0，后面用随机)
        if search == 0:
            q = q0.copy()
        else:
            q = np.random.uniform(-np.pi, np.pi, n)
        
        # 迭代优化
        for i in range(ilimit):
            try:
                # 计算当前位姿和误差
                Te = ets.eval(q)
                e = _angle_axis(Te, Tep)
                E = 0.5 * e @ We @ e
                
                # 检查收敛
                if E < tol:
                    q = (q + np.pi) % (2 * np.pi) - np.pi  # 归一化
                    return IKResult(True, q, "Success")
                
                # 计算雅可比
                J = ets.jacob0(q)
                
                # 计算阻尼矩阵
                if method.startswith('sugi'):
                    Wn = E * np.eye(n) + k * np.eye(n)
                elif method.startswith('wamp'):
                    Wn = k * np.eye(n)
                else:  # chan
                    Wn = k * E * np.eye(n)
                
                # LM 更新: q += (J^T W J + Wn)^{-1} J^T W e
                JtW = J.T @ We
                A = JtW @ J + Wn
                g = JtW @ e
                q += np.linalg.solve(A, g)
                
            except np.linalg.LinAlgError:
                break  # 矩阵奇异，换个初始值重试
    
    return IKResult(False, q, "iteration limit reached")


def ikine_GN(ets, Tep, q0=None, ilimit=50, tol=1e-3, mask=None, pinv=False):
    """
    使用 Gauss-Newton 方法求解逆运动学
    
    Parameters
    ----------
    ets : ETS
        机器人运动学链
    Tep : np.ndarray
        目标位姿 (4x4 齐次变换矩阵)
    q0 : array_like, optional
        初始关节角度
    ilimit : int
        最大迭代次数
    tol : float
        收敛容差
    mask : array_like, optional
        位姿权重
    pinv : bool
        是否使用伪逆
        
    Returns
    -------
    IKResult
        求解结果
    """
    n = ets.n
    
    if q0 is None:
        q0 = np.zeros(n)
    else:
        q0 = np.asarray(q0)
    
    if mask is None:
        mask = np.ones(6)
    else:
        mask = np.asarray(mask)
    
    We = np.diag(mask)
    q = q0.copy()
    
    for i in range(ilimit):
        try:
            Te = ets.eval(q)
            e = _angle_axis(Te, Tep)
            E = 0.5 * e @ We @ e
            
            if E < tol:
                q = (q + np.pi) % (2 * np.pi) - np.pi
                return IKResult(True, q, "Success")
            
            J = ets.jacob0(q)
            JtW = J.T @ We
            
            if pinv:
                # 伪逆: q += J^+ W e
                q += np.linalg.pinv(J) @ We @ e
            else:
                # 正规方程: q += (J^T W J)^{-1} J^T W e
                q += np.linalg.solve(JtW @ J, JtW @ e)
                
        except np.linalg.LinAlgError:
            break
    
    return IKResult(False, q, "iteration limit reached")


def ikine_NR(ets, Tep, q0=None, ilimit=50, tol=1e-3, mask=None, pinv=False):
    """
    使用 Newton-Raphson 方法求解逆运动学
    在本实现中与 Gauss-Newton 等价
    """
    return ikine_GN(ets, Tep, q0, ilimit, tol, mask, pinv)


def ikine_QP(ets, Tep, q0=None, ilimit=50, tol=1e-3, mask=None, 
             kj=0.01, ks=1.0):
    """
    使用二次规划方法求解逆运动学 (简化版)
    
    本方法在简化版中退化为带关节正则化的 GN 方法
    
    Parameters
    ----------
    ets : ETS
        机器人运动学链
    Tep : np.ndarray
        目标位姿
    q0 : array_like, optional
        初始关节角度
    ilimit : int
        最大迭代次数
    tol : float
        收敛容差
    mask : array_like, optional
        位姿权重
    kj : float
        关节正则化系数
    ks : float
        步长缩放系数
        
    Returns
    -------
    IKResult
        求解结果
    """
    n = ets.n
    
    if q0 is None:
        q0 = np.zeros(n)
    else:
        q0 = np.asarray(q0)
    
    if mask is None:
        mask = np.ones(6)
    else:
        mask = np.asarray(mask)
    
    We = np.diag(mask)
    q = q0.copy()
    
    for i in range(ilimit):
        try:
            Te = ets.eval(q)
            e = _angle_axis(Te, Tep)
            E = 0.5 * e @ We @ e
            
            if E < tol:
                q = (q + np.pi) % (2 * np.pi) - np.pi
                return IKResult(True, q, "Success")
            
            J = ets.jacob0(q)
            JtW = J.T @ We
            # 带正则化: (J^T W J + kj*I)^{-1} J^T W e
            A = JtW @ J + kj * np.eye(n)
            dq = np.linalg.solve(A, JtW @ e)
            q += ks * dq
                
        except np.linalg.LinAlgError:
            break
    
    return IKResult(False, q, "iteration limit reached")
