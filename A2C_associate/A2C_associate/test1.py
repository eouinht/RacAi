import math
import numpy as np
import cvxpy as cp

# ----------------------------
# 1) Breakpoints cho x và y = f(x)
# ----------------------------
def f(x):
    return 1.0 / math.log2(1.0 + 2.0 * x)

x_min = 0.05   # phải > 0 vì f(0)=inf
x_max = 5.0
N = 20         # số đoạn (càng lớn càng chính xác)

xs = np.linspace(x_min, x_max, N+1)
ys = np.array([f(float(x)) for x in xs])

# ----------------------------
# 2) MILP trong CVXPY (chọn đúng 1 đoạn)
#    Dùng biến z_i (boolean) chọn đoạn i, và s_i (0..1) nội suy trong đoạn đó
# ----------------------------
x = cp.Variable()
t = cp.Variable(nonneg=True)

z = cp.Variable(N, boolean=True)   # z_i = 1 nếu chọn đoạn i
s = cp.Variable(N)                 # s_i ∈ [0,1], nhưng chỉ "active" khi z_i=1

constraints = []

# Chọn đúng 1 đoạn
constraints += [cp.sum(z) == 1]

# Kích hoạt s_i chỉ khi z_i = 1
constraints += [s >= 0, s <= z]    # nếu z_i=0 => s_i<=0 => s_i=0; nếu z_i=1 => 0<=s_i<=1

# Biểu diễn x là nội suy tuyến tính trên đoạn được chọn:
# x = sum_i [ z_i * x_i + s_i * (x_{i+1}-x_i) ]
dx = xs[1:] - xs[:-1]
constraints += [
    x == z @ xs[:-1] + s @ dx,
    x >= x_min,
    x <= x_max
]

# Ràng buộc t >= nội suy tuyến tính của f(x):
# t >= sum_i [ z_i * y_i + s_i * (y_{i+1}-y_i) ]
dy = ys[1:] - ys[:-1]
constraints += [
    t >= z @ ys[:-1] + s @ dy
]

# ----------------------------
# 3) Mục tiêu ví dụ
# ----------------------------
obj = cp.Minimize(t + 0.1 * x)
prob = cp.Problem(obj, constraints)

# Chọn solver MILP bạn có (ưu tiên: GUROBI/CPLEX/MOSEK; miễn phí: CBC, GLPK_MI, SCIP)
# prob.solve(solver=cp.GUROBI)
# prob.solve(solver=cp.SCIP)
prob.solve(solver=cp.CBC)          # hoặc cp.GLPK_MI nếu bạn cài

print("status:", prob.status)
print("x* =", x.value)
print("t* =", t.value)
print("true f(x*) =", f(float(x.value)))
print("approx gap t - f(x) =", float(t.value) - f(float(x.value)))
