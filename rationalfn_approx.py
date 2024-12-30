import numpy as np
import cvxpy as cp
x_data = np.array([
0, 0.2, 0.5, 0.66666666, 1
], dtype=float)
y_data = np.array([
0.333333, 0.4545454545, 0.6, 0.6666666666, 0.7083333
], dtype=float)

n = len(x_data)

A = np.vstack([
    np.ones(n),
    -x_data * y_data,
    -y_data
]).T

b_vector = -x_data

v = cp.Variable(3)

objective = cp.Minimize(cp.sum_squares(A @ v - b_vector))

problem = cp.Problem(objective)
problem.solve()

b_opt, c_opt, d_opt = v.value

print(f"Optimal coefficients:")
print(f"b = {b_opt}")
print(f"c = {c_opt}")
print(f"d = {d_opt}")

def rational_function(x):
    return (x + b_opt) / (c_opt*x+d_opt)


x_test = 6
y_pred = rational_function(x_test)
print(f"For x = {x_test}, predicted y = {y_pred}")
