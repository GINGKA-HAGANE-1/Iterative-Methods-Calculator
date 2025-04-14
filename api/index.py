from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

def jacobi_method(A, b, tolerance=1e-6, max_iterations=1000):
    n = len(A)
    x = np.zeros(n)
    x_new = np.zeros(n)
    iterations = []
    
    iterations.append({"iteration": 0, "values": np.round(x, 5).tolist(), "steps": []})
    
    for iteration in range(max_iterations):
        x_old = x.copy()
        steps = []
        
        for i in range(n):
            sum_ax = 0
            variable_names = ['x', 'y', 'z']
            step = f"{variable_names[i]} = (1/{A[i][i]}) * ({b[i]}"
            for j in range(n):
                if i != j:
                    sum_ax += A[i][j] * x_old[j]
                    step += f" + ({-A[i][j]})({np.round(x_old[j], 5)})"
            x_new[i] = (b[i] - sum_ax) / A[i][i]
            step += f") = {np.round(x_new[i], 5)}"
            steps.append(step)
        
        x = x_new.copy()
        iterations.append({
            "iteration": iteration + 1,
            "values": np.round(x, 5).tolist(),
            "steps": steps
        })
        
        if np.allclose(x, x_old, rtol=tolerance):
            return {"iterations": iterations, "converged": True, "final": np.round(x, 5).tolist()}
    
    return {"iterations": iterations, "converged": False, "final": np.round(x, 5).tolist()}

def gauss_seidel_method(A, b, tolerance=1e-6, max_iterations=1000):
    n = len(A)
    x = np.zeros(n)
    iterations = []
    
    iterations.append({"iteration": 0, "values": np.round(x, 5).tolist(), "steps": []})
    
    for iteration in range(max_iterations):
        x_old = x.copy()
        steps = []
        
        for i in range(n):
            sum_ax = 0
            variable_names = ['x', 'y', 'z']
            step = f"{variable_names[i]} = (1/{A[i][i]}) * ({b[i]}"
            for j in range(n):
                if i != j:
                    sum_ax += A[i][j] * x[j]
                    step += f" + ({-A[i][j]})({np.round(x[j], 5)})"
            x[i] = (b[i] - sum_ax) / A[i][i]
            step += f") = {np.round(x[i], 5)}"
            steps.append(step)
        
        iterations.append({
            "iteration": iteration + 1,
            "values": np.round(x, 5).tolist(),
            "steps": steps
        })
        
        if np.allclose(x, x_old, rtol=tolerance):
            return {"iterations": iterations, "converged": True, "final": np.round(x, 5).tolist()}
    
    return {"iterations": iterations, "converged": False, "final": np.round(x, 5).tolist()}

def power_method(A, tolerance=1e-6, max_iterations=1000):
    n = len(A)
    x_initial = np.array([1, 1, 1])  # Initial vector before normalization
    
    # Add initial multiplication AX₀
    Ax0 = np.dot(A, x_initial)
    x = x_initial / np.sqrt(np.sum(x_initial**2))  # Normalize to get x₁
    
    iterations = []
    # Add AX₀ calculation
    iterations.append({
        "iteration": 0,
        "matrix": A.tolist(),
        "vector": x_initial.tolist(),
        "result": np.round(Ax0, 5).tolist(),
        "eigenvalue": float(np.round(np.max(np.abs(Ax0)), 5)),
        "eigenvector": np.round(x, 5).tolist(),
        "next_vector_label": f"X1 = [{', '.join([f'{v:.5f}' for v in x])}]"
    })
    
    for iteration in range(max_iterations):
        Ax = np.dot(A, x)
        eigenvalue = np.max(np.abs(Ax))
        x_new = Ax / eigenvalue
        
        iterations.append({
            "iteration": iteration + 1,
            "matrix": A.tolist(),
            "vector": np.round(x, 5).tolist(),
            "result": np.round(Ax, 5).tolist(),
            "eigenvalue": float(np.round(eigenvalue, 5)),
            "eigenvector": np.round(x_new, 5).tolist(),
            "next_vector_label": f"X{iteration + 2} = [{', '.join([f'{v:.5f}' for v in x_new])}]"
        })
        
        if np.allclose(x, x_new, rtol=tolerance):
            break
        x = x_new
    
    return {"iterations": iterations}

# Add new route for power method
@app.route('/power-method', methods=['POST'])
def solve_power_method():
    data = request.get_json()
    A = np.array(data['matrix'])
    result = power_method(A)
    return jsonify(result)

@app.route('/')
def index():
    return render_template('index.html')

def is_diagonally_dominant(A):
    n = len(A)
    for i in range(n):
        diagonal = abs(A[i][i])
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diagonal <= row_sum:
            return False
    return True

@app.route('/solve', methods=['POST'])
def solve():
    data = request.get_json()
    A = np.array(data['matrix'])
    b = np.array(data['vector'])
    
    try:
        jacobi_result = jacobi_method(A, b)
        gauss_result = gauss_seidel_method(A, b)
        
        return jsonify({
            'jacobi': jacobi_result,
            'gauss_seidel': gauss_result
        })
    except Exception as e:
        return jsonify({
            'error': 'An error occurred while solving the system.',
            'details': str(e)
        })

def newton_forward_interpolation(x_values, y_values, x_target):
    n = len(x_values)
    h = float(x_values[1] - x_values[0])  # Convert to float
    
    # Create difference table with dynamic size
    F = [[0 for i in range(n)] for j in range(n)]
    
    # Fill first column with y values
    for i in range(n):
        F[i][0] = float(y_values[i])  # Convert to float
    
    # Calculate forward differences
    for j in range(1, n):
        for i in range(n-j):
            F[i][j] = float(F[i+1][j-1] - F[i][j-1])  # Convert to float
    
    # Calculate u = (x - x₀)/h
    u = float((x_target - x_values[0])/h)  # Convert to float
    
    # Calculate the interpolated value using Newton's forward formula
    y_target = float(F[0][0])  # Convert to float
    u_term = u
    fact = 1
    
    steps = []
    steps.append(f"h = {h}")
    steps.append(f"u = (x - x₀)/h = ({x_target} - {float(x_values[0])})/{h} = {u}")
    
    # Dynamic calculation based on number of points
    for j in range(1, n):
        term = float((u_term * F[0][j]) / fact)  # Convert to float
        steps.append(f"Term {j}: ({u_term} × {F[0][j]})/{fact} = {term}")
        y_target += term
        u_term *= (u - j)
        fact *= (j + 1)
    
    # Clean up the difference table for display
    cleaned_table = []
    for i in range(n):
        row = []
        for j in range(n-i):
            row.append(float(F[i][j]))  # Convert to float
        cleaned_table.append(row)
    
    return {
        "difference_table": cleaned_table,
        "steps": steps,
        "result": float(y_target)  # Convert to float
    }

def newton_backward_interpolation(x_values, y_values, x_target):
    n = len(x_values)
    h = float(x_values[1] - x_values[0])  # Convert to float
    
    # Calculate the backward difference table
    F = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        F[i][0] = float(y_values[i])  # Convert to float
    
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            F[i][j] = float(F[i][j-1] - F[i-1][j-1])  # Convert to float
    
    # Calculate u = (x - xₙ)/h
    u = float((x_target - x_values[n-1])/h)  # Convert to float
    
    # Calculate the interpolated value
    y_target = float(F[n-1][0])  # Convert to float
    u_term = float(u)  # Convert to float
    fact = 1
    
    steps = []
    steps.append(f"h = {h}")
    steps.append(f"u = (x - xₙ)/h = ({x_target} - {float(x_values[n-1])})/{h} = {u}")
    
    for j in range(1, n):
        term = float((u_term * F[n-1][j]) / fact)  # Convert to float
        steps.append(f"Term {j}: ({u_term} × {F[n-1][j]})/{fact} = {term}")
        y_target += term
        u_term *= (u + j)
        fact *= (j + 1)
    
    # Convert all values in difference table to float
    difference_table = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(float(F[i][j]))  # Convert to float
        difference_table.append(row)
    
    return {
        "difference_table": difference_table,
        "steps": steps,
        "result": float(y_target)  # Convert to float
    }

# Add this route after your existing routes
@app.route('/interpolate', methods=['POST'])
def interpolate():
    try:
        data = request.get_json()
        x_values = np.array(data['x_values'])
        y_values = np.array(data['y_values'])
        x_target = data['x_target']
        method = data['method']
        
        if method == 'forward':
            result = newton_forward_interpolation(x_values, y_values, x_target)
        else:
            result = newton_backward_interpolation(x_values, y_values, x_target)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

# At the bottom of app.py
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)