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

# At the bottom of app.py
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)