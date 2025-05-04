from flask import Flask, render_template, request, jsonify
import numpy as np
import math

from euler import euler_method, modified_euler_method
from flask_cors import CORS, cross_origin  # Add this import
try:
    from flask import Flask, request, jsonify, render_template
    import numpy as np
    import traceback
    import sys
except Exception as e:
    print(f"Import Error: {str(e)}")
    raise e

app = Flask(__name__)
CORS(app)

@app.errorhandler(Exception)
def handle_error(error):
    print(f"Error: {str(error)}")
    print(f"Traceback: {traceback.format_exc()}")
    return jsonify({
        "error": str(error),
        "traceback": traceback.format_exc()
    }), 500

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
@cross_origin()
def solve():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No data provided',
                'details': 'Request body is empty'
            }), 400

        if 'matrix' not in data or 'vector' not in data:
            return jsonify({
                'error': 'Missing required fields',
                'details': 'Both matrix and vector are required'
            }), 400

        A = np.array(data['matrix'])
        b = np.array(data['vector'])

        # Validate matrix dimensions
        if A.shape != (3, 3):
            return jsonify({
                'error': 'Invalid matrix dimensions',
                'details': 'Matrix must be 3x3'
            }), 400

        # Validate vector dimensions
        if b.shape != (3,):
            return jsonify({
                'error': 'Invalid vector dimensions',
                'details': 'Vector must have 3 elements'
            }), 400

        # Check for diagonal dominance
        if not is_diagonally_dominant(A):
            return jsonify({
                'error': 'Matrix is not diagonally dominant',
                'details': 'The matrix must be diagonally dominant for the methods to converge'
            }), 400

        # Check for zero diagonal elements
        if any(A[i][i] == 0 for i in range(3)):
            return jsonify({
                'error': 'Invalid matrix',
                'details': 'Diagonal elements cannot be zero'
            }), 400

        jacobi_result = jacobi_method(A, b)
        gauss_result = gauss_seidel_method(A, b)

        return jsonify({
            'jacobi': jacobi_result,
            'gauss_seidel': gauss_result
        })

    except ValueError as ve:
        return jsonify({
            'error': 'Invalid input data',
            'details': str(ve)
        }), 400
    except Exception as e:
        print('Error in /solve:', str(e))  # Server-side logging
        return jsonify({
            'error': 'An error occurred while solving the system',
            'details': str(e)
        }), 500

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
    steps.append(f"Newton Forward Interpolation")
    steps.append(f"Target x = {x_target}")
    steps.append(f"\nCalculation Steps:")
    steps.append(f"h = {h}")
    steps.append(f"u = (x - x₀)/h = ({x_target} - {float(x_values[0])})/{h} = {u}")
    
    # Prepare the difference table for display in the same format as backward interpolation
    cleaned_table = []
    if n >= 4:  # Assuming at least 4 points for consistency with backward method
        cleaned_table.append([F[0][0], F[0][1], F[0][2], F[0][3]])  # First row
        cleaned_table.append([F[1][0], F[1][1], F[1][2]])  # Second row
        cleaned_table.append([F[2][0], F[2][1]])  # Third row
        cleaned_table.append([F[3][0]])  # Fourth row
    else:
        # Handle cases with fewer points
        for i in range(n):
            row = []
            for j in range(n-i):
                row.append(float(F[i][j]))
            cleaned_table.append(row)
    
    # Add formula explanation
    steps.append(f"\nNewton's Forward Interpolation Formula:")
    steps.append(f"f(x) = f(x₀) + u∆f(x₀) + (u(u-1)/2!)∆²f(x₀) + (u(u-1)(u-2)/3!)∆³f(x₀) + ...")
    
    # Dynamic calculation based on number of points with more detailed steps
    steps.append(f"\nTerm-by-term calculation:")
    steps.append(f"First term: f(x₀) = {F[0][0]}")
    
    # Add running sum to track the result
    running_sum = float(F[0][0])
    steps.append(f"Sum after term 0: {running_sum}")
    
    for j in range(1, n):
        term = float((u_term * F[0][j]) / fact)  # Convert to float
        
        # Create a more detailed step explanation
        if j == 1:
            steps.append(f"Second term: u∆f(x₀) = {u} × {F[0][j]} = {u * F[0][j]}")
        elif j == 2:
            steps.append(f"Third term: (u(u-1)/2!)∆²f(x₀) = ({u}×{u-1}/2) × {F[0][j]} = {(u*(u-1)/2)} × {F[0][j]} = {term}")
        elif j == 3:
            steps.append(f"Fourth term: (u(u-1)(u-2)/3!)∆³f(x₀) = ({u}×{u-1}×{u-2}/6) × {F[0][j]} = {(u*(u-1)*(u-2)/6)} × {F[0][j]} = {term}")
        else:
            steps.append(f"Term {j+1}: ({u_term}/{fact}) × {F[0][j]} = {term}")
        
        running_sum += term
        steps.append(f"Sum after term {j}: {running_sum}")
        
        # Update for next iteration
        u_term *= (u - j)
        fact *= (j + 1)
    
    # Final result with more explanation
    steps.append(f"\nFinal Result:")
    steps.append(f"f({x_target}) = {float(y_target)}")
    
    return {
        "difference_table": cleaned_table,
        "steps": steps,
        "result": float(y_target)  # Convert to float
    }

def newton_backward_interpolation(x_values, y_values, x_target):
    n = len(x_values)
    h = float(x_values[1] - x_values[0])
    
    # Calculate the backward difference table
    F = [[0 for i in range(n)] for j in range(n)]
    
    # Fill first column with y values
    for i in range(n):
        F[i][0] = float(y_values[i])
    
    # Calculate backward differences correctly
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            F[i][j] = float(F[i][j-1] - F[i-1][j-1])
    
    # Prepare the difference table for display
    cleaned_table = []
    cleaned_table.append([F[0][0], F[1][1], F[2][2], F[3][3]])  # First row
    cleaned_table.append([F[1][0], F[2][1], F[3][2]])  # Second row
    cleaned_table.append([F[2][0], F[3][1]])  # Third row
    cleaned_table.append([F[3][0]])  # Fourth row
    
    # Rest of the calculation remains the same
    u = float((x_target - x_values[n-1])/h)
    y_target = float(F[n-1][0])
    u_term = float(u)
    fact = 1
    
    steps = []
    steps.append(f"h = {h}")
    steps.append(f"u = (x - xₙ)/h = ({x_target} - {float(x_values[n-1])})/{h} = {u}")
    
    for j in range(1, n):
        term = float((u_term * F[n-1][j]) / fact)
        steps.append(f"Term {j}: ({u_term} × {F[n-1][j]})/{fact} = {term}")
        y_target += term
        u_term *= (u + j)
        fact *= (j + 1)
    
    return {
        "difference_table": cleaned_table,
        "steps": steps,
        "result": float(y_target)
    }

# Add after your other interpolation methods
def newton_divided_difference(x_values, y_values, x_target):
    n = len(x_values)
    F = [[0 for i in range(n)] for j in range(n)]
    
    # Fill first column with y values
    for i in range(n):
        F[i][0] = float(y_values[i])
    
    # Calculate divided differences
    for j in range(1, n):
        for i in range(n-j):
            F[i][j] = float((F[i+1][j-1] - F[i][j-1]) / (x_values[i+j] - x_values[i]))
    
    # Prepare the divided difference table for display
    cleaned_table = []
    for i in range(n):
        row = []
        for j in range(n-i):
            row.append(float(F[i][j]))
        cleaned_table.append(row)
    
    # Calculate interpolation value with detailed steps
    y_target = float(F[0][0])
    term = 1.0
    steps = []
    steps.append(f"f(x) = {F[0][0]}")
    
    # Build the formula step by step
    formula = f"f(x) = {F[0][0]}"
    for j in range(1, n):
        term *= (x_target - x_values[j-1])
        contribution = term * F[0][j]
        y_target += contribution
        
        # Add term to formula
        factor_terms = [f"(x - {x_values[k]})" for k in range(j)]
        formula_term = f" + ({' × '.join(factor_terms)}) × {F[0][j]}"
        formula += formula_term
        steps.append(f"Formula after term {j}: {formula}")
        
        # Show numerical calculation
        steps.append(f"Term {j} calculation: ({' × '.join([f'({x_target} - {x_values[k]})' for k in range(j)])})"
                    f" × {F[0][j]} = {contribution}")
        steps.append(f"Sum after term {j}: {y_target}")
    
    return {
        "difference_table": cleaned_table,
        "steps": steps,
        "result": float(y_target)
    }

# Add before the interpolate route
def lagrange_interpolation(x_values, y_values, x_target):
    n = len(x_values)
    y_target = 0
    steps = []
    
    # Calculate each Lagrange term
    for i in range(n):
        # Calculate the Lagrange basis polynomial
        numerator = []
        denominator = []
        
        for j in range(n):
            if i != j:
                numerator.append(f"(x - {x_values[j]})")
                denominator.append(f"({x_values[i]} - {x_values[j]})")
        
        # Calculate the numerical values
        term = y_values[i]
        for j in range(n):
            if i != j:
                term *= (x_target - x_values[j]) / (x_values[i] - x_values[j])
        
        # Format the step for display
        step = f"L{i}(x) = {y_values[i]} × "
        step += f"({' × '.join(numerator)}) / ({' × '.join(denominator)})"
        steps.append(step)
        
        # Show numerical calculation
        num_calc = f"L{i}({x_target}) = {y_values[i]} × "
        num_terms = [f"({x_target} - {x_values[j]})/({x_values[i]} - {x_values[j]})" 
                    for j in range(n) if i != j]
        num_calc += f"{' × '.join(num_terms)} = {term:.4f}"
        steps.append(num_calc)
        
        y_target += term
        steps.append(f"Sum after term {i+1}: {y_target:.4f}")
    
    return {
        "steps": steps,
        "result": float(y_target)
    }

# Modify the interpolate route to include the divided difference method
@app.route('/interpolate', methods=['POST'])
def interpolate():
    try:
        data = request.get_json()
        # Add validation for x_values and y_values
        if 'x_values' not in data or 'y_values' not in data:
            return jsonify({
                'error': 'Missing x_values or y_values'
            }), 400
            
        if len(data['x_values']) == 0 or len(data['y_values']) == 0:
            return jsonify({
                'error': 'Empty x_values or y_values'
            }), 400

        x_values = np.array([float(x) for x in data['x_values']])
        y_values = np.array([float(y) for y in data['y_values']])
        x_target = float(data['x_target'])
        method = data['method']
        
        if len(x_values) != len(y_values):
            return jsonify({
                'error': 'Number of x values must match number of y values'
            }), 400

        if method == 'forward':
            result = newton_forward_interpolation(x_values, y_values, x_target)
        elif method == 'backward':
            result = newton_backward_interpolation(x_values, y_values, x_target)
        elif method == 'divided':
            result = newton_divided_difference(x_values, y_values, x_target)
        else:
            result = lagrange_interpolation(x_values, y_values, x_target)
        
        # Add x_target to the response
        result['x_target'] = x_target
        result['points'] = list(zip(x_values.tolist(), y_values.tolist()))
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

# Add after the interpolation methods
# Add these imports at the top if not already present
import numpy as np
from scipy import integrate
import sympy as sp

# Add new route for Euler method
@app.route('/euler', methods=['POST'])
@cross_origin()
def solve_euler():
    try:
        data = request.get_json()
        
        # Extract parameters
        function_str = data.get('function')
        x0 = float(data.get('x0'))
        y0 = float(data.get('y0'))
        h = float(data.get('h'))
        n = int(data.get('n'))
        method = data.get('method')
        
        # Validate inputs
        if not function_str or not isinstance(x0, (int, float)) or not isinstance(y0, (int, float)) or \
           not isinstance(h, (int, float)) or not isinstance(n, int):
            return jsonify({"error": "Invalid input parameters"}), 400
        
        # Parse the function string
        x, y = sp.symbols('x y')
        try:
            f = sp.sympify(function_str)
            f_lambda = sp.lambdify((x, y), f, 'numpy')
        except Exception as e:
            return jsonify({"error": f"Error parsing function: {str(e)}"}), 400
        
        # Choose the appropriate method
        if method == 'euler':
            steps = euler_method(f_lambda, x0, y0, h, n)
        elif method == 'modified':
            steps = modified_euler_method(f_lambda, x0, y0, h, n)
        else:
            return jsonify({"error": "Invalid method specified"}), 400
        
        return jsonify({"steps": steps})
    
    except Exception as e:
        print(f"Error in Euler method: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Add new route for numerical integration
@app.route('/integrate', methods=['POST'])
@cross_origin()
def numerical_integration():
    try:
        data = request.get_json()
        
        # Extract parameters
        lower_limit = data.get('lower_limit')
        upper_limit = data.get('upper_limit')
        intervals = int(data.get('intervals'))
        function_str = data.get('function')
        method = data.get('method')
        
        # Convert string limits to float (handle pi and other math constants)
        try:
            # Handle mathematical expressions in limits
            if any(x in lower_limit.lower() for x in ['pi', 'π', 'sin', 'cos', 'tan', 'log', 'exp', 'sqrt']):
                # Replace π with pi for consistency
                lower_limit = lower_limit.lower().replace('π', 'np.pi')
                # Make sure numpy functions are properly prefixed
                for func in ['sin', 'cos', 'tan', 'log', 'exp', 'sqrt']:
                    if func in lower_limit.lower() and f'np.{func}' not in lower_limit.lower():
                        lower_limit = lower_limit.lower().replace(func, f'np.{func}')
                # Evaluate the expression
                lower = eval(lower_limit)
            else:
                lower = float(lower_limit)
                
            if any(x in upper_limit.lower() for x in ['pi', 'π', 'sin', 'cos', 'tan', 'log', 'exp', 'sqrt']):
                upper_limit = upper_limit.lower().replace('π', 'np.pi')
                for func in ['sin', 'cos', 'tan', 'log', 'exp', 'sqrt']:
                    if func in upper_limit.lower() and f'np.{func}' not in upper_limit.lower():
                        upper_limit = upper_limit.lower().replace(func, f'np.{func}')
                upper = eval(upper_limit)
            else:
                upper = float(upper_limit)
        except Exception as e:
            return jsonify({"error": f"Error parsing limits: {str(e)}"}), 400
        
        # In the numerical_integration function
        # Parse the function
        x = sp.symbols('x')
        try:
            # Check for multiple variables
            function_str = function_str.replace('np.', '')
            if 'y' in function_str or 'z' in function_str:
                return jsonify({"error": "Integration only supports functions of one variable (x). Please remove other variables."}), 400
                
            f = sp.sympify(function_str)
            f_lambda = sp.lambdify(x, f, modules=['numpy'])
        except Exception as e:
            return jsonify({"error": f"Error parsing function: {str(e)}"}), 400
        
        # Calculate using different methods
        if method == 'trapezoidal':
            result, steps = trapezoidal_rule(f_lambda, lower, upper, intervals)
        elif method == 'simpson':
            result, steps = simpson_rule(f_lambda, lower, upper, intervals)
        else:
            # Use scipy for a reference value
            result, _ = integrate.quad(f_lambda, lower, upper)
            steps = [f"Using scipy.integrate.quad: {result:.6f}"]
        
        return jsonify({
            "result": float(result),
            "steps": steps
        })
    
    except Exception as e:
        print(f"Error in numerical integration: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Helper functions for numerical integration
def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    
    steps = []
    steps.append(f"Trapezoidal Rule")
    steps.append(f"Interval: [{a}, {b}]")
    steps.append(f"Number of subintervals: {n}")
    steps.append(f"\nCalculation Steps:")
    steps.append(f"h = (b - a)/n = ({b} - {a})/{n} = {h:.14f}")
    steps.append(f"Dividing interval [{a}, {b}] into {n} equal parts")
    
    # Points calculation
    steps.append(f"Points calculation:")
    for i in range(n+1):
        steps.append(f"x{i} = {x[i]:.5f}, f(x{i}) = {y[i]:.5f}")
    
    # Individual trapezoids
    steps.append(f"Calculating areas of individual trapezoids:")
    trapezoid_areas = []
    for i in range(n):
        area = (h/2) * (y[i] + y[i+1])
        trapezoid_areas.append(area)
        steps.append(f"Trapezoid {i+1}: (h/2)(f(x{i}) + f(x{i+1})) = ({h}/2)({y[i]:.5f} + {y[i+1]:.5f}) = {area:.5f}")
    
    # Formula and detailed calculation
    steps.append(f"Trapezoidal Rule Formula:")
    steps.append(f"∫f(x)dx ≈ (h/2)[f(x₀) + 2(f(x₁) + ... + f(xₙ₋₁)) + f(xₙ)]")
    
    middle_terms = ' + '.join([f"{y[i]:.5f}" for i in range(1, n)])
    steps.append(f"Detailed Calculation:")
    steps.append(f"= ({h:.5f}/2)[{y[0]:.5f} + 2({middle_terms}) + {y[n]:.5f}]")
    
    # Calculate result
    result = h * (0.5 * y[0] + np.sum(y[1:n]) + 0.5 * y[n])
    steps.append(f"= {result:.5f}")
    
    # Final result
    steps.append(f"\nResult:")
    steps.append(f"∫{a}{b} f(x) dx = {result:.5f}")
    
    return result, steps

def simpson_rule(f, a, b, n):
    if n % 2 != 0:
        n += 1  # Ensure n is even
    
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    
    steps = []
    steps.append(f"Simpson's 1/3 Rule")
    steps.append(f"Interval: [{a}, {b}]")
    steps.append(f"Number of subintervals: {n}")
    steps.append(f"\nCalculation Steps:")
    steps.append(f"h = (b - a)/n = ({b} - {a})/{n} = {h:.5f}")
    steps.append(f"Dividing interval [{a}, {b}] into {n} equal parts")
    
    # Points calculation
    steps.append(f"Points calculation:")
    for i in range(n+1):
        steps.append(f"x{i} = {x[i]:.5f}, f(x{i}) = {y[i]:.5f}")
    
    # Group terms for Simpson's rule
    steps.append(f"Grouping terms:")
    steps.append(f"First term (f₀): {y[0]:.5f}")
    steps.append(f"Last term (fₙ): {y[n]:.5f}")
    
    # Odd-indexed terms
    odd_terms = y[1:n:2]
    odd_terms_str = ", ".join([f"{val:.5f}" for val in odd_terms])
    steps.append(f"Odd-indexed terms (4×): {odd_terms_str}")
    
    # Even-indexed terms
    even_terms = y[2:n:2]
    even_terms_str = ", ".join([f"{val:.5f}" for val in even_terms])
    steps.append(f"Even-indexed terms (2×): {even_terms_str}")
    
    # Formula and detailed calculation
    steps.append(f"Simpson's 1/3 Rule Formula:")
    steps.append(f"∫f(x)dx ≈ (h/3)[f₀ + 4(f₁ + f₃ + ...) + 2(f₂ + f₄ + ...) + fₙ]")
    
    # Calculate the sum of odd-indexed terms
    odd_sum = np.sum(odd_terms)
    # Calculate the sum of even-indexed terms
    even_sum = np.sum(even_terms)
    
    steps.append(f"Detailed Calculation:")
    steps.append(f"= ({h:.5f}/3)[{y[0]:.5f} + 4({odd_terms_str}) + 2({even_terms_str}) + {y[n]:.5f}]")
    steps.append(f"= ({h:.5f}/3)[{y[0]:.5f} + 4({odd_sum:.5f}) + 2({even_sum:.5f}) + {y[n]:.5f}]")
    
    # Final calculation
    total = y[0] + 4*odd_sum + 2*even_sum + y[n]
    steps.append(f"= ({h:.5f}/3)[{total:.5f}]")
    
    # Calculate result
    result = h/3 * (y[0] + 4*np.sum(y[1:n:2]) + 2*np.sum(y[2:n:2]) + y[n])
    steps.append(f"= {result:.5f}")
    
    # Final result
    steps.append(f"\nResult:")
    steps.append(f"∫{a}{b} f(x) dx = {result:.5f}")
    
    return result, steps

# Add new route for numerical differentiation
@app.route('/differentiate', methods=['POST'])
def differentiate():
    try:
        data = request.get_json()
        
        if 'x_values' not in data or 'y_values' not in data:
            return jsonify({
                'error': 'Missing x_values or y_values'
            }), 400
            
        x_values = np.array([float(x) for x in data['x_values']])
        y_values = np.array([float(y) for y in data['y_values']])
        x_target = float(data['x_target'])
        method = data['method']
        
        if len(x_values) != len(y_values):
            return jsonify({
                'error': 'Number of x values must match number of y values'
            }), 400
            
        if method == 'forward':
            result = newton_forward_differentiation(x_values, y_values, x_target)
        else:
            result = newton_backward_differentiation(x_values, y_values, x_target)
            
        result['points'] = list(zip(x_values.tolist(), y_values.tolist()))
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

def newton_forward_differentiation(x_values, y_values, x_target):
    h = x_values[1] - x_values[0]  # Step size
    steps = []
    
    # Calculate forward differences
    differences = [y_values]
    for i in range(1, len(y_values)):
        diff = np.diff(differences[-1])
        differences.append(diff)
    
    steps.append(f"Step size h = {h}")
    steps.append("\nForward Differences Table:")
    
    # Show differences table
    for i, diff in enumerate(differences):
        steps.append(f"Δ^{i}y: {', '.join([f'{v:.5f}' for v in diff])}")
    
    # Calculate first derivative using the correct formula from textbook
    # (dy/dx) = (1/h)(Δy₀ - Δ²y₀/2 + Δ³y₀/3)
    first_deriv = (1/h) * (differences[1][0] - differences[2][0]/2 + differences[3][0]/3)
    second_deriv = differences[2][0] / (h * h)
    third_deriv = differences[3][0] / (h * h * h)
    
    # Show detailed calculations
    steps.append("\nDerivative Calculations:")
    steps.append(f"First Derivative: (dy/dx) = (1/h)(Δy₀ - Δ²y₀/2 + Δ³y₀/3)")
    steps.append(f"= (1/{h})({differences[1][0]} - {differences[2][0]}/2 + {differences[3][0]}/3)")
    steps.append(f"= (1/{h})({differences[1][0]} - {differences[2][0]/2} + {differences[3][0]/3})")
    steps.append(f"= {first_deriv:.5f}")
    
    steps.append(f"\nSecond Derivative: f''(x₀) = Δ²y₀/h² = {differences[2][0]:.5f}/{h*h} = {second_deriv:.5f}")
    steps.append(f"Third Derivative: f'''(x₀) = Δ³y₀/h³ = {differences[3][0]:.5f}/{h*h*h} = {third_deriv:.5f}")
    
    # Calculate f(x) using Taylor series
    x_diff = x_target - x_values[0]
    fx_value = y_values[0]  # f(x₀)
    
    # Add Taylor series terms
    fx_value += first_deriv * x_diff
    fx_value += (second_deriv * x_diff * x_diff) / 2
    fx_value += (third_deriv * x_diff * x_diff * x_diff) / 6
    
    steps.append("\nFunction Value Calculation:")
    steps.append(f"f(x) = f(x₀) + f'(x₀)(x-x₀) + (f''(x₀)/2!)(x-x₀)² + (f'''(x₀)/3!)(x-x₀)³")
    steps.append(f"f({x_target}) = {y_values[0]:.5f} + {first_deriv:.5f}({x_diff:.5f}) + "
                f"({second_deriv:.5f}/2)({x_diff:.5f})² + ({third_deriv:.5f}/6)({x_diff:.5f})³")
    steps.append(f"f({x_target}) = {fx_value:.5f}")
    
    return {
        "steps": steps,
        "derivatives": {
            "derivative_1": round(float(first_deriv), 5),
            "derivative_2": round(float(second_deriv), 5),
            "derivative_3": round(float(third_deriv), 5)
        },
        "fx_value": round(float(fx_value), 5)
    }

def newton_backward_differentiation(x_values, y_values, x_target):
    h = x_values[1] - x_values[0]  # Step size
    steps = []
    
    # Calculate backward differences
    differences = [y_values]
    for i in range(1, len(y_values)):
        diff = np.diff(differences[-1])
        differences.append(diff)
    
    steps.append(f"Step size h = {h}")
    steps.append("\nBackward Differences Table:")
    
    # Show differences table
    for i, diff in enumerate(differences):
        steps.append(f"∇^{i}y: {', '.join([f'{v:.5f}' for v in diff])}")
    
    # Calculate first derivative using the backward difference formula
    # (dy/dx) = (1/h)(∇yₙ - ∇²yₙ/2 + ∇³yₙ/3)
    first_deriv = (1/h) * (differences[1][-1] - differences[2][-1]/2 + differences[3][-1]/3)
    second_deriv = differences[2][-1] / (h * h)
    third_deriv = differences[3][-1] / (h * h * h)
    
    # Show detailed calculations
    steps.append("\nDerivative Calculations:")
    steps.append(f"First Derivative: (dy/dx) = (1/h)(∇yₙ - ∇²yₙ/2 + ∇³yₙ/3)")
    steps.append(f"= (1/{h})({differences[1][-1]} - {differences[2][-1]}/2 + {differences[3][-1]}/3)")
    steps.append(f"= (1/{h})({differences[1][-1]} - {differences[2][-1]/2} + {differences[3][-1]/3})")
    steps.append(f"= {first_deriv:.5f}")
    
    steps.append(f"\nSecond Derivative: f''(xₙ) = ∇²yₙ/h² = {differences[2][-1]:.5f}/{h*h} = {second_deriv:.5f}")
    steps.append(f"Third Derivative: f'''(xₙ) = ∇³yₙ/h³ = {differences[3][-1]:.5f}/{h*h*h} = {third_deriv:.5f}")
    
    # Calculate f(x) using Taylor series
    x_diff = x_target - x_values[-1]
    fx_value = y_values[-1]  # f(xₙ)
    
    # Add Taylor series terms
    fx_value += first_deriv * x_diff
    fx_value += (second_deriv * x_diff * x_diff) / 2
    fx_value += (third_deriv * x_diff * x_diff * x_diff) / 6
    
    steps.append("\nFunction Value Calculation:")
    steps.append(f"f(x) = f(xₙ) + f'(xₙ)(x-xₙ) + (f''(xₙ)/2!)(x-xₙ)² + (f'''(xₙ)/3!)(x-xₙ)³")
    steps.append(f"f({x_target}) = {y_values[-1]:.5f} + {first_deriv:.5f}({x_diff:.5f}) + "
                f"({second_deriv:.5f}/2)({x_diff:.5f})² + ({third_deriv:.5f}/6)({x_diff:.5f})³")
    steps.append(f"f({x_target}) = {fx_value:.5f}")
    
    return {
        "steps": steps,
        "derivatives": {
            "derivative_1": round(float(first_deriv), 5),
            "derivative_2": round(float(second_deriv), 5),
            "derivative_3": round(float(third_deriv), 5)
        },
        "fx_value": round(float(fx_value), 5)
    }

# Add new route for numerical integration


# Add this at the end of your file, before if __name__ == "__main__":
@app.errorhandler(Exception)
def handle_exception(e):
    print(f"Error: {str(e)}")
    import traceback
    print(traceback.format_exc())
    return jsonify({
        'error': 'An unexpected error occurred',
        'details': str(e)
    }), 500

if __name__ == '__main__':
    app.run(port=8080)