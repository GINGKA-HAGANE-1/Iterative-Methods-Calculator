from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# ... (keep all your existing functions) ...

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    data = request.get_json()
    A = np.array(data['matrix'])
    b = np.array(data['vector'])
    
    jacobi_result = jacobi_method(A, b)
    gauss_result = gauss_seidel_method(A, b)
    
    return jsonify({
        'jacobi': jacobi_result,
        'gauss_seidel': gauss_result
    })