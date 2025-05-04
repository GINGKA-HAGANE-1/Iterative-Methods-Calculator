import numpy as np

def euler_method(f, x0, y0, h, n):
    """Implement the Euler method for solving ODEs"""
    x = np.zeros(n+1)
    y = np.zeros(n+1)
    x[0] = x0
    y[0] = y0
    
    steps = []
    steps.append(f"Initial values: x₀ = {x0}, y₀ = {y0}")
    steps.append(f"Step size h = {h}")
    steps.append("\nCalculation steps:")
    
    for i in range(n):
        x[i+1] = x[i] + h
        slope = f(x[i], y[i])
        y[i+1] = y[i] + h * slope
        
        steps.append(f"Step {i+1}:")
        steps.append(f"  x_{i+1} = x_{i} + h = {x[i]:.5f} + {h:.5f} = {x[i+1]:.5f}")
        steps.append(f"  f(x_{i}, y_{i}) = f({x[i]:.5f}, {y[i]:.5f}) = {slope:.5f}")
        steps.append(f"  y_{i+1} = y_{i} + h * f(x_{i}, y_{i}) = {y[i]:.5f} + {h:.5f} * {slope:.5f} = {y[i+1]:.5f}")
    
    # Format results for return
    results = []
    for i in range(n+1):
        results.append({
            "step": i,
            "x": float(x[i]),
            "y": float(y[i]),
            "formatted": f"({x[i]:.5f}, {y[i]:.5f})"
        })
    
    return {
        "explanation": steps,
        "points": results
    }

def modified_euler_method(f, x0, y0, h, n):
    """Implement the Modified Euler method (Heun's method) for solving ODEs"""
    x = np.zeros(n+1)
    y = np.zeros(n+1)
    x[0] = x0
    y[0] = y0
    
    steps = []
    steps.append(f"Initial values: x₀ = {x0}, y₀ = {y0}")
    steps.append(f"Step size h = {h}")
    steps.append("\nCalculation steps:")
    
    for i in range(n):
        x[i+1] = x[i] + h
        k1 = f(x[i], y[i])
        y_pred = y[i] + h * k1
        k2 = f(x[i+1], y_pred)
        y[i+1] = y[i] + h * (k1 + k2) / 2
        
        steps.append(f"Step {i+1}:")
        steps.append(f"  x_{i+1} = x_{i} + h = {x[i]:.5f} + {h:.5f} = {x[i+1]:.5f}")
        steps.append(f"  k1 = f(x_{i}, y_{i}) = f({x[i]:.5f}, {y[i]:.5f}) = {k1:.5f}")
        steps.append(f"  y_pred = y_{i} + h * k1 = {y[i]:.5f} + {h:.5f} * {k1:.5f} = {y_pred:.5f}")
        steps.append(f"  k2 = f(x_{i+1}, y_pred) = f({x[i+1]:.5f}, {y_pred:.5f}) = {k2:.5f}")
        steps.append(f"  y_{i+1} = y_{i} + h * (k1 + k2)/2 = {y[i]:.5f} + {h:.5f} * ({k1:.5f} + {k2:.5f})/2 = {y[i+1]:.5f}")
    
    # Format results for return
    results = []
    for i in range(n+1):
        results.append({
            "step": i,
            "x": float(x[i]),
            "y": float(y[i]),
            "formatted": f"({x[i]:.5f}, {y[i]:.5f})"
        })
    
    return {
        "explanation": steps,
        "points": results
    }