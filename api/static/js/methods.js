// Jacobi Method
function solveJacobi() {
    const matrix = [];
    let hasEmptyInputs = false;

    // Add validation and logging

    for (let i = 0; i < 3; i++) {
        matrix[i] = [];
        for (let j = 0; j < 3; j++) {
            const input = document.querySelector(`.jacobi-matrix-cell[data-row="${i}"][data-col="${j}"]`);
            const value = parseFloat(input.value);
            if (isNaN(value)) {
                input.style.borderColor = 'red';
                hasEmptyInputs = true;
            } else {
                input.style.borderColor = '';
                matrix[i][j] = value;
            }
        }
    }

    const vectorInputs = document.querySelectorAll('.jacobi-vector-cell');
    const vector = [];
    vectorInputs.forEach(input => {
        const value = parseFloat(input.value);
        if (isNaN(value)) {
            input.style.borderColor = 'red';
            hasEmptyInputs = true;
        } else {
            input.style.borderColor = '';
            vector.push(value);
        }
    });

    if (hasEmptyInputs) {
        alert('Please fill in all matrix and vector values with numbers');
        return;
    }

    fetch('/solve', {  // Changed from http://127.0.0.1:8080/solve
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            matrix: matrix,
            vector: vector
        }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (!data.jacobi) {
            throw new Error('Invalid response format from server');
        }
        
        const container = document.querySelector(`#jacobi-result .iterations`);
        
        displayResults('jacobi', data.jacobi);
    })
    .catch(error => {
        console.error('Error details:', error);
        alert('Error: ' + error.message);
    });
}

// Gauss-Seidel Method
// For the Gauss-Seidel method, update the fetch URL
function solveGaussSeidel() {
    const matrix = [];
    let hasEmptyInputs = false;

    // Add validation and logging

    for (let i = 0; i < 3; i++) {
        matrix[i] = [];
        for (let j = 0; j < 3; j++) {
            const input = document.querySelector(`.gauss-matrix-cell[data-row="${i}"][data-col="${j}"]`);
            const value = parseFloat(input.value);
            if (isNaN(value)) {
                input.style.borderColor = 'red';
                hasEmptyInputs = true;
            } else {
                input.style.borderColor = '';
                matrix[i][j] = value;
            }
        }
    }

    const vectorInputs = document.querySelectorAll('.gauss-vector-cell');
    const vector = [];
    vectorInputs.forEach(input => {
        const value = parseFloat(input.value);
        if (isNaN(value)) {
            input.style.borderColor = 'red';
            hasEmptyInputs = true;
        } else {
            input.style.borderColor = '';
            vector.push(value);
        }
    });

    if (hasEmptyInputs) {
        alert('Please fill in all matrix and vector values with numbers');
        return;
    }

    fetch('/solve', {  // Changed from http://127.0.0.1:8080/solve
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            matrix: matrix,
            vector: vector
        }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (!data.gauss_seidel) {
            throw new Error('Invalid response format from server');
        }
        displayResults('gauss-seidel', data.gauss_seidel);
    })
    .catch(error => {
        console.error('Error details:', error);
        alert('Error: ' + error.message);
    });
}

// Power Method
function solvePowerMethod() {
    const matrix = [];
    for (let i = 0; i < 3; i++) {
        matrix[i] = [];
        for (let j = 0; j < 3; j++) {
            matrix[i][j] = parseFloat(document.querySelector(`.power-matrix-cell[data-row="${i}"][data-col="${j}"]`).value);
        }
    }
    
    fetch('/power-method', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            matrix: matrix
        }),
    })
    .then(response => response.json())
    .then(data => {
        displayPowerResults(data);
    });
}

// Display Functions
function displayResults(method, results) {
    const container = document.querySelector(`#${method}-result .iterations`);
    container.innerHTML = '';
    
    const initialDiv = document.createElement('div');
    initialDiv.className = 'iteration';
    initialDiv.innerHTML = `
        <h3>Initial Guess (X₀)</h3>
        <div class="values">
            x = 0<br>
            y = 0<br>
            z = 0
        </div>
    `;
    container.appendChild(initialDiv);
    
    results.iterations.forEach(iteration => {
        if (iteration.steps && iteration.steps.length > 0) {
            const iterationDiv = document.createElement('div');
            iterationDiv.className = 'iteration-step';
            
            let stepsHtml = `<h4>Iteration ${iteration.iteration}</h4>`;
            iteration.steps.forEach(step => {
                stepsHtml += `<div class="step-calculation">${step}</div>`;
            });
            
            iterationDiv.innerHTML = stepsHtml;
            container.appendChild(iterationDiv);
        }
    });
    
    if (results.converged) {
        const finalDiv = document.createElement('div');
        finalDiv.className = 'final-result';
        finalDiv.innerHTML = `
            <h4>Final Solution:</h4>
            <div class="values">
                x = ${results.final[0].toFixed(5)}<br>
                y = ${results.final[1].toFixed(5)}<br>
                z = ${results.final[2].toFixed(5)}
            </div>
        `;
        container.appendChild(finalDiv);
    }
}

function displayPowerResults(results) {
    const container = document.querySelector('#power-method-result .iterations');
    container.innerHTML = '';
    
    results.iterations.forEach(iteration => {
        const iterationDiv = document.createElement('div');
        iterationDiv.className = 'iteration-step';
        
        iterationDiv.innerHTML = `
            <h3>AX${iteration.iteration} =</h3>
            <div class="matrix-calculation">
                [${iteration.matrix[0].join(' ')}] × [${iteration.vector[0].toFixed(5)}]   [${iteration.result[0].toFixed(5)}]
                [${iteration.matrix[1].join(' ')}] × [${iteration.vector[1].toFixed(5)}] = [${iteration.result[1].toFixed(5)}] = ${iteration.eigenvalue.toFixed(5)} X${iteration.iteration + 1}
                [${iteration.matrix[2].join(' ')}] × [${iteration.vector[2].toFixed(5)}]   [${iteration.result[2].toFixed(5)}]
            </div>
            <div class="next-vector">
                ${iteration.next_vector_label}
            </div>
        `;
        
        container.appendChild(iterationDiv);
    });
}
