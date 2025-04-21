// Common function to update points input fields
function updatePoints(method) {
    const prefixes = {
        'nf': 'Newton Forward',
        'nb': 'Newton Backward',
        'nd': 'Newton Divided',
        'l': 'Lagrange'
    };
    
    const n = parseInt(document.getElementById(`${method}-points`).value);
    const container = document.getElementById(`${method}-points-container`);
    container.innerHTML = '';
    
    for (let i = 0; i < n; i++) {
        const pointDiv = document.createElement('div');
        pointDiv.className = 'point-input';
        pointDiv.innerHTML = `
            <div class="input-group">
                <label>x${i}:</label>
                <input type="text" inputmode="decimal" pattern="[0-9]*[.]?[0-9]*|-[0-9]*[.]?[0-9]*" class="${method}-x form-control" data-index="${i}">
            </div>
            <div class="input-group">
                <label>y${i}:</label>
                <input type="text" inputmode="decimal" pattern="[0-9]*[.]?[0-9]*|-[0-9]*[.]?[0-9]*" class="${method}-y form-control" data-index="${i}">
            </div>
        `;
        container.appendChild(pointDiv);
    }
}

// Initialize points inputs for all methods
document.addEventListener('DOMContentLoaded', () => {
    const methods = ['nf', 'nb', 'nd', 'l'];
    methods.forEach(method => {
        const pointsInput = document.getElementById(`${method}-points`);
        if (pointsInput) {
            pointsInput.addEventListener('change', () => updatePoints(method));
            updatePoints(method);
        }
    });
});

// Calculation functions for each method
function calculateNewtonForward() {
    const x_values = Array.from(document.querySelectorAll('.nf-x')).map(input => parseFloat(input.value)).filter(val => !isNaN(val));
    const y_values = Array.from(document.querySelectorAll('.nf-y')).map(input => parseFloat(input.value)).filter(val => !isNaN(val));
    const x_target = parseFloat(document.getElementById('nf-x-value').value);

    if (!x_values.length || !y_values.length || isNaN(x_target)) {
        document.getElementById('nf-result').innerHTML = '<div class="error">Please fill in all values correctly</div>';
        return;
    }

    fetch('/interpolate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            x_values: x_values,
            y_values: y_values,
            x_target: x_target,
            method: 'forward'
        })
    })
    .then(response => response.json())
    .then(data => displayInterpolationResults('nf', data))
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('nf-result').innerHTML = '<div class="error">An error occurred during calculation.</div>';
    });
}

function calculateNewtonBackward() {
    const x_values = Array.from(document.querySelectorAll('.nb-x')).map(input => parseFloat(input.value)).filter(val => !isNaN(val));
    const y_values = Array.from(document.querySelectorAll('.nb-y')).map(input => parseFloat(input.value)).filter(val => !isNaN(val));
    const x_target = parseFloat(document.getElementById('nb-x-value').value);

    if (!x_values.length || !y_values.length || isNaN(x_target)) {
        document.getElementById('nb-result').innerHTML = '<div class="error">Please fill in all values correctly</div>';
        return;
    }

    fetch('/interpolate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            x_values: x_values,
            y_values: y_values,
            x_target: x_target,
            method: 'newton_backward'
        })
    })
    .then(response => response.json())
    .then(data => displayInterpolationResults('nb', data))
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('nb-result').innerHTML = '<div class="error">An error occurred during calculation.</div>';
    });
}

function calculateNewtonDivided() {
    const x_values = Array.from(document.querySelectorAll('.nd-x')).map(input => parseFloat(input.value)).filter(val => !isNaN(val));
    const y_values = Array.from(document.querySelectorAll('.nd-y')).map(input => parseFloat(input.value)).filter(val => !isNaN(val));
    const x_target = parseFloat(document.getElementById('nd-x-value').value);

    if (!x_values.length || !y_values.length || isNaN(x_target)) {
        document.getElementById('nd-result').innerHTML = '<div class="error">Please fill in all values correctly</div>';
        return;
    }

    fetch('/interpolate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            x_values: x_values,
            y_values: y_values,
            x_target: x_target,
            method: 'newton_divided'
        })
    })
    .then(response => response.json())
    .then(data => displayInterpolationResults('nd', data))
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('nd-result').innerHTML = '<div class="error">An error occurred during calculation.</div>';
    });
}

function calculateLagrange() {
    const x_values = Array.from(document.querySelectorAll('.l-x')).map(input => parseFloat(input.value)).filter(val => !isNaN(val));
    const y_values = Array.from(document.querySelectorAll('.l-y')).map(input => parseFloat(input.value)).filter(val => !isNaN(val));
    const x_target = parseFloat(document.getElementById('l-x-value').value);

    if (!x_values.length || !y_values.length || isNaN(x_target)) {
        document.getElementById('l-result').innerHTML = '<div class="error">Please fill in all values correctly</div>';
        return;
    }

    fetch('/interpolate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            x_values: x_values,
            y_values: y_values,
            x_target: x_target,
            method: 'lagrange'
        })
    })
    .then(response => response.json())
    .then(data => displayInterpolationResults('l', data))
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('l-result').innerHTML = '<div class="error">An error occurred during calculation.</div>';
    });
}

// Display results for all interpolation methods
function displayInterpolationResults(prefix, data) {
    if (data.error) {
        document.getElementById(`${prefix}-result`).innerHTML = `<div class="error">${data.error}</div>`;
        return;
    }

    let html = '<div class="calculation-steps">';
    
    // Display method title and input values
    const methodNames = {
        'nf': 'Newton Forward',
        'nb': 'Newton Backward',
        'nd': 'Newton Divided',
        'l': 'Lagrange'
    };

    html += `
        <div class="result-section">
            <h3 class="section-title">${methodNames[prefix]}</h3>
            <div class="method-explanation">
                <p>Interpolating at x = ${data.x_target}</p>
                <p>Using points: ${data.points ? data.points.map(([x, y]) => `(${x}, ${y})`).join(', ') : ''}</p>
            </div>
        </div>`;

    // Display steps
    if (data.steps && data.steps.length > 0) {
        html += `
            <div class="result-section">
                <h3 class="section-title">Calculation Steps:</h3>
                <div class="steps-container">`;

        data.steps.forEach((step, index) => {
            html += `
                <div class="calculation-step">
                    <div class="step-number">Step ${index + 1}</div>
                    <div class="step-content">${step}</div>
                </div>`;
        });

        html += '</div></div>';
    }

    // Display final result
    html += `
        <div class="result-section">
            <h3 class="section-title">Final Result:</h3>
            <div class="final-value">f(${data.x_target}) = ${data.result}</div>
        </div>`;

    html += '</div>';
    
    document.getElementById(`${prefix}-result`).innerHTML = html;

    // Display table for Newton's Divided Differences if available
    if (data.table && prefix === 'nd') {
        const tableHtml = generateDividedDiffTable(data.table);
        document.getElementById('nd-table').innerHTML = tableHtml;
    }
}

// Generate divided differences table
function generateDividedDiffTable(tableData) {
    let html = '<div class="table-container"><table class="interpolation-table">';
    
    // Generate header row
    html += '<tr><th>x</th><th>y</th>';
    for (let i = 1; i < tableData[0].length - 1; i++) {
        html += `<th>Δ${i}</th>`;
    }
    html += '</tr>';

    // Generate data rows
    tableData.forEach(row => {
        html += '<tr>';
        row.forEach(cell => {
            html += `<td>${cell !== null ? cell.toFixed(4) : ''}</td>`;
        });
        html += '</tr>';
    });

    html += '</table></div>';
    return html;
}

function calculateInterpolation(method) {
    // Get all point inputs
    const pointsContainer = document.getElementById(`${method}-points-container`);
    const xInputs = pointsContainer.querySelectorAll(`.${method}-x-point`);
    const yInputs = pointsContainer.querySelectorAll(`.${method}-y-point`);
    const x_target = parseFloat(document.getElementById(`${method}-x-target`).value);

    // Collect and validate x and y values
    const x_values = [];
    const y_values = [];
    
    xInputs.forEach((input, index) => {
        const x = parseFloat(input.value);
        const y = parseFloat(yInputs[index].value);
        if (!isNaN(x) && !isNaN(y)) {
            x_values.push(x);
            y_values.push(y);
        }
    });

    // Validate that we have points
    if (x_values.length === 0 || y_values.length === 0) {
        document.getElementById(`${method}-result`).innerHTML = 
            '<div class="error">Please enter valid x and y values</div>';
        return;
    }

    // Make the API call
    fetch('/interpolate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            x_values: x_values,
            y_values: y_values,
            x_target: x_target,
            method: method
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById(`${method}-result`).innerHTML = 
                `<div class="error">${data.error}</div>`;
            return;
        }
        
        // Display results
        const resultDiv = document.getElementById(`${method}-result`);
        let html = '<div class="calculation-steps">';
        
        // Display points being used
        html += '<h4>Points Used:</h4>';
        html += '<div class="points-used">';
        data.points.forEach(([x, y]) => {
            html += `(${x}, ${y}), `;
        });
        html = html.slice(0, -2); // Remove last comma and space
        html += '</div>';
        
        // Display difference table if available
        if (data.difference_table) {
            html += '<h4>Difference Table:</h4>';
            html += '<table class="diff-table">';
            data.difference_table.forEach(row => {
                html += '<tr>' + row.map(val => `<td>${val}</td>`).join('') + '</tr>';
            });
            html += '</table>';
        }

        // Display calculation steps
        html += '<h4>Calculation Steps:</h4>';
        data.steps.forEach(step => {
            html += `<div class="step">${step}</div>`;
        });

        // Display final result
        html += `<div class="final-result">
            <h4>Final Result:</h4>
            <p>f(${data.x_target}) = ${data.result}</p>
        </div>`;

        resultDiv.innerHTML = html;
    })
    .catch(error => {
        document.getElementById(`${method}-result`).innerHTML = 
            '<div class="error">An error occurred during calculation</div>';
    });
}
