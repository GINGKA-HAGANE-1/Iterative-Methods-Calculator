function updateDiffPoints() {
    const numPoints = parseInt(document.getElementById('diff-points').value);
    const container = document.querySelector('.point-pairs');
    container.innerHTML = '';

    for (let i = 0; i < numPoints; i++) {
        container.innerHTML += `
            <div class="point-pair">
                <div class="point-input">
                    <label>x${i}:</label>
                    <input type="text" class="diff-x form-control" data-index="${i}">
                </div>
                <div class="point-input">
                    <label>y${i}:</label>
                    <input type="text" class="diff-y form-control" data-index="${i}">
                </div>
            </div>
        `;
    }
}

function calculateDifferentiation() {
    const method = document.getElementById('diff-method').value;
    const xInputs = document.querySelectorAll('.diff-x');
    const yInputs = document.querySelectorAll('.diff-y');
    const x_target = parseFloat(document.getElementById('diff-x-value').value);
    
    // Collect x and y values
    const x_values = [];
    const y_values = [];
    
    xInputs.forEach(input => {
        if (input.value.trim() !== '') {
            x_values.push(parseFloat(input.value));
        }
    });
    
    yInputs.forEach(input => {
        if (input.value.trim() !== '') {
            y_values.push(parseFloat(input.value));
        }
    });
    
    // Validate inputs
    if (x_values.length < 4) {
        document.getElementById('diff-result').innerHTML = '<div class="error">Please use 4 points for Newton Forward Differentiation to calculate up to third derivative. Current points: ' + x_values.length + '</div>';
        return;
    }
    
    if (isNaN(x_target)) {
        document.getElementById('diff-result').innerHTML = '<div class="error">Please enter a valid point of differentiation</div>';
        return;
    }
    
    // Send data to server
    fetch('/differentiate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            method: method,
            x_values: x_values,
            y_values: y_values,
            x_target: x_target
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            document.getElementById('diff-result').innerHTML = `<div class="error">${data.error}</div>`;
            return;
        }
        
        // Display results with null check
        let resultHtml = '<div class="diff-results">';
        resultHtml += '<div class="calculation-steps">';
        if (data.steps && Array.isArray(data.steps)) {
            data.steps.forEach(step => {
                resultHtml += `<div class="step">${step}</div>`;
            });
        }
        resultHtml += '</div>';
        
        // Display function value if available with proper x_target check
        if (data.fx_value !== undefined) {
            resultHtml += `<div class="function-value"><h3>Function Value</h3>`;
            resultHtml += `<div class="value">f(${x_target}) = ${data.fx_value}</div></div>`;
        }
        
        // Display derivatives with null check
        resultHtml += '<div class="derivatives-summary">';
        resultHtml += '<h3>Derivatives Summary</h3>';
        if (data.derivatives && typeof data.derivatives === 'object') {
            Object.entries(data.derivatives).forEach(([key, value]) => {
                const order = key.split('_')[1];
                resultHtml += `<div class="derivative ${key}">
                                <span class="order">${order}${getOrdinalSuffix(order)} Derivative:</span>
                                <span class="value">${value}</span>
                              </div>`;
            });
        }
        resultHtml += '</div>';
        
        document.getElementById('diff-result').innerHTML = resultHtml;
    })
    .catch(error => {
        
        console.error('Error:', error);
        document.getElementById('diff-result').innerHTML = `<div class="error">An error occurred: ${error.message}</div>`;
    });
}

function getOrdinalSuffix(n) {
    const s = ["th", "st", "nd", "rd"];
    const v = n % 100;
    return s[(v - 20) % 10] || s[v] || s[0];
}

// Add event listener for points input
document.addEventListener('DOMContentLoaded', function() {
    const pointsInput = document.getElementById('diff-points');
    if (pointsInput) {
        pointsInput.addEventListener('change', updateDiffPoints);
        updateDiffPoints(); // Initialize points inputs
    }
});