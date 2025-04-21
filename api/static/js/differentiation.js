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
    const x_target = document.getElementById('diff-x-value').value;
    
    // Collect x and y values
    const x_values = [];
    const y_values = [];
    
    xInputs.forEach(input => {
        if (input.value) {
            x_values.push(input.value);
        }
    });
    
    yInputs.forEach(input => {
        if (input.value) {
            y_values.push(input.value);
        }
    });
    
    // Validate inputs
    if (x_values.length < 3 || y_values.length < 3) {
        document.getElementById('diff-result').innerHTML = '<div class="error">Please enter at least 3 points</div>';
        return;
    }
    
    if (!x_target) {
        document.getElementById('diff-result').innerHTML = '<div class="error">Please enter the point of differentiation</div>';
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
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('diff-result').innerHTML = `<div class="error">${data.error}</div>`;
            return;
        }
        
        // Display results
        let resultHtml = '<div class="diff-results">';
        resultHtml += '<div class="calculation-steps">';
        data.steps.forEach(step => {
            resultHtml += `<div class="step">${step}</div>`;
        });
        resultHtml += '</div>';
        
        // Display derivatives
        resultHtml += '<div class="derivatives-summary">';
        resultHtml += '<h3>Derivatives Summary</h3>';
        Object.entries(data.derivatives).forEach(([key, value]) => {
            const order = key.split('_')[1];
            resultHtml += `<div class="derivative ${key}">
                            <span class="order">${order}${getOrdinalSuffix(order)} Derivative:</span>
                            <span class="value">${value}</span>
                          </div>`;
        });
        resultHtml += '</div>';
        
        document.getElementById('diff-result').innerHTML = resultHtml;
    })
    .catch(error => {
        document.getElementById('diff-result').innerHTML = '<div class="error">An error occurred while calculating</div>';
        console.error('Error:', error);
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