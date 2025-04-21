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
    const xTarget = document.getElementById('diff-x-value').value;

    const x_values = Array.from(xInputs).map(input => input.value);
    const y_values = Array.from(yInputs).map(input => input.value);

    fetch('/differentiate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            x_values: x_values,
            y_values: y_values,
            x_target: xTarget,
            method: method
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('diff-result').innerHTML = `<div class="error">${data.error}</div>`;
            return;
        }

        let resultHtml = '<div class="diff-results">';
        
        // Display steps
        resultHtml += '<div class="calculation-steps">';
        data.steps.forEach(step => {
            if (step.includes("===")) {
                resultHtml += `<h3 class="results-header">${step.replace(/=/g, '')}</h3>`;
            } else if (step.includes("Derivative")) {
                resultHtml += `<div class="derivative-section"><h4>${step}</h4>`;
            } else if (step.includes("f(x)")) {
                resultHtml += `<div class="fx-section"><h4>${step}</h4>`;
            } else {
                resultHtml += `<div class="step">${step}</div>`;
            }
        });
        resultHtml += '</div>';
        
        // Display derivatives summary
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
        
        // Display f(x) value
        resultHtml += '<div class="fx-value">';
        resultHtml += '<h3>Function Value</h3>';
        resultHtml += `<div class="value">f(x) = ${data.fx_value}</div>`;
        resultHtml += '</div>';
        
        resultHtml += '</div>';
        document.getElementById('diff-result').innerHTML = resultHtml;
    })
    .catch(error => {
        document.getElementById('diff-result').innerHTML = '<div class="error">An error occurred while calculating</div>';
    });
}

// Add event listener for points input
document.addEventListener('DOMContentLoaded', function() {
    const pointsInput = document.getElementById('diff-points');
    if (pointsInput) {
        pointsInput.addEventListener('change', updateDiffPoints);
        updateDiffPoints(); // Initialize points inputs
    }
});


function getOrdinalSuffix(n) {
    const s = ["th", "st", "nd", "rd"];
    const v = n % 100;
    return s[(v - 20) % 10] || s[v] || s[0];
}