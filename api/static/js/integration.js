function calculateNumericalIntegration() {
    // Get input values
    const method = document.getElementById('integration-method').value;
    const lowerLimit = document.getElementById('ni-lower').value.trim();
    const upperLimit = document.getElementById('ni-upper').value.trim();
    const intervals = parseInt(document.getElementById('ni-intervals').value);
    const functionStr = document.getElementById('ni-function').value.trim();

    // Process pi symbols in limits
    const processLimit = (limit) => {
        if (!limit) return '';
        return limit.replace('π', 'pi').replace('PI', 'pi').replace('Pi', 'pi');
    };

    const processedLower = processLimit(lowerLimit);
    const processedUpper = processLimit(upperLimit);

    // Input validation
    if (!processedLower || !processedUpper || isNaN(intervals)) {
        document.getElementById('ni-result').innerHTML = 
            '<div class="error">Please enter valid numeric values for limits and intervals</div>';
        return;
    }

    if (!functionStr) {
        document.getElementById('ni-result').innerHTML = 
            '<div class="error">Please enter a function</div>';
        return;
    }

    // Make API call
    fetch('/integrate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            lower_limit: processedLower,
            upper_limit: processedUpper,
            intervals: intervals,
            function: functionStr,
            method: method
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('ni-result').innerHTML = 
                `<div class="error">${data.error}</div>`;
            return;
        }

        // Display results
        let html = '<div class="integration-results">';
        
        // Method title
        html += `<h4>${method === 'trapezoidal' ? 'Trapezoidal Rule' : "Simpson's 1/3 Rule"}</h4>`;
        
        // Function and interval info
        html += `<div class="integration-info">
            <p>∫ ${functionStr} dx</p>
            <p>Interval: [${lowerLimit}, ${upperLimit}]</p>
            <p>Number of subintervals: ${intervals}</p>
        </div>`;

        // Display calculation steps
        if (data.steps && data.steps.length > 0) {
            html += '<div class="calculation-steps">';
            html += '<h4>Calculation Steps:</h4>';
            data.steps.forEach(step => {
                html += `<div class="step">${step}</div>`;
            });
            html += '</div>';
        }

        // Display final result
        html += `<div class="final-result">
            <h4>Result:</h4>
            <p>∫<sub>${lowerLimit}</sub><sup>${upperLimit}</sup> ${functionStr} dx = ${data.result}</p>
        </div>`;

        html += '</div>';
        
        document.getElementById('ni-result').innerHTML = html;
    })
    .catch(error => {
        document.getElementById('ni-result').innerHTML = 
            '<div class="error">An error occurred during calculation</div>';
        console.error('Error:', error);
    });
}