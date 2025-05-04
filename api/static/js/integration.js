// Add event listeners for example functions
document.addEventListener('DOMContentLoaded', function() {
    const examples = document.querySelectorAll('.example-code');
    examples.forEach(example => {
        example.addEventListener('click', function() {
            document.getElementById('ni-function').value = this.textContent.trim();
        });
    });
});

function calculateNumericalIntegration() {
    // Get input values
    const method = document.getElementById('integration-method').value;
    const lowerLimit = document.getElementById('ni-lower').value.trim();
    const upperLimit = document.getElementById('ni-upper').value.trim();
    const intervals = parseInt(document.getElementById('ni-intervals').value);
    const functionStr = document.getElementById('ni-function').value.trim();

    // Process math symbols in limits
    const processLimit = (limit) => {
        if (!limit) return '';
        // Check if it's a simple number first
        if (!isNaN(parseFloat(limit))) {
            return limit; // Return as is if it's a simple number
        }
        // Otherwise process pi and other symbols
        return limit
            .replace(/π/g, 'np.pi')
            .replace(/PI/g, 'np.pi')
            .replace(/Pi/g, 'np.pi')
            .replace(/pi/g, 'np.pi');
    };

    const processedLower = processLimit(lowerLimit);
    const processedUpper = processLimit(upperLimit);

    // Input validation
    if ((!processedLower && processedLower !== '0') || (!processedUpper && processedUpper !== '0') || isNaN(intervals)) {
        document.getElementById('ni-result').innerHTML = 
            '<div class="error">Please enter valid numeric values for limits and intervals</div>';
        return;
    }

    if (!functionStr) {
        document.getElementById('ni-result').innerHTML = 
            '<div class="error">Please enter a function</div>';
        return;
    }

    // Check for multiple variables (only x is allowed)
    if (functionStr.includes('y') || functionStr.includes('z')) {
        document.getElementById('ni-result').innerHTML = 
            '<div class="error">Integration only supports functions of one variable (x). Please remove other variables.</div>';
        return;
    }

    // Show loading state
    document.getElementById('ni-result').innerHTML = '<div class="loading">Calculating...</div>';

    // Debug: Log what's being sent to the server
    const requestData = {
        lower_limit: processedLower,
        upper_limit: processedUpper,
        intervals: intervals,
        function: functionStr,
        method: method
    };
    

    // Make API call
    fetch('http://127.0.0.1:8080/integrate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
    })
    .then(response => {
        
        if (!response.ok) {
            return response.text().then(text => {
                
                throw new Error(`HTTP error! status: ${response.status}`);
            });
        }
        return response.json();
    })
    
    .then(data => {
        
        if (data.error) {
            document.getElementById('ni-result').innerHTML = 
                `<div class="error">${data.error}</div>`;
            return;
        }
        
        // Display results
        let html = '<div class="integration-result">';
        html += `<h4>Method: ${method.charAt(0).toUpperCase() + method.slice(1)}</h4>`;
        html += `<div class="function">∫ ${functionStr} dx from ${lowerLimit} to ${upperLimit}</div>`;
        
        // Display steps
        if (data.steps && data.steps.length > 0) {
            html += '<div class="steps-container">';
            data.steps.forEach(step => {
                html += `<div class="step">${step}</div>`;
            });
            html += '</div>';
        }
        
        // Display final result
        html += `<div class="final-result">Result: ${data.result.toFixed(6)}</div>`;
        html += '</div>';
        
        document.getElementById('ni-result').innerHTML = html;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('ni-result').innerHTML = 
            `<div class="error">Error: ${error.message}. Make sure the server is running at http://127.0.0.1:8080</div>`;
    });
}