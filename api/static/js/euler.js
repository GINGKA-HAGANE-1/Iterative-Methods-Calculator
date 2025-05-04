// Function to solve ODE using Euler method
async function solveEuler() {
    // Get input values
    const functionStr = document.getElementById('function').value;
    const x0 = parseFloat(document.getElementById('x0').value);
    const y0 = parseFloat(document.getElementById('y0').value);
    const h = parseFloat(document.getElementById('step-size').value);
    const n = parseInt(document.getElementById('num-steps').value);
    const method = document.querySelector('input[name="euler-method"]:checked').value;

    // Validate inputs
    if (!functionStr || isNaN(x0) || isNaN(y0) || isNaN(h) || isNaN(n)) {
        alert('Please fill in all fields with valid numbers');
        return;
    }

    try {
        const response = await fetch('http://127.0.0.1:8080/euler', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                function: functionStr,
                x0: x0,
                y0: y0,
                h: h,           
                n: n,           
                method: method
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            throw new TypeError('Response was not JSON');
        }

        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        // Display steps
        displaySteps(data.steps);

        // Plot the graph
        plotGraph(data.points);
    } catch (error) {
        if (error.message.includes('Failed to fetch')) {
            alert('Error: Server is not running. Please start the Flask server.');
        } else if (error.message.includes('404')) {
            alert('Error: Euler method endpoint not found. Please check server configuration.');
        } else {
            alert(`Error: ${error.message}`);
        }
        console.error('Error details:', error);
    }
}

// Function to display solution steps
function displaySteps(steps) {
    const stepsContainer = document.getElementById('euler-steps');
    stepsContainer.innerHTML = steps.map(step => `<div class="step">${step}</div>`).join('');
}

// Function to plot the solution graph
function plotGraph(points) {
    const canvas = document.getElementById('euler-graph');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear previous graph
    ctx.clearRect(0, 0, width, height);

    // Get data ranges
    const xValues = points.map(p => p[0]);
    const yValues = points.map(p => p[1]);
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);

    // Add padding to ranges
    const xPadding = (xMax - xMin) * 0.1;
    const yPadding = (yMax - yMin) * 0.1;

    // Scale factors
    const xScale = (width - 40) / ((xMax - xMin) + 2 * xPadding);
    const yScale = (height - 40) / ((yMax - yMin) + 2 * yPadding);

    // Transform coordinates
    function transformX(x) {
        return 20 + (x - xMin + xPadding) * xScale;
    }

    function transformY(y) {
        return height - (20 + (y - yMin + yPadding) * yScale);
    }

    // Draw axes
    ctx.beginPath();
    ctx.strokeStyle = '#666';
    ctx.moveTo(20, height - 20);
    ctx.lineTo(width - 20, height - 20);
    ctx.moveTo(20, 20);
    ctx.lineTo(20, height - 20);
    ctx.stroke();

    // Plot points and connect them
    ctx.beginPath();
    ctx.strokeStyle = '#2196F3';
    ctx.lineWidth = 2;
    ctx.moveTo(transformX(points[0][0]), transformY(points[0][1]));

    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(transformX(points[i][0]), transformY(points[i][1]));
    }
    ctx.stroke();

    // Plot points
    ctx.fillStyle = '#2196F3';
    points.forEach(point => {
        ctx.beginPath();
        ctx.arc(transformX(point[0]), transformY(point[1]), 3, 0, 2 * Math.PI);
        ctx.fill();
    });
}

function calculateEuler() {
    // Get input values
    const method = document.querySelector('input[name="euler-method"]:checked').value;
    const x0 = parseFloat(document.getElementById('euler-x0').value);
    const y0 = parseFloat(document.getElementById('euler-y0').value);
    const h = parseFloat(document.getElementById('euler-h').value);
    const n = parseInt(document.getElementById('euler-n').value);
    let func = document.getElementById('euler-function').value;

    // Validate inputs
    if (isNaN(x0) || isNaN(y0) || isNaN(h) || isNaN(n)) {
        alert('Please enter valid numerical values');
        return;
    }

    if (!func) {
        alert('Please enter a differential equation');
        return;
    }

    // Prepare data for API request
    const data = {
        method: method,
        x0: x0,
        y0: y0,
        h: h,
        n: n,
        function: func
    };

    // Send request to backend
    fetch('http://127.0.0.1:8080/euler', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            throw new TypeError('Response was not JSON');
        }
        return response.json();
    })
    .then(data => {
        // Make sure we have a valid container ID and properly formatted data
        if (data && data.steps) {
            // Create a container for Euler results if it doesn't exist
            const container = document.querySelector('#euler-result .iterations');
            if (container) {
                container.innerHTML = '';
                
                // Display the steps
                if (data.steps.explanation) {
                    const stepsDiv = document.createElement('div');
                    stepsDiv.className = 'euler-steps';
                    data.steps.explanation.forEach(step => {
                        const stepDiv = document.createElement('div');
                        stepDiv.className = 'step';
                        stepDiv.textContent = step;
                        stepsDiv.appendChild(stepDiv);
                    });
                    container.appendChild(stepsDiv);
                }
                
                // Display the points
                if (data.steps.points && data.steps.points.length > 0) {
                    const pointsDiv = document.createElement('div');
                    pointsDiv.className = 'euler-points';
                    pointsDiv.innerHTML = '<h4>Solution Points:</h4>';
                    
                    const pointsList = document.createElement('div');
                    data.steps.points.forEach(point => {
                        const pointDiv = document.createElement('div');
                        pointDiv.textContent = point.formatted;
                        pointsList.appendChild(pointDiv);
                    });
                    
                    pointsDiv.appendChild(pointsList);
                    container.appendChild(pointsDiv);
                }
            } else {
                console.error('Euler results container not found');
            }
        } else {
            console.error('Invalid data format received from server:', data);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        if (error.message.includes('404')) {
            alert('Error: Euler method endpoint not found. Please check server configuration.');
        } else if (error instanceof TypeError) {
            alert('Error: Server returned invalid response format. Please check server configuration.');
        } else {
            alert(`Error: ${error.message}`);
        }
    });
}

function displayResults(data) {
    const resultDiv = document.getElementById('euler-result');
    const stepsDiv = document.getElementById('euler-steps');

    // Clear previous results
    resultDiv.innerHTML = '';
    stepsDiv.innerHTML = '';

    // Check if there's an error in the response
    if (data.error) {
        resultDiv.innerHTML = `<div class="error">${data.error}</div>`;
        if (data.details) {
            stepsDiv.innerHTML = `<div class="error-details">${data.details}</div>`;
        }
        return;
    }

    // Create table for points
    let tableHTML = '<table class="result-table"><tr><th>Step</th><th>x</th><th>y</th></tr>';
    if (data.points && Array.isArray(data.points)) {
        data.points.forEach((point, index) => {
            tableHTML += `<tr>
                <td>${index}</td>
                <td>${point[0].toFixed(5)}</td>
                <td>${point[1].toFixed(5)}</td>
            </tr>`;
        });
    }
    tableHTML += '</table>';
    resultDiv.innerHTML = tableHTML;

    // Display calculation steps
    let stepsHTML = '<h4>Calculation Steps:</h4>';
    if (data.steps && Array.isArray(data.steps)) {
        data.steps.forEach(step => {
            stepsHTML += `<div class="step">${step}</div>`;
        });
    }
    stepsDiv.innerHTML = stepsHTML;
}

// Add event listeners for example functions
document.addEventListener('DOMContentLoaded', function() {
    const examples = document.querySelectorAll('.example-code');
    examples.forEach(example => {
        example.addEventListener('click', function() {
            document.getElementById('euler-function').value = this.textContent;
        });
    });
    
    // Add visual feedback for inputs
    const inputs = document.querySelectorAll('.enhanced-input');
    inputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.parentElement.classList.add('focused');
        });
        input.addEventListener('blur', function() {
            this.parentElement.classList.remove('focused');
            
            // Validate input
            if (this.value === '') {
                this.classList.add('invalid');
            } else {
                this.classList.remove('invalid');
            }
        });
    });
});