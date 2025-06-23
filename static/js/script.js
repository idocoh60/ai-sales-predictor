
function sendPrediction() {
    const totalSpentValue = document.getElementById("totalSpent").value;
    const data = {
        NumPurchases: document.getElementById("numPurchases").value,
        TotalSpent: totalSpentValue,
        DaysSinceLastPurchase: document.getElementById("daysSince").value
    };

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    })
    .then(res => res.json())
    .then(result => {
        document.getElementById("formSection").style.display = "none";
        document.getElementById("result").style.display = "block";
        document.getElementById("willBuy").innerText = result.willBuy;
        document.getElementById("expectedAmount").innerText = result.expectedAmount + "₪";
        document.getElementById("recommendedProduct").innerText = result.recommendedProduct;
        document.getElementById("salesPitch").innerText = result.salesPitch;

        renderCharts(parseFloat(data.TotalSpent), parseFloat(result.expectedAmount), result.probability);
    });
}

function resetForm() {
    document.getElementById("formSection").style.display = "block";
    document.getElementById("result").style.display = "none";
    document.getElementById("numPurchases").value = "";
    document.getElementById("totalSpent").value = "";
    document.getElementById("daysSince").value = "";
}

function renderCharts(prevAmount, predictedAmount, probability) {
    const ctx1 = document.getElementById('spendingChart').getContext('2d');
    new Chart(ctx1, {
        type: 'bar',
        data: {
            labels: ['Previous Spending', 'Predicted Spending'],
            datasets: [{
                label: '₪',
                data: [prevAmount, predictedAmount],
                backgroundColor: ['#7cb5ec', '#66bb6a']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false }
            }
        }
    });

    const ctx2 = document.getElementById('probabilityChart').getContext('2d');
    new Chart(ctx2, {
        type: 'doughnut',
        data: {
            labels: ['Likely to Buy', 'Unlikely'],
            datasets: [{
                data: [Math.round(probability * 100), 100 - Math.round(probability * 100)],
                backgroundColor: ['#81c784', '#ef5350']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'bottom' }
            }
        }
    });
}
