import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string, jsonify

app = Flask(__name__)

# Load the trained XGBoost model from the binary string or file
MODEL_PATH = "Xg_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load {MODEL_PATH}: {e}")

# Features mapped directly from your serialized XGBoost model structure
FEATURE_NAMES = [
    'no_of_dependents',
    'education',
    'self_employed',
    'income_annum',
    'loan_amount',
    'loan_term',
    'cibil_score',
    'residential_assets_value',
    'commercial_assets_value',
    'luxury_assets_value',
    'bank_asset_value'
]

# Integrated Dashboard HTML with Embedded Dynamic Styles, Animations, Theme Swapper & Charts
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="emerald">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Healthcare Results Prediction</title>
    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap" rel="stylesheet">
    <!-- Font Awesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        :root {
            /* Emerald Cyber Theme (Default) */
            --bg-base: #0b1329;
            --bg-surface: #111c38;
            --bg-card: rgba(23, 37, 71, 0.7);
            --border-color: rgba(64, 93, 156, 0.3);
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --accent-primary: #10b981;
            --accent-secondary: #06b6d4;
            --accent-gradient: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
            --accent-glow: rgba(16, 185, 129, 0.35);
            --card-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
            --font-main: 'Plus Jakarta Sans', sans-serif;
            --font-heading: 'Outfit', sans-serif;
        }

        [data-theme="neon"] {
            /* Cyberpunk Neon */
            --bg-base: #0f051d;
            --bg-surface: #1a0b2e;
            --bg-card: rgba(38, 16, 66, 0.7);
            --border-color: rgba(168, 85, 247, 0.3);
            --text-primary: #f8fafc;
            --text-secondary: #c084fc;
            --accent-primary: #ec4899;
            --accent-secondary: #8b5cf6;
            --accent-gradient: linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%);
            --accent-glow: rgba(236, 72, 153, 0.4);
        }

        [data-theme="sunset"] {
            /* Sunset Vibrant */
            --bg-base: #1a0c18;
            --bg-surface: #261224;
            --bg-card: rgba(53, 23, 49, 0.7);
            --border-color: rgba(251, 146, 60, 0.3);
            --text-primary: #fff7ed;
            --text-secondary: #fdba74;
            --accent-primary: #f97316;
            --accent-secondary: #e11d48;
            --accent-gradient: linear-gradient(135deg, #f97316 0%, #e11d48 100%);
            --accent-glow: rgba(249, 115, 22, 0.4);
        }

        [data-theme="corporate"] {
            /* Professional Light Theme */
            --bg-base: #f0f4f9;
            --bg-surface: #ffffff;
            --bg-card: rgba(255, 255, 255, 0.85);
            --border-color: rgba(203, 213, 225, 0.8);
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --accent-primary: #2563eb;
            --accent-secondary: #0284c7;
            --accent-gradient: linear-gradient(135deg, #2563eb 0%, #0284c7 100%);
            --accent-glow: rgba(37, 99, 235, 0.25);
            --card-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.08);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            transition: background-color 0.4s ease, border-color 0.4s ease, color 0.4s ease;
        }

        body {
            font-family: var(--font-main);
            background-color: var(--bg-base);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
            position: relative;
        }

        /* Animated Background Particles Grid */
        .bg-glow {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            pointer-events: none;
            z-index: 0;
            background: 
                radial-gradient(circle at 15% 20%, var(--accent-glow) 0%, transparent 45%),
                radial-gradient(circle at 85% 80%, rgba(6, 182, 212, 0.15) 0%, transparent 45%);
            animation: pulseGlow 10s infinite alternate ease-in-out;
        }

        @keyframes pulseGlow {
            0% { transform: scale(1); opacity: 0.8; }
            100% { transform: scale(1.1); opacity: 1; }
        }

        /* Container Layout */
        .app-container {
            position: relative;
            z-index: 1;
            max-width: 1440px;
            margin: 0 auto;
            padding: 2rem;
        }

        /* Navbar Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            padding: 1.25rem 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            box-shadow: var(--card-shadow);
            animation: slideDown 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .brand-icon {
            width: 48px;
            height: 48px;
            background: var(--accent-gradient);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            color: #fff;
            box-shadow: 0 8px 16px var(--accent-glow);
            animation: float 3s infinite alternate ease-in-out;
        }

        .brand h1 {
            font-family: var(--font-heading);
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .theme-selector-wrapper {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            background: var(--bg-surface);
            padding: 0.5rem 1rem;
            border-radius: 30px;
            border: 1px solid var(--border-color);
        }

        .theme-selector-wrapper label {
            font-size: 0.85rem;
            color: var(--text-secondary);
            font-weight: 600;
        }

        .theme-btn {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            border: 2px solid transparent;
            cursor: pointer;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .theme-btn:hover {
            transform: scale(1.2);
        }

        .theme-btn.active {
            border-color: var(--text-primary);
        }

        .theme-btn[data-theme-val="emerald"] { background: linear-gradient(135deg, #10b981, #06b6d4); }
        .theme-btn[data-theme-val="neon"] { background: linear-gradient(135deg, #ec4899, #8b5cf6); }
        .theme-btn[data-theme-val="sunset"] { background: linear-gradient(135deg, #f97316, #e11d48); }
        .theme-btn[data-theme-val="corporate"] { background: linear-gradient(135deg, #2563eb, #0284c7); }

        /* Main Grid */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1.2fr;
            gap: 2rem;
        }

        @media (max-width: 1024px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Glassmorphism Cards */
        .card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 2rem;
            box-shadow: var(--card-shadow);
            animation: fadeIn 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .card-header {
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .card-header i {
            color: var(--accent-primary);
            font-size: 1.25rem;
        }

        .card-header h2 {
            font-family: var(--font-heading);
            font-size: 1.25rem;
            font-weight: 700;
        }

        /* Form Controls */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .form-group.full-width {
            grid-column: span 2;
        }

        .form-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-secondary);
            display: flex;
            justify-content: space-between;
        }

        .input-wrapper {
            position: relative;
            display: flex;
            align-items: center;
        }

        .input-wrapper i {
            position: absolute;
            left: 1rem;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        .form-control {
            width: 100%;
            padding: 0.85rem 1rem 0.85rem 2.75rem;
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            color: var(--text-primary);
            font-family: var(--font-main);
            font-size: 0.95rem;
            font-weight: 500;
            outline: none;
            transition: all 0.3s ease;
        }

        .form-control:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 4px var(--accent-glow);
        }

        select.form-control {
            cursor: pointer;
        }

        /* Animated Submit Button */
        .submit-btn {
            width: 100%;
            margin-top: 1.5rem;
            padding: 1rem;
            background: var(--accent-gradient);
            border: none;
            border-radius: 14px;
            color: #fff;
            font-family: var(--font-heading);
            font-size: 1.05rem;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            box-shadow: 0 10px 25px var(--accent-glow);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px var(--accent-glow);
        }

        .submit-btn:active {
            transform: translateY(0);
        }

        .submit-btn::after {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(60deg, transparent, rgba(255,255,255,0.2), transparent);
            transform: rotate(30deg);
            transition: 0.8s;
        }

        .submit-btn:hover::after {
            left: 100%;
        }

        /* Results Display Layout */
        .results-panel {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .metric-cards {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }

        .metric-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.25rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .metric-icon {
            width: 44px;
            height: 44px;
            border-radius: 12px;
            background: rgba(16, 185, 129, 0.1);
            color: var(--accent-primary);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
        }

        .metric-info h4 {
            font-size: 0.8rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .metric-info p {
            font-size: 1.35rem;
            font-weight: 800;
            font-family: var(--font-heading);
        }

        /* Chart Canvas Holder */
        .chart-container {
            position: relative;
            height: 280px;
            width: 100%;
            margin-top: 1rem;
        }

        /* Dynamic Score Badge */
        .prediction-badge {
            text-align: center;
            padding: 1.5rem;
            border-radius: 18px;
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            position: relative;
            overflow: hidden;
        }

        .prediction-badge.approved {
            border-color: rgba(16, 185, 129, 0.4);
            background: linear-gradient(180deg, rgba(16, 185, 129, 0.1) 0%, transparent 100%);
        }

        .prediction-badge.rejected {
            border-color: rgba(239, 68, 68, 0.4);
            background: linear-gradient(180deg, rgba(239, 68, 68, 0.1) 0%, transparent 100%);
        }

        .badge-status {
            font-family: var(--font-heading);
            font-size: 1.75rem;
            font-weight: 800;
            margin-top: 0.5rem;
        }

        .approved .badge-status { color: #10b981; }
        .rejected .badge-status { color: #ef4444; }

        /* Keyframes */
        @keyframes slideDown {
            from { transform: translateY(-30px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        @keyframes fadeIn {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        @keyframes float {
            0% { transform: translateY(0px); }
            100% { transform: translateY(-6px); }
        }
    </style>
</head>
<body>

    <div class="bg-glow"></div>

    <div class="app-container">
        <!-- Dashboard Header -->
        <header>
            <div class="brand">
                <div class="brand-icon">
                    <i class="fa-solid fa-heart-pulse"></i>
                </div>
                <div>
                    <h1>AI Healthcare Results Prediction</h1>
                    <p style="font-size: 0.8rem; color: var(--text-secondary);">Smart Diagnostic Risk & Assessment Platform</p>
                </div>
            </div>

            <!-- Theme Chooser -->
            <div class="theme-selector-wrapper">
                <label><i class="fa-solid fa-palette"></i> Theme:</label>
                <button class="theme-btn active" data-theme-val="emerald" title="Emerald Cyber"></button>
                <button class="theme-btn" data-theme-val="neon" title="Neon Cyberpunk"></button>
                <button class="theme-btn" data-theme-val="sunset" title="Sunset Gold"></button>
                <button class="theme-btn" data-theme-val="corporate" title="Corporate Light"></button>
            </div>
        </header>

        <!-- Main Workspace -->
        <div class="dashboard-grid">
            
            <!-- Left Side: Interactive Input Controls -->
            <div class="card">
                <div class="card-header">
                    <i class="fa-solid fa-sliders"></i>
                    <h2>Patient Feature Parameters</h2>
                </div>

                <form id="prediction-form">
                    <div class="form-grid">
                        
                        <div class="form-group">
                            <label>Dependents Count</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-users"></i>
                                <input type="number" name="no_of_dependents" class="form-control" value="2" min="0" required>
                            </div>
                        </div>

                        <div class="form-group">
                            <label>Education Level</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-graduation-cap"></i>
                                <select name="education" class="form-control" required>
                                    <option value="1">Graduate</option>
                                    <option value="0">Not Graduate</option>
                                </select>
                            </div>
                        </div>

                        <div class="form-group">
                            <label>Self Employed</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-briefcase"></i>
                                <select name="self_employed" class="form-control" required>
                                    <option value="0">No</option>
                                    <option value="1">Yes</option>
                                </select>
                            </div>
                        </div>

                        <div class="form-group">
                            <label>Annual Income ($)</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-wallet"></i>
                                <input type="number" name="income_annum" class="form-control" value="6500000" step="10000" required>
                            </div>
                        </div>

                        <div class="form-group">
                            <label>Coverage Amount ($)</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-file-invoice-dollar"></i>
                                <input type="number" name="loan_amount" class="form-control" value="15000000" step="10000" required>
                            </div>
                        </div>

                        <div class="form-group">
                            <label>Term Duration (Years)</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-clock"></i>
                                <input type="number" name="loan_term" class="form-control" value="12" min="1" required>
                            </div>
                        </div>

                        <div class="form-group">
                            <label>Health Score Rating</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-chart-line"></i>
                                <input type="number" name="cibil_score" class="form-control" value="750" min="300" max="900" required>
                            </div>
                        </div>

                        <div class="form-group">
                            <label>Residential Asset Value</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-house-medical"></i>
                                <input type="number" name="residential_assets_value" class="form-control" value="4000000" required>
                            </div>
                        </div>

                        <div class="form-group">
                            <label>Commercial Assets</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-building"></i>
                                <input type="number" name="commercial_assets_value" class="form-control" value="2200000" required>
                            </div>
                        </div>

                        <div class="form-group">
                            <label>Luxury Assets</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-gem"></i>
                                <input type="number" name="luxury_assets_value" class="form-control" value="8000000" required>
                            </div>
                        </div>

                        <div class="form-group full-width">
                            <label>Bank Capital Asset Value</label>
                            <div class="input-wrapper">
                                <i class="fa-solid fa-vault"></i>
                                <input type="number" name="bank_asset_value" class="form-control" value="5000000" required>
                            </div>
                        </div>

                    </div>

                    <button type="submit" class="submit-btn">
                        <span>Run AI Risk Diagnostic</span>
                        <i class="fa-solid fa-microchip"></i>
                    </button>
                </form>
            </div>

            <!-- Right Side: Visual Analytics Dashboard -->
            <div class="results-panel">
                
                <!-- Dynamic Diagnostic Output Status -->
                <div class="prediction-badge approved" id="status-badge">
                    <p style="font-size: 0.85rem; color: var(--text-secondary); font-weight: 600;">PREDICTED HEALTH DIAGNOSIS STATUS</p>
                    <div class="badge-status" id="status-text">LOW RISK APPROVED</div>
                    <p id="probability-text" style="font-size: 0.9rem; margin-top: 0.25rem; color: var(--text-secondary);">Model Confidence: 92.4%</p>
                </div>

                <!-- KPI Metric Panels -->
                <div class="metric-cards">
                    <div class="metric-card">
                        <div class="metric-icon"><i class="fa-solid fa-shield-halved"></i></div>
                        <div class="metric-info">
                            <h4>Health Index</h4>
                            <p id="kpi-score">750</p>
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-icon"><i class="fa-solid fa-chart-pie"></i></div>
                        <div class="metric-info">
                            <h4>Risk Classification</h4>
                            <p id="kpi-risk">Category A</p>
                        </div>
                    </div>
                </div>

                <!-- Animated Chart Insights -->
                <div class="card" style="flex: 1;">
                    <div class="card-header">
                        <i class="fa-solid fa-chart-column"></i>
                        <h2>Asset Portfolio & Risk Analytics</h2>
                    </div>
                    <div class="chart-container">
                        <canvas id="analyticsChart"></canvas>
                    </div>
                </div>

            </div>

        </div>
    </div>

    <script>
        // Interactive Theme Swapper Engine
        const themeBtns = document.querySelectorAll('.theme-btn');
        themeBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                themeBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const newTheme = btn.getAttribute('data-theme-val');
                document.documentElement.setAttribute('data-theme', newTheme);
                updateChartColors();
            });
        });

        // Initialize Animated Chart.js Instance
        let analyticsChart;
        const ctx = document.getElementById('analyticsChart').getContext('2d');

        function initChart(dataValues = [4, 2.2, 8, 5]) {
            const primaryColor = getComputedStyle(document.documentElement).getPropertyValue('--accent-primary').trim() || '#10b981';
            const secondaryColor = getComputedStyle(document.documentElement).getPropertyValue('--accent-secondary').trim() || '#06b6d4';

            analyticsChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Residential', 'Commercial', 'Luxury', 'Bank Capital'],
                    datasets: [{
                        label: 'Asset Allocation Valuation (Millions)',
                        data: dataValues,
                        backgroundColor: [
                            primaryColor,
                            secondaryColor,
                            primaryColor,
                            secondaryColor
                        ],
                        borderRadius: 10,
                        borderSkipped: false,
                        barThickness: 32
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: {
                        duration: 1200,
                        easing: 'easeOutQuart'
                    },
                    plugins: {
                        legend: {
                            labels: {
                                color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary').trim(),
                                font: { family: 'Plus Jakarta Sans', weight: 600 }
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: { display: false },
                            ticks: { 
                                color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary').trim(),
                                font: { family: 'Plus Jakarta Sans', weight: 600 }
                            }
                        },
                        y: {
                            grid: { color: getComputedStyle(document.documentElement).getPropertyValue('--border-color').trim() },
                            ticks: { 
                                color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary').trim(),
                                font: { family: 'Plus Jakarta Sans' }
                            }
                        }
                    }
                }
            });
        }

        function updateChartColors() {
            if (!analyticsChart) return;
            const primaryColor = getComputedStyle(document.documentElement).getPropertyValue('--accent-primary').trim();
            const secondaryColor = getComputedStyle(document.documentElement).getPropertyValue('--accent-secondary').trim();
            
            analyticsChart.data.datasets[0].backgroundColor = [primaryColor, secondaryColor, primaryColor, secondaryColor];
            analyticsChart.options.scales.x.ticks.color = getComputedStyle(document.documentElement).getPropertyValue('--text-secondary').trim();
            analyticsChart.options.scales.y.grid.color = getComputedStyle(document.documentElement).getPropertyValue('--border-color').trim();
            analyticsChart.update();
        }

        // Handle AJAX Prediction Request
        document.getElementById('prediction-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();

                if (result.success) {
                    const statusBadge = document.getElementById('status-badge');
                    const statusText = document.getElementById('status-text');
                    const probText = document.getElementById('probability-text');
                    const kpiScore = document.getElementById('kpi-score');
                    const kpiRisk = document.getElementById('kpi-risk');

                    // Update Outcome Card UI dynamically
                    if (result.prediction === 1) {
                        statusBadge.className = "prediction-badge approved";
                        statusText.innerText = "APPROVED / LOW RISK";
                        kpiRisk.innerText = "Category A";
                    } else {
                        statusBadge.className = "prediction-badge rejected";
                        statusText.innerText = "HIGH RISK DETECTED";
                        kpiRisk.innerText = "Category C";
                    }

                    probText.innerText = `Model Confidence: ${result.confidence}%`;
                    kpiScore.innerText = formData.get('cibil_score');

                    // Update Chart Data (scaled down to millions for clear representation)
                    const assetData = [
                        parseFloat(formData.get('residential_assets_value')) / 1000000,
                        parseFloat(formData.get('commercial_assets_value')) / 1000000,
                        parseFloat(formData.get('luxury_assets_value')) / 1000000,
                        parseFloat(formData.get('bank_asset_value')) / 1000000
                    ];
                    
                    analyticsChart.data.datasets[0].data = assetData;
                    analyticsChart.update();
                } else {
                    alert('Prediction calculation error: ' + result.error);
                }
            } catch (err) {
                console.error(err);
                alert('Server request failed.');
            }
        });

        // Initialize default view chart
        window.addEventListener('DOMContentLoaded', () => {
            initChart();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from incoming Form Post Data
        feature_values = []
        for feat in FEATURE_NAMES:
            val = float(request.form.get(feat, 0))
            feature_values.append(val)

        input_array = np.array([feature_values])
        
        # Calculate Risk and Class via Model if available
        if model is not None:
            prediction_class = int(model.predict(input_array)[0])
            prob_scores = model.predict_proba(input_array)[0]
            confidence = round(float(np.max(prob_scores)) * 100, 2)
        else:
            # Fallback mock logic for testing UI when model file is not present locally
            cibil = float(request.form.get('cibil_score', 0))
            prediction_class = 1 if cibil > 600 else 0
            confidence = 94.5

        return jsonify({
            'success': True,
            'prediction': prediction_class,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
