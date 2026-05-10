from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

@app.route('/')
def index():
    # Parse the evaluation summary table
    summary_path = os.path.join(RESULTS_DIR, 'eval_summary.txt')
    table_data = []
    
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # The actual data starts after the 4th line (header) and ends before the last 2 lines
            for line in lines[4:-2]:
                parts = line.split()
                if len(parts) >= 9:
                    # Parts: ['Static', '6.807', '₹', '2,000,000.0', '0.196', '0.804', '₹', '1000.0']
                    # Sometimes the ₹ symbol merges or splits, let's be robust
                    # Let's clean the array to just numbers and words
                    clean_parts = [p for p in parts if p != '₹']
                    if len(clean_parts) >= 6:
                        table_data.append({
                            'agent': clean_parts[0],
                            'reward': clean_parts[1],
                            'revenue': clean_parts[2],
                            'fairness': clean_parts[3],
                            'scalper': str(float(clean_parts[4]) * 100) + "%" if '.' in clean_parts[4] else clean_parts[4],
                            'price': clean_parts[5]
                        })

    # Group images
    images = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.png')]
    dashboards = [img for img in images if 'dashboard' in img]
    heatmaps = [img for img in images if 'heatmap' in img]
    others = [img for img in images if img not in dashboards and img not in heatmaps]

    return render_template('index.html', table_data=table_data, dashboards=dashboards, heatmaps=heatmaps, others=others)

@app.route('/results/<path:filename>')
def results(filename):
    return send_from_directory(RESULTS_DIR, filename)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 DASHBOARD LAUNCHING")
    print("👉 Go to: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
