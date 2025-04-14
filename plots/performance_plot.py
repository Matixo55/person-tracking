import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
from plotly.subplots import make_subplots

avg = {'yolov6l': 0.8234, 'yolov7x': 0.8094, 'yolov6m6': 0.8025, 'yolov6l6': 0.8016, 'yolov7': 0.801, 'yolov6m': 0.7905, 'yolov7-e6e': 0.7785, 'yolov7-e6': 0.778, 'yolov6n': 0.775, 'yolov7-d6': 0.7743, 'yolov7-w6': 0.7714, 'yolov6s': 0.7673, 'yolov4-p5_': 0.7652, 'yolov4-p5': 0.7625, 'yolov6s6': 0.7621, 'yolov4-p6_': 0.7407, 'yolov4-p6': 0.7392, 'yolov6n6': 0.7119, 'yolov4-p7': 0.7015, 'yolo11x': 0.6819, 'yolov8x': 0.6816, 'yolov9c': 0.6765, 'yolov8m': 0.6764, 'yolo12x': 0.6738, 'yolov9m': 0.6715, 'yolo11m': 0.6711, 'yolov9e': 0.6705, 'yolo11l': 0.6691, 'yolov8l': 0.6688, 'yolov8s': 0.6626, 'yolo12l': 0.6611, 'yolo11s': 0.6603, 'yolo12m': 0.6589, 'yolov5x6u': 0.6505, 'yolov5m6u': 0.6431, 'yolov5l6u': 0.6419, 'yolov10x': 0.6411, 'yolo12s': 0.6405, 'yolov9s': 0.6395, 'yolov10b': 0.6364, 'yolov10m': 0.6262, 'yolov8n': 0.623, 'yolo11n': 0.6224, 'yolov9t': 0.6131, 'yolo12n': 0.6111, 'yolov5s6u': 0.6015, 'yolov10s': 0.594, 'yolov5n6u': 0.544, 'yolov10n': 0.5412}

def parse_model_centric(file_path, target_attributes):
    with open(file_path, 'r') as f:
        data = f.read()

    lines = data.strip().split("\n")
    all_attributes = [
        "mAP", "FPS", "Total_Time", "Frames_Processed", "Max_Memory", "Avg_Memory",
        "Max_GPU_Memory", "Avg_GPU_Memory", "Max_GPU_Load", "Avg_GPU_Load",
        "start_gpu_memory_used", "start_gpu_load", "start_memory_usage_mb"
    ]

    result = {}

    for line in lines:
        model_part, value_part = line.split(": ")
        model = model_part.strip()

        # Initialize model entry
        result[model] = {}

        map_value, other_values = value_part.split(" ")
        values = [map_value] + other_values.split(",")

        # Only include target attributes
        for attr, value in zip(all_attributes, values):
            if attr in target_attributes:
                try:
                    result[model][attr] = float(value)
                except ValueError:
                    result[model][attr] = value

    return result


target_attrs = ["FPS",  "Avg_GPU_Load", "Avg_GPU_Memory"]
model_data = parse_model_centric("all_results_benchmark.txt", target_attrs)

for key,value in model_data.items():
    model_data[key]["accuracy"] = avg[key]

for key in model_data.keys():
    print(model_data[key], key)

df = pd.DataFrame.from_dict(model_data, orient='index').reset_index()
df.rename(columns={'index': 'model'}, inplace=True)

# Create figure with subplots
fig = make_subplots(
    rows=2, cols=2,
    specs=[
        [{"type": "scene", "rowspan": 2, "colspan": 1}, {"type": "xy"}],
        [None, {"type": "xy"}]
    ],
    subplot_titles=["", "FPS vs. Accuracy", "GPU Load vs. Accuracy"]
)

# Add 3D scatter plot
fig.add_trace(
    go.Scatter3d(
        x=df['FPS'],
        y=df['Avg_GPU_Load'],
        z=df['accuracy'],
        text=df['model'],
        mode='markers+text',
        marker=dict(
            size=12,
            color=df['Avg_GPU_Memory'],
            colorscale='Viridis',
            colorbar=dict(title='GPU Memory'),
            opacity=0.8
        ),
        textposition="top center",
        name="Models"
    ),
    row=1, col=1
)

# Add 2D projections as separate subplots
# FPS vs Accuracy (top right)
fig.add_trace(
    go.Scatter(
        x=df['FPS'],
        y=df['accuracy'],
        mode='markers+text',
        marker=dict(
            size=10,
            color=df['Avg_GPU_Memory'],
            colorscale='Viridis',
            opacity=0.8
        ),
        text=df['model'],
        textposition="top center",
        name="FPS vs Accuracy"
    ),
    row=1, col=2
)

# GPU Load vs Accuracy (bottom right)
fig.add_trace(
    go.Scatter(
        x=df['Avg_GPU_Load'],
        y=df['accuracy'],
        mode='markers+text',
        marker=dict(
            size=10,
            color=df['Avg_GPU_Memory'],
            colorscale='Viridis',
            opacity=0.8
        ),
        text=df['model'],
        textposition="top center",
        name="GPU Load vs Accuracy"
    ),
    row=2, col=2
)

# Update layout
fig.update_layout(
    title='YOLO Models Performance Comparison',
    scene=dict(
        xaxis_title='FPS',
        yaxis_title='GPU Load (%)',
        zaxis_title='Accuracy',
        aspectmode='cube',
        dragmode='orbit',
    ),
    width=1800,
    height=900,
    margin=dict(l=20, r=20, b=20, t=60),
    showlegend=False
)

# Update 2D subplot axes
fig.update_xaxes(title_text="FPS", row=1, col=2)
fig.update_yaxes(title_text="Accuracy", row=1, col=2)
fig.update_xaxes(title_text="GPU Load (%)", row=2, col=2)
fig.update_yaxes(title_text="Accuracy", row=2, col=2)

# Show the figure
fig.show()

# Optional: Save the figure as an HTML file for sharing
pio.write_html(fig, 'yolo_performance_comparison.html')