import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

model_groups = {
    "yolov4": ["yolov4-p5", "yolov4-p5_", "yolov4-p6", "yolov4-p6_", "yolov4-p7"],
    "yolov5": ["yolov5n6u", "yolov5s6u", "yolov5m6u", "yolov5l6u", "yolov5x6u"],
    "yolov6": ["yolov6n", "yolov6s", "yolov6m", "yolov6l", "yolov6n6", "yolov6s6", "yolov6m6", "yolov6l6"],
    "yolov7": ["yolov7", "yolov7x", "yolov7-w6", "yolov7-e6", "yolov7-d6", "yolov7-e6e"],
    "yolov8": ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
    "yolov9": ["yolov9t", "yolov9s", "yolov9m", "yolov9c", "yolov9e"],
    "yolov10": ["yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10x"],
    "yolo11": ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
    "yolo12": ["yolo12n", "yolo12s", "yolo12m", "yolo12l", "yolo12x"],
}
version_colors = {
    "yolov4": "red",
    "yolov5": "lightblue",
    "yolov6": "lightgreen",
    "yolov7": "salmon",
    "yolov8": "gold",
    "yolov9": "mediumpurple",
    "yolov10": "lightcoral",
    "yolo11": "mediumaquamarine",
    "yolo12": "plum",
}
def parse_results(file_path):
    results = defaultdict(dict)

    with open(file_path, 'r') as f:
        content = f.read()

    # Split by double newlines or video ID patterns
    blocks = content.split('\n\n')
    if len(blocks) == 1:  # If no double newlines, try splitting by ID pattern
        import re
        blocks = re.split(r'(?=^[^\s].*:$)', content, flags=re.MULTILINE)
        blocks = [b for b in blocks if b.strip()]

    for block in blocks:
        lines = block.strip().split('\n')

        # First line should be the ID
        id_line = lines[0].strip()
        current_id = id_line.rstrip(':')

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            parts = line.split(':', 1)  # Split on first colon only
            if len(parts) == 2:
                model = parts[0].strip()
                score = float(parts[1].strip())
                results[current_id][model] = score

    return results


def create_model_rankings(results_dict):
    """
    Create a dictionary that contains the ranking of each model for each ID
    """
    rankings = {}

    for video_id, models in results_dict.items():
        # Sort models by score in descending order
        sorted_models = sorted(models.items(), key=lambda x: x[1], reverse=True)

        # Create a dictionary with model:rank pairs
        rankings[video_id] = {}
        for rank, (model, _) in enumerate(sorted_models, 1):
            rankings[video_id][model] = rank

    return rankings


def calculate_average_rankings(rankings):
    """
    Calculate the average ranking for each model
    """
    model_centric = {}
    for video_id, model_ranks in rankings.items():
        for model, rank in model_ranks.items():
            if model not in model_centric:
                model_centric[model] = []
            model_centric[model].append(rank)

    # Calculate average ranking for each model
    avg_rankings = {}
    for model, ranks in model_centric.items():
        avg_rankings[model] = {
            'avg': np.mean(ranks),
            'count': len(ranks),
            # 'std': np.std(ranks)
        }

    return avg_rankings, model_centric


def calculate_average_scores(results_dict):
    model_scores = defaultdict(list)
    for video_id, models in results_dict.items():
        for model, score in models.items():
            model_scores[model].append(score)

    avg_scores = {}
    for model, scores in model_scores.items():
        avg_scores[model] = {
            'avg': np.mean(scores),
            'count': len(scores),
            'std': np.std(scores)
        }

    return avg_scores, model_scores


def plot_average_rankings(avg_rankings, model_centric):
    """
    Create plots for the average rankings without showing deviation range
    """
    # Sort models by average ranking (ascending, lower is better)
    sorted_models = sorted(avg_rankings.items(), key=lambda x: x[1]['avg'])
    # Prepare data for plotting
    models = [model for model, _ in sorted_models]
    avgs = [data['avg'] for _, data in sorted_models]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot: Bar chart with average rankings (no error bars)
    x = np.arange(len(models))
    bars = ax.bar(x, avgs, alpha=0.7)

    # Add actual average scores as text on bars
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{avgs[i]:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Models')
    ax.set_ylabel('Average Ranking (lower is better)')
    ax.set_title('Average Ranking Performance by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 43)  # Set y-axis limit to 40

    plt.tight_layout()
    plt.savefig('model_rankings.png', dpi=300)
    plt.show()


def plot_models_by_version(avg_rankings, model_centric):
    """
    Create plot for the average rankings with models ordered by version and size,
    without showing deviation range
    """

    # Create a flat ordered list of all models
    ordered_models = []
    for version, models in model_groups.items():
        ordered_models.extend(models[::-1])
    print(ordered_models)
    # Filter and prepare data for plotting (only include models that are in avg_rankings)
    filtered_models = []
    avgs = []
    stds = []  # still collect for reference but won't display
    model_versions = []  # To track which version each model belongs to

    for model in ordered_models:
        # Try exact match first
        if model in avg_rankings:
            filtered_models.append(model)
            avgs.append(avg_rankings[model]['avg'])
            # stds.append(avg_rankings[model]['std'])

            # Find which version this model belongs to
            for version, models_list in model_groups.items():
                if model in models_list:
                    model_versions.append(version)
                    break
        # Some models might have different formats, try partial matching
        else:
            matching_models = [m for m in avg_rankings.keys() if model in m or m in model]
            if matching_models:
                # Use the first match
                match = matching_models[0]
                filtered_models.append(match)
                avgs.append(avg_rankings[match]['avg'])
                # stds.append(avg_rankings[match]['std'])

                # Find which version this model belongs to
                for version, models_list in model_groups.items():
                    if any(m in match or match in m for m in models_list):
                        model_versions.append(version)
                        break

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))

    # Set the width of each bar
    bar_width = 0.8

    # Create group boundaries and calculate positions
    group_boundaries = []
    curr_pos = 0
    x_positions = []
    version_midpoints = {}

    prev_version = None
    for i, version in enumerate(model_versions):
        # If we're starting a new version group
        if version != prev_version:
            if prev_version is not None:
                group_boundaries.append(curr_pos)
            version_midpoints[version] = curr_pos
            prev_version = version

        x_positions.append(curr_pos)
        curr_pos += 1

        # Update the midpoint (will be overwritten multiple times, but that's ok)
        version_midpoints[version] = (version_midpoints[version] + curr_pos - 1) / 2

    # Get colors for each bar
    bar_colors = [version_colors.get(v, "gray") for v in model_versions]

    # Plot bars at calculated positions (no error bars)
    bars = ax.bar(x_positions, avgs, width=bar_width, alpha=0.8, color=bar_colors)

    # Add actual average scores as text on bars
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{avgs[i]:.1f}', ha='center', va='bottom', fontsize=8)

    # Add version separators
    for boundary in group_boundaries:
        ax.axvline(x=boundary - 0.5, color='black', linestyle='--', alpha=0.5)

    ax.set_ylabel('Average Ranking (lower is better)', fontsize=12)
    ax.set_title('Average Ranking Performance by Model Version and Size', fontsize=14)

    # Set the tick positions and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(filtered_models, rotation=45, ha='right', fontsize=10)

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 43)  # Set y-axis limit to 40

    # Add some extra space at the bottom for version labels
    plt.subplots_adjust(bottom=0.2)

    plt.tight_layout()
    plt.savefig('model_rankings_by_version.png', dpi=300)
    plt.show()


def plot_average_scores(avg_scores, model_centric):
    """
    Create plots for the actual average scores without showing deviation range
    """
    # Sort models by average score (descending, higher is better)
    sorted_models = sorted(avg_scores.items(), key=lambda x: x[1]['avg'], reverse=True)

    models = [model for model, _ in sorted_models]
    avgs = [data['avg'] for _, data in sorted_models]
    print({model: round(float(avg),4) for model, avg in zip(models, avgs)})
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot: Bar chart with average scores (no error bars)
    x = np.arange(len(models))
    ax.bar(x, avgs, alpha=0.7)

    # Add actual average scores as text on bars
    # for i, bar in enumerate(bars):
    #     ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
    #             f'{avgs[i]}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Models')
    ax.set_ylabel('Average Score')
    ax.set_title('Average Score Performance by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('model_scores.png', dpi=300)
    plt.show()


def plot_models_by_score(avg_scores, model_centric):
    """
    Create plot for the actual average scores with models ordered by version and size,
    without showing deviation range
    """

    # Create a flat ordered list of all models
    ordered_models = []
    for version, models in model_groups.items():
        ordered_models.extend(models[::-1])

    # Filter and prepare data for plotting (only include models that are in avg_rankings)
    filtered_models = []
    avgs = []
    model_versions = []  # To track which version each model belongs to

    for model in ordered_models:
        filtered_models.append(model)
        avgs.append(avg_scores[model]['avg'])

        # Find which version this model belongs to
        for version, models_list in model_groups.items():
            if model in models_list:
                model_versions.append(version)
                break

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))

    # Use different colors for different model versions


    # Set the width of each bar
    bar_width = 0.8

    # Create group boundaries and calculate positions
    group_boundaries = []
    curr_pos = 0
    x_positions = []
    version_midpoints = {}

    prev_version = None
    for i, version in enumerate(model_versions):
        # If we're starting a new version group
        if version != prev_version:
            if prev_version is not None:
                group_boundaries.append(curr_pos)
            version_midpoints[version] = curr_pos
            prev_version = version

        x_positions.append(curr_pos)
        curr_pos += 1

        # Update the midpoint (will be overwritten multiple times, but that's ok)
        version_midpoints[version] = (version_midpoints[version] + curr_pos - 1) / 2

    # Get colors for each bar
    bar_colors = [version_colors.get(v, "gray") for v in model_versions]

    ax.bar(x_positions, avgs, width=bar_width, alpha=0.8, color=bar_colors)

    # Add actual average scores as text on bars
    # for i, bar in enumerate(bars):
    #     ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
    #             f'{avgs[i]}', ha='center', va='bottom', fontsize=8)

    # Add version separators
    for boundary in group_boundaries:
        ax.axvline(x=boundary - 0.5, color='black', linestyle='--', alpha=0.5)

    ax.set_ylabel('Average Score', fontsize=12)
    ax.set_title('Average Score Performance by Model Version and Size', fontsize=14)

    # Set the tick positions and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(filtered_models, rotation=45, ha='right', fontsize=10)

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add some extra space at the bottom for version labels
    plt.subplots_adjust(bottom=0.2)

    plt.tight_layout()
    plt.savefig('model_scores_by_version.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # file_path = "all_results_DETRAC.txt"
    file_path = "all_results_personpath22.txt"

    # Parse results
    parsed_results = parse_results(file_path)

    # Create rankings
    rankings = create_model_rankings(parsed_results)

    # Calculate average rankings
    avg_rankings, model_centric = calculate_average_rankings(rankings)

    avg_scores, model_centric_scores = calculate_average_scores(parsed_results)

    # Plot the results
    plot_average_rankings(avg_rankings, model_centric)
    plot_average_scores(avg_scores, model_centric_scores)

    plot_models_by_version(avg_rankings, model_centric)
    plot_models_by_score(avg_scores, model_centric_scores)