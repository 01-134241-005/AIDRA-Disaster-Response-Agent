import csv
import random

def create_grid():
    # S = start, X = blocked, R = hazard, H = hospital
    grid = [
        ['S', '.', '.', 'R', '.'],
        ['.', 'X', '.', 'R', '.'],
        ['.', '.', '.', '.', '.'],
        ['R', '.', '.', 'X', '.'],
        ['.', '.', '.', '.', 'H']
    ]
    return grid


def get_victims():
    grid = create_grid()
    victims = []

    # Required victim counts
    required_counts = {
        "Critical": 2,
        "Moderate": 2,
        "Minor": 1
    }

    current_counts = {
        "Critical": 0,
        "Moderate": 0,
        "Minor": 0
    }

    used_positions = set()

    file_path = r"C:/Users/Administrator/Desktop/AIDRA_Project/dataset/synthetic_medical_triage.csv"

    with open(file_path, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:

            severity = row["triage"]

            # Stop when enough victims collected
            if current_counts == required_counts:
                break

            # Skip extra victims
            if severity not in required_counts:
                continue

            if current_counts[severity] >= required_counts[severity]:
                continue

            # Generate valid random grid position
            while True:
                x = random.randint(0, 4)
                y = random.randint(0, 4)

                # Avoid blocked/start/hospital/repeated cells
                if grid[x][y] not in ['X', 'S', 'H'] and (x, y) not in used_positions:
                    used_positions.add((x, y))
                    break

            victims.append(((x, y), severity))

            current_counts[severity] += 1

    return victims