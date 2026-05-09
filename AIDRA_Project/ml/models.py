# ml/models.py
import csv
import random
import math

# =========================
# SIMPLE KNN CLASSIFIER
# =========================
class SimpleKNN:
    def __init__(self, k=3):
        self.k = k
        self.X = []
        self.y = []

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        preds = []
        for xt in X_test:
            distances = []
            for i, xi in enumerate(self.X):
                # Euclidean distance (can use Manhattan, both fine)
                d = math.sqrt(sum((xt[j] - xi[j])**2 for j in range(len(xt))))
                distances.append((d, self.y[i]))
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            # Majority vote
            votes = {}
            for _, label in k_nearest:
                votes[label] = votes.get(label, 0) + 1
            pred = max(votes, key=votes.get)
            preds.append(pred)
        return preds

# =========================
# SIMPLE NAIVE BAYES (Gaussian)
# =========================
class SimpleNaiveBayes:
    def fit(self, X, y):
        self.classes = list(set(y))
        self.class_priors = {}
        self.mean = {}
        self.var = {}
        for c in self.classes:
            X_c = [X[i] for i in range(len(X)) if y[i] == c]
            self.class_priors[c] = len(X_c) / len(X)
            self.mean[c] = []
            self.var[c] = []
            for j in range(len(X[0])):
                col = [row[j] for row in X_c]
                mean = sum(col) / len(col)
                var = sum((x - mean)**2 for x in col) / len(col)
                self.mean[c].append(mean)
                self.var[c].append(var if var > 0 else 1e-6)

    def gaussian_pdf(self, x, mean, var):
        exponent = math.exp(-((x - mean)**2) / (2 * var))
        return (1 / math.sqrt(2 * math.pi * var)) * exponent

    def predict(self, X_test):
        preds = []
        for xt in X_test:
            probs = {}
            for c in self.classes:
                prior = self.class_priors[c]
                likelihood = 1.0
                for j in range(len(xt)):
                    likelihood *= self.gaussian_pdf(xt[j], self.mean[c][j], self.var[c][j])
                probs[c] = prior * likelihood
            preds.append(max(probs, key=probs.get))
        return preds

# =========================
# SYNTHETIC DATA GENERATION (balanced: 200 each)
# =========================
def generate_balanced_dataset():
    data = []
    # Define feature distributions for each severity
    # Features: [distance_from_base (0-10), hazard_presence (0/1), injury_severity (1-5), heart_rate (60-140)]
    # We'll generate 200 samples per class
    for severity in ['Critical', 'Moderate', 'Minor']:
        for _ in range(200):
            if severity == 'Critical':
                dist = random.uniform(5, 10)
                hazard = random.choice([0,1])  # could be 0 or 1
                injury = random.randint(4,5)
                hr = random.randint(110, 140)
            elif severity == 'Moderate':
                dist = random.uniform(2, 7)
                hazard = random.choice([0,1])
                injury = random.randint(2,4)
                hr = random.randint(85, 115)
            else:  # Minor
                dist = random.uniform(0, 4)
                hazard = 0
                injury = random.randint(1,3)
                hr = random.randint(60, 95)
            data.append([dist, hazard, injury, hr, severity])
    random.shuffle(data)
    return data

# =========================
# TRAIN MODELS AND RETURN BEST ONE
# =========================
def train_models():
    # Generate synthetic dataset
    dataset = generate_balanced_dataset()
    X = [row[:4] for row in dataset]
    y = [row[4] for row in dataset]
    
    # Split into train (80%) and test (20%)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train KNN and NB
    knn = SimpleKNN(k=3)
    nb = SimpleNaiveBayes()
    knn.fit(X_train, y_train)
    nb.fit(X_train, y_train)
    
    # Evaluate accuracy
    knn_pred = knn.predict(X_test)
    nb_pred = nb.predict(X_test)
    knn_acc = sum(1 for i in range(len(y_test)) if knn_pred[i] == y_test[i]) / len(y_test)
    nb_acc = sum(1 for i in range(len(y_test)) if nb_pred[i] == y_test[i]) / len(y_test)
    
    print(f"[ML] SimpleKNN accuracy: {knn_acc:.3f}, SimpleNB accuracy: {nb_acc:.3f}")
    
    # Return the best model (KNN if better or equal)
    best_model = knn if knn_acc >= nb_acc else nb
    return best_model

# =========================
# PREDICT SEVERITY FOR A VICTIM BASED ON LOCATION
# =========================
def predict_severity(model, victim_location, grid):
    x, y = victim_location
    dist = abs(x) + abs(y)
    hazard = 1 if grid[x][y] == 'R' else 0
    # Make Critical more likely for long distances or hazards
    if dist >= 4 or hazard:
        injury_sev = 5
        heart_rate = 130
    elif dist >= 2:
        injury_sev = 3
        heart_rate = 100
    else:
        injury_sev = 1
        heart_rate = 70
    features = [[dist, hazard, injury_sev, heart_rate]]
    return model.predict(features)[0]