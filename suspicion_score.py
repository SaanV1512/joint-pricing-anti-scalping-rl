from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# Ignore sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Feature Container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UserBehavior:
    """
    Captures behavioral features of a single user/session during ticket sale.
    """
    tickets_bought: int   = 0      # f1
    speed:          float = 0.0    # f2
    attempts:       int   = 1      # f3
    bought_max:     int   = 0      # f4

    def to_array(self) -> np.ndarray:
        return np.array([self.tickets_bought, self.speed, self.attempts, self.bought_max])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Heuristic Engine (Domain Knowledge Base)
# ─────────────────────────────────────────────────────────────────────────────

class HeuristicSuspicionScorer:
    """
    Computes normalised suspicion score S' ∈ [0, 1] using human domain knowledge.
    
    S  = w1*f1 + w2*f2 + w3*(1/f3) + w4*f4
    S' = S / S_max
    """
    def __init__(self, w1=0.35, w2=0.40, w3=0.10, w4=0.15, S_max=12.0):
        assert abs(w1 + w2 + w3 + w4 - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.weights = np.array([w1, w2, w3, w4])
        self.S_max   = S_max

    def score(self, b: UserBehavior) -> float:
        # Bots usually have fewer attempts because they use optimized scripts, 
        # so we inverse the attempt count (1/f3) for the heuristic penalty.
        f = np.array([b.tickets_bought, b.speed, 1.0 / max(b.attempts, 1), b.bought_max])
        raw_s = np.dot(self.weights, f)
        return min(raw_s / self.S_max, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Machine Learning Anomaly Engine (Isolation Forest)
# ─────────────────────────────────────────────────────────────────────────────

class MLAnomalyScorer:
    """
    Unsupervised ML model that detects bots by isolating anomalous behavior
    in the 4-dimensional feature space. Highly effective against adversarial 
    bots that try to reverse-engineer the heuristic weights.
    """
    def __init__(self, contamination: float = 0.30):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for the ML Anomaly Engine. "
                              "Run: pip install scikit-learn")
        
        # Isolation Forest builds random Decision Trees. Anomalies (bots) have 
        # distinct, extreme patterns and get isolated closer to the root of the tree.
        self.model = IsolationForest(
            n_estimators=100, 
            contamination=contamination, 
            random_state=42
        )
        self.is_fitted = False

    def fit(self, historical_data: List[UserBehavior]):
        """Train the Isolation Forest on historical traffic data."""
        X = np.vstack([b.to_array() for b in historical_data])
        self.model.fit(X)
        self.is_fitted = True

    def score(self, b: UserBehavior) -> float:
        """
        Returns anomaly score mapped to [0, 1].
        1.0 means highly anomalous (Definite Scalper), 0.0 means normal (Genuine).
        """
        if not self.is_fitted:
            return 0.5  # Fallback if not trained
        
        X = b.to_array().reshape(1, -1)
        # score_samples returns negative anomaly score. 
        # Lower means more abnormal. We invert and normalize it.
        raw_score = self.model.score_samples(X)[0] 
        
        # Map roughly from [-0.8, -0.4] -> [0, 1]
        norm_score = (abs(raw_score) - 0.4) / 0.4
        return float(np.clip(norm_score, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hybrid Ensemble System
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleSuspicionSystem:
    """
    Combines Heuristic domain knowledge with Unsupervised ML for robust bot detection.
    """
    def __init__(self, ml_weight: float = 0.6):
        self.heuristic = HeuristicSuspicionScorer()
        self.ml        = MLAnomalyScorer() if SKLEARN_AVAILABLE else None
        self.ml_weight = ml_weight if SKLEARN_AVAILABLE else 0.0

    def warmup(self, dataset: List[UserBehavior]):
        if self.ml:
            self.ml.fit(dataset)

    def compute_s_prime(self, b: UserBehavior) -> Tuple[float, float, float]:
        """Returns: (Ensemble_Score, Heuristic_Score, ML_Score)"""
        h_score = self.heuristic.score(b)
        m_score = self.ml.score(b) if self.ml and self.ml.is_fitted else h_score
        
        final_score = (1 - self.ml_weight) * h_score + self.ml_weight * m_score
        return final_score, h_score, m_score

    def classify(self, b: UserBehavior, threshold: float = 0.40) -> str:
        s_prime, _, _ = self.compute_s_prime(b)
        return "🛑 BOT / SCALPER" if s_prime >= threshold else "✅ GENUINE FAN"

    def aggregate_suspicion(self, behaviors: List[UserBehavior]) -> float:
        """Computes the mean hybrid suspicion score over a batch of users."""
        if not behaviors:
            return 0.0
        return sum(self.compute_s_prime(b)[0] for b in behaviors) / len(behaviors)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation / Demonstration
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*70}")
    print("  🚀 ADVANCED SUSPICION SCORING SYSTEM (Hybrid ML + Heuristic)")
    print(f"{'='*70}\n")
    
    if not SKLEARN_AVAILABLE:
        print("⚠️ Warning: scikit-learn not found. ML Engine disabled.")
        print("To see full power, run: pip install scikit-learn\n")

    # 1. Generate Synthetic Warmup Data (Training the Unsupervised ML)
    #    80% Genuine users, 20% scalper bots
    np.random.seed(42)
    training_data = []
    
    for _ in range(800): # Genuine
        training_data.append(UserBehavior(
            tickets_bought=np.random.randint(1, 3),
            speed=np.random.uniform(0.1, 1.5),
            attempts=np.random.randint(2, 6),
            bought_max=0
        ))
    for _ in range(200): # Scalpers
        training_data.append(UserBehavior(
            tickets_bought=4,
            speed=np.random.uniform(5.0, 15.0),
            attempts=1,
            bought_max=1
        ))

    # 2. Init and Train Ensemble
    system = EnsembleSuspicionSystem(ml_weight=0.7)
    if SKLEARN_AVAILABLE:
        print("⚙️  Training Unsupervised Isolation Forest on 1,000 traffic sessions...")
        system.warmup(training_data)
        print("✅ Model trained successfully!\n")

    # 3. Test Cases
    test_cases = [
        ("Slow Mobile App User", UserBehavior(tickets_bought=2, speed=0.4, attempts=4, bought_max=0)),
        ("Fast Web User",        UserBehavior(tickets_bought=3, speed=2.1, attempts=2, bought_max=0)),
        ("Basic Script Bot",     UserBehavior(tickets_bought=4, speed=8.5, attempts=1, bought_max=1)),
        ("Adversarial Bot",      UserBehavior(tickets_bought=3, speed=4.5, attempts=1, bought_max=1)),
    ]

    print(f"{'User Profile':<22} | {'Heuristic':<10} | {'ML Anomaly':<10} | {'Final S':<8} | {'Classification'}")
    print("-" * 85)

    for name, b in test_cases:
        final_s, h_s, m_s = system.compute_s_prime(b)
        classification = system.classify(b)
        print(f"{name:<22} | {h_s:^10.3f} | {m_s:^10.3f} | {final_s:^8.3f} | {classification}")
        
    print(f"\n{'='*70}\n")
