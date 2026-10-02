
"""
Credit Card Fraud Detection using Vanilla GCN
Pure Python: no imports, no NumPy, no ML library.
Everything implemented manually.
"""

E = 2.718281828459045


# ---------------------------------------------------------------
# 1. Small helper functions (our own matrix library)
# ---------------------------------------------------------------

def zeros(r, c):
    return [[0.0] * c for _ in range(r)]


def transpose(M):
    return [
        [M[i][j] for i in range(len(M))]
        for j in range(len(M[0]))
    ]


def matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    C = zeros(n, p)

    for i in range(n):
        for k in range(m):
            a = A[i][k]

            if a != 0.0:
                for j in range(p):
                    C[i][j] += a * B[k][j]

    return C


def relu(M):
    return [
        [x if x > 0 else 0.0 for x in row]
        for row in M
    ]


def softmax_rows(M):
    out = []

    for row in M:
        mx = max(row)
        ex = [E ** (x - mx) for x in row]
        s = sum(ex)

        out.append([x / s for x in ex])

    return out


def argmax(row):
    best = 0

    for i in range(1, len(row)):
        if row[i] > row[best]:
            best = i

    return best


class RNG:
    """Simple random generator without importing random."""

    def __init__(self, seed=7):
        self.s = seed

    def uniform(self, lo, hi):
        self.s = (
            1103515245 * self.s + 12345
        ) % 2147483648

        return lo + (hi - lo) * (
            self.s / 2147483648
        )


# ---------------------------------------------------------------
# 2. Build adjacency matrix
# ---------------------------------------------------------------

def build_adjacency(n, edges):

    A = zeros(n, n)

    # Create connections
    for u, v in edges:
        A[u][v] = 1.0
        A[v][u] = 1.0

    # Add self-loops
    for i in range(n):
        A[i][i] = 1.0

    return A


# ---------------------------------------------------------------
# 3. Normalize adjacency matrix
# A_hat = D^-1/2 (A + I) D^-1/2
# ---------------------------------------------------------------

def normalize(A):

    n = len(A)

    deg = [sum(row) for row in A]

    A_hat = zeros(n, n)

    for i in range(n):
        for j in range(n):

            if A[i][j] != 0.0:

                A_hat[i][j] = A[i][j] / (
                    (deg[i] ** 0.5) *
                    (deg[j] ** 0.5)
                )

    return A_hat


# ---------------------------------------------------------------
# 4. Vanilla GCN Model
# ---------------------------------------------------------------

class GCN:

    def __init__(self, in_dim, hidden, out_dim):

        rng = RNG()

        self.W1 = [
            [rng.uniform(-0.8, 0.8) for _ in range(hidden)]
            for _ in range(in_dim)
        ]

        self.W2 = [
            [rng.uniform(-0.8, 0.8) for _ in range(out_dim)]
            for _ in range(hidden)
        ]

    # Forward propagation
    def forward(self, A_hat, X):

        # First graph convolution
        self.AX = matmul(A_hat, X)

        self.Z1 = matmul(self.AX, self.W1)

        self.H1 = relu(self.Z1)

        # Second graph convolution
        self.AH = matmul(A_hat, self.H1)

        self.Z2 = matmul(self.AH, self.W2)

        # Fraud / legitimate probabilities
        self.P = softmax_rows(self.Z2)

        return self.P

    # Backpropagation
    def backward(self, A_hat, Y, train_idx, lr):

        n = len(self.P)
        c = len(self.P[0])

        dZ2 = zeros(n, c)

        # Calculate output gradient
        for i in train_idx:

            for j in range(c):

                dZ2[i][j] = (
                    self.P[i][j] - Y[i][j]
                ) / len(train_idx)

        # Gradient of second weight matrix
        dW2 = matmul(
            transpose(self.AH),
            dZ2
        )

        # Gradient passed to first layer
        dH1 = matmul(
            A_hat,
            matmul(dZ2, transpose(self.W2))
        )

        # ReLU derivative
        dZ1 = [
            [
                dH1[i][j]
                if self.Z1[i][j] > 0
                else 0.0
                for j in range(len(dH1[0]))
            ]
            for i in range(n)
        ]

        # Gradient of first weight matrix
        dW1 = matmul(
            transpose(self.AX),
            dZ1
        )

        # Update W1
        for i in range(len(self.W1)):
            for j in range(len(self.W1[0])):

                self.W1[i][j] -= lr * dW1[i][j]

        # Update W2
        for i in range(len(self.W2)):
            for j in range(len(self.W2[0])):

                self.W2[i][j] -= lr * dW2[i][j]


# ---------------------------------------------------------------
# 5. INPUT TRANSACTION DATA
# ---------------------------------------------------------------

names = [
    "T001", "T002", "T003", "T004",
    "T005", "T006", "T007", "T008",
    "T009", "T010", "T011", "T012",
    "T013", "T014"
]

# Features:
# [transaction_amount, transaction_time, risk_indicator]
#
# All features are scaled between 0 and 1.
#
# Higher risk_indicator represents an unusual
# transaction pattern in this illustrative dataset.

X = [

    # Legitimate transactions
    [0.10, 0.10, 0.10],
    [0.12, 0.12, 0.15],
    [0.15, 0.15, 0.10],
    [0.11, 0.18, 0.12],
    [0.18, 0.20, 0.15],
    [0.14, 0.22, 0.10],
    [0.20, 0.25, 0.18],
    [0.16, 0.28, 0.12],

    # Fraudulent transactions
    [0.90, 0.80, 0.90],
    [0.92, 0.82, 0.95],
    [0.88, 0.85, 0.90],
    [0.95, 0.87, 0.92],
    [0.91, 0.90, 0.95],
    [0.89, 0.92, 0.88]
]


# ---------------------------------------------------------------
# 6. TRANSACTION CONNECTIONS
# ---------------------------------------------------------------

# Transactions with similar characteristics are connected.

edges = [

    # Legitimate transaction group
    (0, 1), (0, 2), (1, 3),
    (2, 3), (3, 4), (4, 5),
    (5, 6), (6, 7), (1, 6),

    # Fraud transaction group
    (8, 9), (8, 10), (9, 11),
    (10, 11), (11, 12), (12, 13),
    (9, 13), (8, 12),

    # Transactions with different labels can also connect
    (7, 13), (4, 10)
]


# Actual labels:
# 0 = Legitimate
# 1 = Fraud

true_labels = [0] * 8 + [1] * 6


# Only a few transactions have verified labels.
# Other labels are hidden during training.

known = {
    0: 0,
    1: 0,
    8: 1,
    9: 1
}


# ---------------------------------------------------------------
# 7. TRAIN THE GCN
# ---------------------------------------------------------------

def main():

    n = len(X)

    # Build and normalize graph
    A = build_adjacency(n, edges)

    A_hat = normalize(A)

    # Prepare training labels
    train_idx = sorted(known.keys())

    Y = zeros(n, 2)

    for i, lab in known.items():
        Y[i][lab] = 1.0

    # Initialize GCN
    model = GCN(
        in_dim=3,
        hidden=4,
        out_dim=2
    )

    print("CREDIT CARD FRAUD DETECTION USING GCN")
    print("-" * 65)

    print("Total transactions:", n)
    print("Connections:", len(edges))
    print("Features per transaction:", len(X[0]))
    print("Known labeled transactions:", len(known))

    print("-" * 65)

    # Training
    epochs = 300
    learning_rate = 0.5

    for epoch in range(1, epochs + 1):

        # Forward propagation
        P = model.forward(A_hat, X)

        # Average confidence on known labels
        mean_conf = sum(
            P[i][known[i]]
            for i in train_idx
        ) / len(train_idx)

        # Backpropagation
        model.backward(
            A_hat,
            Y,
            train_idx,
            learning_rate
        )

        if epoch == 1 or epoch % 50 == 0:

            test = [
                i for i in range(n)
                if i not in known
            ]

            correct = sum(
                1 for i in test
                if argmax(P[i]) == true_labels[i]
            )

            accuracy = correct / len(test)

            print(
                "Epoch:", epoch,
                "| Confidence:", round(mean_conf, 3),
                "| Unlabeled accuracy:", round(accuracy * 100, 2), "%"
            )

    # -----------------------------------------------------------
    # 8. FINAL PREDICTIONS
    # -----------------------------------------------------------

    P = model.forward(A_hat, X)

    print("-" * 65)

    print(
        f"{'Transaction':<15}"
        f"{'P(Fraud)':>12}"
        f"{'Predicted':>15}"
        f"{'Actual':>12}"
    )

    print("-" * 65)

    for i in range(n):

        predicted = (
            "FRAUD"
            if argmax(P[i]) == 1
            else "LEGITIMATE"
        )

        actual = (
            "FRAUD"
            if true_labels[i] == 1
            else "LEGITIMATE"
        )

        tag = "(labeled)" if i in known else ""

        print(
            f"{names[i]:<15}"
            f"{P[i][1]:>12.3f}"
            f"{predicted:>15}"
            f"{actual:>12}  {tag}"
        )

    print("-" * 65)


# ---------------------------------------------------------------
# 9. EXECUTION
# ---------------------------------------------------------------

if __name__ == "__main__":
    main()
