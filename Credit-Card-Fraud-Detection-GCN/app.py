

import streamlit as st
import pandas as pd

E = 2.718281828459045
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

    def __init__(self, seed=7):
        self.s = seed

    def uniform(self, lo, hi):
        self.s = (
            1103515245 * self.s + 12345
        ) % 2147483648

        return lo + (hi - lo) * (
            self.s / 2147483648
        )
def build_adjacency(n, edges):

    A = zeros(n, n)

    for u, v in edges:
        A[u][v] = 1.0
        A[v][u] = 1.0

    for i in range(n):
        A[i][i] = 1.0

    return A


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

    def forward(self, A_hat, X):

        self.AX = matmul(A_hat, X)

        self.Z1 = matmul(self.AX, self.W1)

        self.H1 = relu(self.Z1)

        self.AH = matmul(A_hat, self.H1)

        self.Z2 = matmul(self.AH, self.W2)

        self.P = softmax_rows(self.Z2)

        return self.P
    def backward(self, A_hat, Y, train_idx, lr):

        n = len(self.P)
        c = len(self.P[0])

        dZ2 = zeros(n, c)

        for i in train_idx:
            for j in range(c):
                dZ2[i][j] = (
                    self.P[i][j] - Y[i][j]
                ) / len(train_idx)

        dW2 = matmul(
            transpose(self.AH),
            dZ2
        )

        dH1 = matmul(
            A_hat,
            matmul(dZ2, transpose(self.W2))
        )

        dZ1 = [
            [
                dH1[i][j]
                if self.Z1[i][j] > 0
                else 0.0
                for j in range(len(dH1[0]))
            ]
            for i in range(n)
        ]

        dW1 = matmul(
            transpose(self.AX),
            dZ1
        )

        for i in range(len(self.W1)):
            for j in range(len(self.W1[0])):
                self.W1[i][j] -= lr * dW1[i][j]

        # Update second weight matrix
        for i in range(len(self.W2)):
            for j in range(len(self.W2[0])):
                self.W2[i][j] -= lr * dW2[i][j]
names = [
    "T001", "T002", "T003", "T004",
    "T005", "T006", "T007", "T008",
    "T009", "T010", "T011", "T012",
    "T013", "T014"
]
X = [

   
    [0.10, 0.10, 0.10],
    [0.12, 0.12, 0.15],
    [0.15, 0.15, 0.10],
    [0.11, 0.18, 0.12],
    [0.18, 0.20, 0.15],
    [0.14, 0.22, 0.10],
    [0.20, 0.25, 0.18],
    [0.16, 0.28, 0.12],

 
    [0.90, 0.80, 0.90],
    [0.92, 0.82, 0.95],
    [0.88, 0.85, 0.90],
    [0.95, 0.87, 0.92],
    [0.91, 0.90, 0.95],
    [0.89, 0.92, 0.88]
]


edges = [

    (0, 1), (0, 2), (1, 3),
    (2, 3), (3, 4), (4, 5),
    (5, 6), (6, 7), (1, 6),

    (8, 9), (8, 10), (9, 11),
    (10, 11), (11, 12), (12, 13),
    (9, 13), (8, 12),
    (7, 13), (4, 10)
]
true_labels = [0] * 8 + [1] * 6
known = {
    0: 0,
    1: 0,
    8: 1,
    9: 1
}
def train_model(epochs, learning_rate):

    n = len(X)

    A = build_adjacency(n, edges)
    A_hat = normalize(A)

    train_idx = sorted(known.keys())

    Y = zeros(n, 2)

    for i, lab in known.items():
        Y[i][lab] = 1.0

    model = GCN(
        in_dim=3,
        hidden=4,
        out_dim=2
    )

    history = {
        "epoch": [],
        "confidence": [],
        "accuracy": []
    }

    test_idx = [
        i for i in range(n)
        if i not in known
    ]

    for epoch in range(1, epochs + 1):

        P = model.forward(A_hat, X)

        mean_conf = sum(
            P[i][known[i]]
            for i in train_idx
        ) / len(train_idx)

        correct = sum(
            1 for i in test_idx
            if argmax(P[i]) == true_labels[i]
        )

        accuracy = correct / len(test_idx)

        if epoch == 1 or epoch % 10 == 0:
            history["epoch"].append(epoch)
            history["confidence"].append(mean_conf)
            history["accuracy"].append(accuracy * 100)

        model.backward(
            A_hat,
            Y,
            train_idx,
            learning_rate
        )

    predictions = model.forward(A_hat, X)

    return predictions, history, test_idx
def calculate_metrics(predicted, actual):

    tp = fp = tn = fn = 0

    for p, a in zip(predicted, actual):

        if p == 1 and a == 1:
            tp += 1

        elif p == 1 and a == 0:
            fp += 1

        elif p == 0 and a == 0:
            tn += 1

        else:
            fn += 1

    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(0.000001, precision + recall)

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1 Score": f1 * 100
    }

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")
st.subheader("Vanilla Graph Convolutional Network (GCN)")

st.write(
    "An interactive dashboard for detecting fraudulent "
    "transactions using a manually implemented GCN."
)

st.info(
    "Demonstration note: This version uses 14 illustrative "
    "transactions and manually defined graph connections. "
    "It is not yet trained on the full Kaggle dataset."
)

st.sidebar.header("Model Settings")

epochs = st.sidebar.slider(
    "Training Epochs",
    min_value=100,
    max_value=2000,
    value=1000,
    step=100
)

learning_rate = st.sidebar.slider(
    "Learning Rate",
    min_value=0.001,
    max_value=0.2,
    value=0.05,
    step=0.001
)

threshold = st.sidebar.slider(
    "Fraud Detection Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

st.sidebar.write("Adjust the settings and train again.")

train_button = st.sidebar.button(
    "🚀 Train GCN Model",
    use_container_width=True
)
if "gcn_results" not in st.session_state or train_button:

    with st.spinner("Training Vanilla GCN..."):

        P, history, test_idx = train_model(
            epochs,
            learning_rate
        )

        st.session_state.gcn_results = {
            "P": P,
            "history": history,
            "test_idx": test_idx,
            "epochs": epochs,
            "learning_rate": learning_rate
        }

results = st.session_state.gcn_results

P = results["P"]
history = results["history"]
test_idx = results["test_idx"]
predicted_labels = [
    1 if P[i][1] >= threshold else 0
    for i in range(len(names))
]

actual_test = [
    true_labels[i] for i in test_idx
]

predicted_test = [
    predicted_labels[i] for i in test_idx
]

metrics = calculate_metrics(
    predicted_test,
    actual_test
)

fraud_count = sum(predicted_labels)
legitimate_count = len(names) - fraud_count
st.divider()
st.header("📊 Transaction Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", len(names))
col2.metric("Predicted Fraud", fraud_count)
col3.metric("Predicted Legitimate", legitimate_count)
col4.metric("Known Training Labels", len(known))
st.divider()
st.header("Transaction Distribution")

distribution_data = [
    {"Type": "Legitimate", "Count": legitimate_count},
    {"Type": "Fraud", "Count": fraud_count}
]

st.vega_lite_chart(
    {
        "data": {"values": distribution_data},
        "mark": {"type": "bar"},
        "encoding": {
            "x": {
                "field": "Type",
                "type": "nominal",
                "title": "Predicted Transaction Type"
            },
            "y": {
                "field": "Count",
                "type": "quantitative",
                "title": "Number of Transactions"
            },
            "tooltip": ["Type", "Count"]
        }
    },
    use_container_width=True
)
st.divider()
st.header("Fraud Probability by Transaction")

probability_data = []

for i in range(len(names)):

    probability_data.append({
        "Transaction": names[i],
        "Fraud Probability": round(P[i][1], 4)
    })

st.vega_lite_chart(
    {
        "data": {"values": probability_data},
        "mark": {"type": "bar"},
        "encoding": {
            "x": {
                "field": "Transaction",
                "type": "ordinal",
                "title": "Transaction ID",
                "sort": names
            },
            "y": {
                "field": "Fraud Probability",
                "type": "quantitative",
                "scale": {"domain": [0, 1]},
                "title": "Probability of Fraud"
            },
            "tooltip": ["Transaction", "Fraud Probability"]
        }
    },
    use_container_width=True
)

st.caption(
    "A probability above the selected threshold is classified as fraud."
)
st.divider()
st.header("📈 GCN Training Performance")

history_data = []

for i in range(len(history["epoch"])):

    history_data.append({
        "Epoch": history["epoch"][i],
        "Training Confidence": history["confidence"][i],
        "Unlabeled Accuracy (%)": history["accuracy"][i]
    })

st.vega_lite_chart(
    {
        "data": {"values": history_data},
        "mark": {
            "type": "line",
            "point": True
        },
        "encoding": {
            "x": {
                "field": "Epoch",
                "type": "quantitative",
                "title": "Training Epoch"
            },
            "y": {
                "field": "Training Confidence",
                "type": "quantitative",
                "title": "Training Confidence",
                "scale": {"domain": [0, 1]}
            },
            "tooltip": [
                "Epoch",
                "Training Confidence",
                "Unlabeled Accuracy (%)"
            ]
        }
    },
    use_container_width=True
)

st.caption(
    "Training confidence measures the average probability "
    "assigned to the correct class for the four labeled transactions."
)
st.divider()
st.header("🔍 Transaction Prediction Results")

rows = []

for i in range(len(names)):

    predicted = (
        "FRAUD"
        if predicted_labels[i] == 1
        else "LEGITIMATE"
    )

    actual = (
        "FRAUD"
        if true_labels[i] == 1
        else "LEGITIMATE"
    )

    rows.append({
        "Transaction": names[i],
        "Fraud Probability": round(P[i][1], 4),
        "Predicted": predicted,
        "Actual": actual,
        "Training Label": "Yes" if i in known else "No"
    })

st.dataframe(
    rows,
    use_container_width=True,
    hide_index=True
)
st.divider()
st.header("📋 Model Evaluation")

st.caption(
    "Metrics below are calculated only on the 10 transactions "
    "whose labels were hidden during training. Their labels are "
    "used only for demonstration-time evaluation."
)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Accuracy", f"{metrics['Accuracy']:.2f}%")
m2.metric("Precision", f"{metrics['Precision']:.2f}%")
m3.metric("Recall", f"{metrics['Recall']:.2f}%")
m4.metric("F1 Score", f"{metrics['F1 Score']:.2f}%")

st.subheader("Confusion Matrix")

confusion_data = [
    {
        "Actual": "Legitimate",
        "Predicted Legitimate": metrics["TN"],
        "Predicted Fraud": metrics["FP"]
    },
    {
        "Actual": "Fraud",
        "Predicted Legitimate": metrics["FN"],
        "Predicted Fraud": metrics["TP"]
    }
]

st.dataframe(
    confusion_data,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "TP = correctly detected fraud; TN = correctly identified "
    "legitimate transactions; FP = legitimate transactions "
    "incorrectly flagged as fraud; FN = missed fraud transactions."
)

st.divider()

with st.expander("About this project"):

    st.write("""
    **Vanilla Graph Convolutional Network**

    The model uses two graph convolution layers.

    Layer 1:
    - Aggregate neighboring transaction features.
    - Multiply by trainable weights.
    - Apply ReLU activation.

    Layer 2:
    - Aggregate neighboring hidden representations.
    - Multiply by output weights.
    - Apply Softmax to predict transaction classes.

    The model is trained using four known transaction labels.
    The other ten labels are hidden during training and used
    only to evaluate this illustrative example.

    This is a small synthetic demonstration, not a production
    fraud detection system.
    """)

st.success("GCN dashboard is ready!")
