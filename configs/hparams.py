DATASET_DEFAULTS = {
    "ogbn-arxiv": dict(
        num_layers=3,
        hidden_channels=256,
        heads=4,
        dropout=0.5,
        lr=0.001,
        epochs=1000,
        sampler="saint",
        batch_size=10000,
        num_neighbors=None,
        saint_walk_length=2,
        saint_num_steps=30,
    ),
    "ogbn-products": dict(
        num_layers=3,
        hidden_channels=128,
        heads=4,
        dropout=0.5,
        lr=0.01,
        epochs=100,
        sampler="neighbor",
        batch_size=1024,
        num_neighbors=[5, 5, 5],
    ),
    "ogbn-mag": dict(
        num_layers=2,
        hidden_channels=256,
        heads=4,
        dropout=0.5,
        lr=0.01,
        epochs=100,
        sampler="neighbor",
        batch_size=512,
        num_neighbors=[5, 5],
    ),
    "ogbn-proteins": dict(
        num_layers=6,
        hidden_channels=64,
        heads=4,
        dropout=0.5,
        lr=0.001,
        epochs=1000,
        sampler="neighbor",
        batch_size=1024,
        num_neighbors=[5, 5, 5, 5, 5, 5],
    ),
    "ogbl-collab": dict(
        num_layers=3,
        hidden_channels=64,
        heads=4,
        dropout=0.5,
        lr=0.001,
        epochs=1000,
        sampler="full",
        batch_size=None,
        num_neighbors=None,
    ),
    "ogbl-citation2": dict(
        num_layers=3,
        hidden_channels=256,
        heads=4,
        dropout=0.5,
        lr=0.01,
        epochs=1000,
        sampler="neighbor",
        batch_size=512,
        num_neighbors=[5, 5, 5],
    ),
}

# ogbn-proteins: "all" trains jointly on all 112 protein-function labels.
# "0" through "111" selects a single binary subtask for same-task DD comparisons.
# All other datasets have exactly one task.
AVAILABLE_TASKS = {
    "ogbn-arxiv":     ["node_classification"],
    "ogbn-products":  ["node_classification"],
    "ogbn-mag":       ["node_classification"],
    "ogbn-proteins":  ["all"] + [str(i) for i in range(112)],
    "ogbl-collab":    ["link_prediction"],
    "ogbl-citation2": ["link_prediction"],
}

DEFAULT_TASK = {k: v[0] for k, v in AVAILABLE_TASKS.items()}

# Drives loss function and evaluator metric choice
TASK_TYPE = {
    "ogbn-arxiv":     "node_clf",
    "ogbn-products":  "node_clf",
    "ogbn-mag":       "node_clf",
    "ogbn-proteins":  "node_clf",
    "ogbl-collab":    "link_pred",
    "ogbl-citation2": "link_pred",
}

# Primary OGB evaluator metric key per dataset
METRIC_KEY = {
    "ogbn-arxiv":     "acc",
    "ogbn-products":  "acc",
    "ogbn-mag":       "acc",
    "ogbn-proteins":  "rocauc",
    "ogbl-collab":    "hits@50",
    "ogbl-citation2": "mrr_list",  # requires .mean() after eval()
}

# Number of output classes for node classification (proteins handled separately)
NUM_CLASSES = {
    "ogbn-arxiv":    40,
    "ogbn-products": 47,
    "ogbn-mag":      349,
    "ogbn-proteins": 112,
}
