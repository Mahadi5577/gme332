"""Global constants shared by all modules."""

RANDOM_SEED = 42
DIMS = [2, 3, 3]
CUTS = {"A|BC": (0,), "AB|C": (0, 1), "AC|B": (0, 2)}   # transposed side of each cut
SOLVER = "CLARABEL"
