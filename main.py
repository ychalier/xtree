from skmultiflow.trees import HoeffdingTree
from skmultiflow.data import RandomTreeGenerator
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential

from hatt import HATT

stream = RandomTreeGenerator(
    tree_random_state=0,
    sample_random_state=0
)

stream.prepare_for_use()
h = [
    HoeffdingTree(),
    HATT()
]

evaluator = EvaluatePrequential(
    pretrain_size=100,
    show_plot=True,
    max_samples=50000,
    metrics=['accuracy','kappa'],
    batch_size=1,
    output_file="output_50000.csv"
)

evaluator.evaluate(stream=stream, model=h, model_names=['HT', 'HATT'])
