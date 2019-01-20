from skmultiflow.trees import HoeffdingTree
from skmultiflow.data import RandomTreeGenerator
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential

from hatt import HoeffdingAnytimeTree

stream = RandomTreeGenerator(
    tree_random_state=0,
    sample_random_state=0
)

stream.prepare_for_use()
h = [
    HoeffdingTree(),
    HoeffdingAnytimeTree()
]

evaluator = EvaluatePrequential(
    pretrain_size=100,
    show_plot=True,
    max_samples=100000,
    metrics=['accuracy'],
    batch_size=1
)

evaluator.evaluate(stream=stream, model=h, model_names=['HT', 'HATT'])
