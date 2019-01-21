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
	HATT(),
]

for i in range(1,100,10):
	evaluator = EvaluatePrequential(
	    pretrain_size=0,
	    show_plot=True,
	    max_samples=20000,
	    metrics=['accuracy','kappa'],
	    batch_size=i,
        output_file="results/output_pretrain_0_batch_size_" + str(i) + ".csv"
	)

	evaluator.evaluate(stream=stream, model=h, model_names=['HT', 'HATT'])
