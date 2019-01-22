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

for i in range(0,10000,1000):
	evaluator = EvaluatePrequential(
	    pretrain_size=i,
	    show_plot=False,
	    max_samples=20000,
	    metrics=['accuracy','kappa'],
	    batch_size=1,
      output_file="results/output_pretrain_"+ str(i)+"_batch_size_" + "1" + ".csv"

	)

	evaluator.evaluate(stream=stream, model=h, model_names=['HT', 'HATT'])
