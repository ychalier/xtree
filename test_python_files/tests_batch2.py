from skmultiflow.trees import HoeffdingTree
from skmultiflow.data import RandomTreeGenerator
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.data import WaveformGenerator

from hatt import HATT

stream = WaveformGenerator()
stream.prepare_for_use()

h = [
    HoeffdingTree(),
	HATT(),
]

print(stream)

for i in range(0,1):
	evaluator = EvaluatePrequential(
	    pretrain_size=0,
	    show_plot=True,
	    max_samples=20000,
	    metrics=['accuracy','kappa'],
	    batch_size=2,
        output_file="ouput_batch_size_2_pretrain_0_waveform.csv")
	evaluator.evaluate(stream=stream, model=h, model_names=['HT', 'HATT'])
