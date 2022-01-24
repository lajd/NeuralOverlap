train:
	julia src/Trainer.jl --project=.

infer:
	julia src/Infer.jl --project=.

test:
	julia src/Test.jl --project=.
