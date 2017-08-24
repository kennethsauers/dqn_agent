import replay_memory
import numpy as np

mem = replay_memory.replay_memory()
for i in range(57780):
	x = np.arange(0,5,1)
	mem.add(x)

for i in range(20):
	q = np.random.choice(5000,1)
	print(mem.internal_memory[q[0]])

print(mem.get_minibatch(32))

mem.report()