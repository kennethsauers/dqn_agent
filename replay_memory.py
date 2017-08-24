import numpy as np

class replay_memory():
	def __init__(self, capacity = 5000):
		self.capacity = capacity
		self.replace_index = 0
		self.internal_memory = []

	def add(self, data):
		if len(self.internal_memory) <= self.capacity:
			self.internal_memory.append(data)
		else:
			if self.replace_index >= self.capacity:
				self.replace_index = 0
			self.internal_memory[self.replace_index] = data
			self.replace_index += 1
		return

	def get_minibatch(self, batch_size = 64):
		
		sampels = np.random.choice(len(self.internal_memory), batch_size)
		minibatch = [self.internal_memory[thing] for thing in sampels]
		return minibatch

	def report(self):
		print("capacity: ", self.capacity)
		print("replace_index: ", self.replace_index)
		print("size internal_memory", len(self.internal_memory))