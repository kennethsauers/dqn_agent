class e_greedy():
	def __init__(self, initial_value=1.0, target_value=0.1, exploration_frames=1e6, fixed=False, verbose=True):
		self.epsilon = initial_value
		self.explore = exploration_frames
		self.delta = 0
		self.traget = target_value

	def peek(self):
		return self.epsilon