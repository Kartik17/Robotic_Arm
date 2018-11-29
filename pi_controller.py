class PI_controller:
	def __init__(self):
		self.ki = 0.0
		self.kp = 0.0
		self.target = 0.0

		self.min_sat_limit = 0.0
		self.max_sat_limit = 0.0
		
		self.error = 0.0
		self.error_sum = 0
		

		self.last_timestamp = 0.0

	def set_kp(self,kp):
		self.kp = kp

	def set_ki(self,ki):
		self.ki = ki

	def set_target(self,target):
		self.target = target

	def control_law(self,current_position,current_timestamp):
		self.error = self.target - current_position
		
		delta_time = current_timestamp -self.last_timestamp
		self.error_sum += self.error*delta_time

		if delta_time == 0:
			return 0

		self.last_timestamp = current_timestamp

		p = self.kp*error

		i = self.ki*self.error_sum

		u = p + i

		return u




