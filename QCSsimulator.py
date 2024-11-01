import numpy as np
import math
import matplotlib.pyplot as plt
import itertools
from scipy import stats
from time import time
import random
import copy
from tqdm.notebook import tqdm as tqdmn
from tqdm import tqdm
import matplotlib.patches as patches
import signal
import time
from scipy.optimize import fsolve
import matplotlib.ticker as ticker

def QCS_simulation(distribution, n_request, w_request, u, k, p, request_rate, travel_time, fwd_time, N, control_time, N_samples, randomseed=2, check_load=True, debugging=False, plot_MST_distrib=False):
	np.random.seed(randomseed)
	random.seed(randomseed)

	total_request_rate = request_rate*u*(u-1)/2

	if distribution == 'parallel':
		# Don't run simulation for load > 1 (this condition implies load > 1)
		MSerT_p1 = travel_time + fwd_time*(2*N + np.ceil(n_request/k))
		#MSerT_p1 = travel_time+fwd_time*np.ceil(n_request/k) # Old/wrong
		if total_request_rate*MSerT_p1 > 1:
			return np.inf, 0

		# Don't run simulation for load > 1 (this condition is load>1, but it is computationally expensive)
		if check_load:
			avg_service_time = travel_time + fwd_time * (2*N + simulate_multiplexed_window_problem(n_request, k, w_request, p, N_samples=N_samples, randomseed=None)[0])
			load = total_request_rate * avg_service_time
			if load > 1:
				return np.inf, 0

		# Service time: (n_request, w_request, k, p)-window problem
		service_time = travel_time + fwd_time * (2*N + simulate_multiplexed_window_problem(n_request, k, w_request, p, N_samples=1, randomseed=None)[0])
			# Note that, in parallel distribution, we must solve the multiplexed window problem
		table_arrival_times = np.random.exponential(scale=1/total_request_rate, size=N_samples)
		table_start_service = [0]
		table_end_service = [service_time]
		table_sojourn_time = [service_time]

		# M/G/1 queue
		for ii in range(N_samples):
			if ii==0:
				continue
			service_time = travel_time + fwd_time * (2*N + simulate_multiplexed_window_problem(n_request, k, w_request, p, N_samples=1, randomseed=None)[0])
				# Note that, in parallel distribution, we must solve the multiplexed window problem
			table_start_service += [max(0,table_end_service[ii-1]-table_arrival_times[ii])]
			table_end_service += [table_start_service[ii] + service_time]
			table_sojourn_time += [table_end_service[ii]]

		reduced_table_sojourn_time = table_sojourn_time[int(0.1*N_samples):]  # Delete the first 10% samples
		MST = np.mean(reduced_table_sojourn_time)
		MSTerr = np.std(reduced_table_sojourn_time)/np.sqrt(len(reduced_table_sojourn_time))

	elif distribution == 'sequential':
		# Don't run simulation for load > 1 (this condition implies load > 1)
		MSerT_p1 = travel_time + fwd_time*(2*N + n_request)
		#MSerT_p1 = travel_time + fwd_time*n_request
		if total_request_rate*MSerT_p1/k > 1:
			return np.inf, 0

		# Don't run simulation for load > 1 (this condition is load>1, but it is computationally expensive)
		if check_load:
			avg_service_time = travel_time + fwd_time * (2*N + simulate_multiplexed_window_problem(n_request, 1, w_request, p, N_samples=N_samples, randomseed=None)[0])
			load = total_request_rate * avg_service_time / k
			if load > 1:
				return np.inf, 0

		if k==1 and False:
			# Service time (k=1)
			service_time = travel_time + fwd_time * (2*N + simulate_multiplexed_window_problem(n_request, 1, w_request, p, N_samples=1, randomseed=None)[0])
				# Note that, in sequential distribution, we must solve the non-multiplexed window problem
			table_arrival_times = np.random.exponential(scale=1/total_request_rate, size=N_samples)
			table_start_service = [0]
			table_end_service = [service_time]
			table_sojourn_time = [service_time]

			for ii in range(N_samples):
				if ii==0:
					continue
				service_time = travel_time + fwd_time * (2*N + simulate_multiplexed_window_problem(n_request, 1, w_request, p, N_samples=1, randomseed=None)[0])
					# Note that, in sequential distribution, we must solve the non-multiplexed window problem
				table_start_service += [max(0,table_end_service[ii-1]-table_arrival_times[ii])]
				table_end_service += [table_start_service[ii] + service_time]
				table_sojourn_time += [table_end_service[ii]]

			reduced_table_sojourn_time = table_sojourn_time[int(0.1*N_samples):]  # Delete the first 10% samples
			MST = np.mean(reduced_table_sojourn_time)
			MSTerr = np.std(reduced_table_sojourn_time)/np.sqrt(len(reduced_table_sojourn_time))

		else:
			arrival_time = 0
			#interarrival_time = np.random.exponential(scale=1/total_request_rate, size=N_samples)
			service_time = travel_time + fwd_time * (2*N + simulate_multiplexed_window_problem(n_request, 1, w_request, p, N_samples=1, randomseed=None)[0])
				# Note that, in sequential distribution, we must solve the non-multiplexed window problem
			server_assigned = 0
			start_service_time = 0
			servers_finish_time = [start_service_time + service_time]+[0]*(k-1)
			sojourn_time = start_service_time - arrival_time + service_time
			MST = sojourn_time
			MSTvar = 0
			transient_state = True
			if debugging or plot_MST_distrib:
				sojourn_times = []
			ii_steady_state = 0

			for ii in range(N_samples):
				if ii==0:
					continue
				interarrival_time = np.random.exponential(scale=1/total_request_rate)
				#arrival_time += interarrival_time
				arrival_time = interarrival_time
				service_time = travel_time + fwd_time * (2*N + simulate_multiplexed_window_problem(n_request, 1, w_request, p, N_samples=1, randomseed=None)[0])
					# Note that, in sequential distribution, we must solve the non-multiplexed window problem
				server_assigned = servers_finish_time.index(min(servers_finish_time))
				#start_service_time = max(arrival_time, servers_finish_time[server_assigned])
				start_service_time = max(0, servers_finish_time[server_assigned] - arrival_time)
				#servers_finish_time[server_assigned] = start_service_time + service_time
				_servers_finish_time = [max(0, tfin-arrival_time) for tfin in servers_finish_time]
				servers_finish_time = copy.deepcopy(_servers_finish_time)
				servers_finish_time[server_assigned] = start_service_time + service_time
				#sojourn_time = start_service_time - arrival_time + service_time
				sojourn_time = start_service_time + service_time
				MSTvar = MSTvar*ii/(ii+1) + (MST-sojourn_time)**2 * ii/(ii+1)**2
				MST = MST*(ii-ii_steady_state)/(ii-ii_steady_state+1) + sojourn_time/(ii-ii_steady_state+1)

				if transient_state and ii > 0.1*N_samples: # Ignore the first 10% samples
					MST = sojourn_time
					MSTvar = 0
					transient_state = False
					ii_steady_state = ii
					if debugging:
						sojourn_times = []

				if debugging or plot_MST_distrib:
					sojourn_times += [sojourn_time]
				if debugging:
					print('Interarrival/service/start-service time: %d / %d / %d'%(arrival_time, service_time, server_assigned))
					print('Server assigned: %d'%server_assigned)
					print('Servers finish time:', servers_finish_time)
			
			MSTerr = np.sqrt(MSTvar/N_samples)
	else:
		raise ValueError('Packet distribution rule not implemented')

	### DEBUGGING
	if plot_MST_distrib:
		try:
			#T = travel_time + fwd_time*n_request
			#MST_theory = T + T**2 / (2* (2/(request_rate*u*(u-1)) - T))
			scale_time = 1000
			xlim_max = 300000/scale_time
			binsize = 1000/scale_time
			try:
				sojourn_times = reduced_table_sojourn_time
			except:
				pass
			plt.figure()
			sojourn_times = [s/scale_time for s in sojourn_times]
			histcounts, bins, _ = plt.hist(sojourn_times, np.arange(-binsize/2,max(sojourn_times)+binsize/2*1.1, binsize))
			plt.plot([MST/scale_time, MST/scale_time], [0,max(histcounts)], '--k')
			#plt.plot([MST_theory, MST_theory], [0,max(histcounts)], '--')
			plt.gca().set_xlim(left=0)
			plt.gca().set_xlim(right=xlim_max)
			if scale_time==1000:
				plt.xlabel('Sojourn time (ms)')
			if scale_time==1:
				plt.xlabel(r'Sojourn time ($\mu$s)')
			plt.ylabel('Counts')

			# SAVE
			if True:
				filename = 'figs/PAPER-heavytail_n%d_w%s_p%.3f_reqrate%.2e_tfly%s_tfwd%s_tcontrol%s'%(n_request,
										w_request, p, request_rate, travel_time, fwd_time, control_time)
				filename += '_Nsamples%d_randseed%d_binsize%d.pdf'%(N_samples, randomseed, binsize)
				plt.savefig(filename, dpi=300, bbox_inches='tight')
			else:
				plt.show()
		except:
			pass

	return MST, MSTerr

def QCS_theory(distribution, n_request, w_request, u, k, p, request_rate, travel_time, fwd_time, N, control_time):
	'''All units of time are in us. For k=1 we use the exact solution. For k>1 we use the approximation from
		Lee and Longton (1959).
	   ---Inputs---
		· distribution:	(str) 'sequential' or 'parallel'.
		· n_request:	(int) number of entangled pairs per request.
		· w_request:	(int) time window of the request.
		· u:	(int) number of users.
		· k:	(int) number of forwarding devices at the switch.
		· p:	(float) probability that the packets arrive at destination and can be decoded successfully.
		· request_rate:	(float) rate of arrival of new requests per pair of users.
		· travel_time:	(float) travel time from one user to another through the switch
						(assume homogeneous network).
		· fwd_time:		(float) time required to forward a packet.
		· N:	(int) number of repeaters between each user and the central repeater.
		· control_time: (float) time required to communicate with the network controller.
	   
	   ---Outputs---
		· MST:	(float) mean sojourn time.
		· MSTerr:	(float) standard error in the mean sojourn time.'''
	if k<1:
		raise ValueError('k should be larger or equal to 1.')

	if distribution=='parallel':
		assert n_request<=w_request*k, 'Window too short for parallel scheme'

		number_of_pairs_of_users = u*(u-1)/2
		total_request_rate = request_rate*number_of_pairs_of_users

		# Window problem: first and second moment of number of batches until completion
		# In sequential distribution, the number of attempts per batch in the window
		# problem is 1
		avg_B, avg_B2 = solve_multiplexed_window_problem(n_request, k, w_request, p)

		# Mean service time
		MSerT = travel_time + fwd_time * (2*N+1) + fwd_time * (avg_B-1)
		#MSerT = travel_time + fwd_time * avg_B # Old/wrong

		# Mean squared service time

		# Mean squared service time
		MSerT2 = (travel_time**2 + 4*travel_time*fwd_time*N + (2*travel_time*N)**2
					+ (travel_time*2*fwd_time + 4*N*fwd_time**2) * avg_B
					+ fwd_time**2 * avg_B2)
		#MSerT2 = travel_time**2 + 2*travel_time*fwd_time*avg_B + fwd_time**2 * avg_B2 # Old/wrong

		# Number of servers M/G/s - Parallel distribution: M/G/1 queue
		s = 1

		# Load > 1? Then queue grows infinitely
		load = total_request_rate*MSerT/s
		if load > 1:
			return np.infty, np.infty, MSerT, MSerT2

		# Mean queueing time - Parallel distribution: M/G/1 queue
		MQT = MSerT2 / (2 * ( 1/total_request_rate - MSerT ))

		# Mean sojourn time
		MST = MQT + MSerT

	elif distribution=='sequential':
		assert n_request<=w_request, 'Window too short for sequential scheme'

		number_of_pairs_of_users = u*(u-1)/2
		total_request_rate = request_rate*number_of_pairs_of_users

		# Window problem: first and second moment of number of batches until completion
		# In sequential distribution, the number of attempts per batch in the window
		# problem is 1
		avg_B, avg_B2 = solve_multiplexed_window_problem(n_request, 1, w_request, p)

		# Mean service time
		MSerT = travel_time + fwd_time * (2*N+1) + fwd_time * (avg_B-1)
		#MSerT = travel_time + fwd_time * avg_B # Old/wrong

		# Mean squared service time
		MSerT2 = (travel_time**2 + 4*travel_time*fwd_time*N + (2*travel_time*N)**2
					+ (travel_time*2*fwd_time + 4*N*fwd_time**2) * avg_B
					+ fwd_time**2 * avg_B2)
		#MSerT2 = travel_time**2 + 2*travel_time*fwd_time*avg_B + fwd_time**2 * avg_B2 # Old/wrong

		# Number of servers in the M/G/s queue
		s = k

		# Load > 1? Then queue grows infinitely
		load = total_request_rate*MSerT/s
		if load > 1:
			return np.infty, np.infty, MSerT, MSerT2

		if s==1:
			# Mean queueing time
			MQT = MSerT2 / (2 * ( 1/total_request_rate - MSerT ))

			# Mean sojourn time
			MST = MQT + MSerT

		else:
			# Squared coefficient of variation of service time
			C2 = (MSerT2 - MSerT**2) / MSerT**2

			# Probability of no of requests in the system (waiting or being processed)
			P_Ns_0 = 1 / (sum([( (total_request_rate*MSerT)**j
								 / (np.math.factorial(j)) ) for j in range(s)])
							+ (total_request_rate*MSerT)**s 
								/ (np.math.factorial(s) * (1-load)) )
			
			# Probability of more than s requests in the system (waiting or being processed)
			P_Ns_k = ((total_request_rate*MSerT)**s * P_Ns_0) / (np.math.factorial(s)*(1-load))

			# Mean queueing time of M/M/s queue with same mean service time
			MQT_MMk = P_Ns_k / (s * (1/MSerT) * (1-load))

			# Mean queueing time
			MQT = (C2+1) * MQT_MMk / 2

			# Mean sojourn time
			MST = MQT + MSerT

			#print(MSerT, MSerT2)
			#print(C2, P_Ns_0, P_Ns_k, MQT_MMk, MQT)

	else:
		raise ValueError('Unknown packet distribution.')

	return MST, MQT, MSerT, MSerT2

def solve_multiplexed_window_problem(n, m, window_size, p, validate_moments=False, plot_distribution=False):
	'''Solves the window problem (see Davies et al. 2023).
	   ---Inputs---
		· n:	(int) total number of successes required.
		· m:	(int) number of attempts per batch (maximum number of successes per batch).
		· window_size:	(int) time window.
		· p:	(float) probability of success of each attempt.
		· validate_moments:	(bool) if True, builds a numerical distribution and computes the
							mean and variance to validate the analytical results. Then, it prints
							both results for both methods.
							For m=1, if True, the solution is computed using the analytical solutions
							for a negative binomial distribtuion. Otherwise (m=1 and False, or m>1),
							the solution is computed by solving the window problem from Davies2023
							(we do NOT use their solution, since they only consider m=1).
		· plot_distribution:	(bool) if True, plots the probability distribution of
								the number of batches until success.
	   
	   ---Outputs---
		· expected_num_batches:	(float) expected number of batches to completion (first moment).
		· expected_num_batches2:	(float) expected number of batches squared (second moment).'''
	assert p<=1, 'p should be between 0 and 1.'

	## p=0
	if p == 0:
		return np.inf, 0

	## p=1
	if p == 1:
		expected_num_batches = np.ceil(n/m)
		expected_num_batches2 = (np.ceil(n/m))**2
		return expected_num_batches, expected_num_batches2

	## p<1, finite window (requires solving the window problem)
	elif not window_size == np.inf:
		raise ValueError('Finite window not implemented (exists a closed-form for n=2 and m=1, see Davies2023)')

	## p<1, infinite window, m=1: the number of batches until success follows a negative binomial
	elif m == 1 and not validate_moments:
		expected_num_batches = n/p
		expected_num_batches2 = n*(1-p)/p**2 + expected_num_batches**2
		return expected_num_batches, expected_num_batches2

	## p<1, infinite window, m>1
	else:
		# Compute the probability distribution of the number of batches until success
		P_equal = [0] # P_equal[b] is the probability that the number of batches is equal to b
		P_larger = [1] # P_larger[b] is the probability that the number of batches is larger than b
		P_normalization = sum(P_equal)
		epsilon = 1e-5 # Tolerance to stop calculating for large number of batches
		b = 1
		while P_normalization < 1-epsilon:
			# Find all patterns of b batches, where each batch can have at most m successes
			# and there are less than n successes in total
			patterns = [seq for seq in itertools.product(list(range(m+1)), repeat=b) if sum(seq) < n]
			# ISSUE: The running time scales so badly because there are a lot of possible patters,
			# i.e., large len(patterns), for increasing b.
			P_larger += [round((1-p)**(m*b)*sum([np.prod([math.comb(m,x) for x in pattern])*(p/(1-p))**(sum(pattern)) for pattern in patterns]), 15)]
			P_equal += [1-P_larger[-1]-P_normalization]
			P_normalization = sum(P_equal)
			b += 1

		# Calculate expected value
		expected_num_batches = sum(P_larger)
		expected_num_batches2 = sum([b**2*P_equal[b] for b in range(len(P_equal))])

		# Validate with alternative calculation
		if validate_moments:
			custm = stats.rv_discrete(name='custm', values=(range(len(P_equal)), P_equal/P_normalization))
			print('Expected number of batches:')
			print('---Analytical: %.3f'%expected_num_batches)
			print('---Numerical: %.3f'%custm.mean())
			print('Expected squared number of batches:')
			print('---Analytical: %.3f'%expected_num_batches2)
			print('---Numerical: %.3f'%(custm.var()+custm.mean()**2))

		# Plot
		if plot_distribution:
			plt.plot(range(len(P_equal)), P_equal, 'ro', ms=12, mec='r')
			plt.vlines(range(len(P_equal)), 0, P_equal, colors='r', lw=4)
			plt.xlabel('Number of batches until success')
			plt.ylabel('PMF')
			plt.show()

		return expected_num_batches, expected_num_batches2

def simulate_multiplexed_window_problem(n, m, window_size, p, N_samples, randomseed, return_distribution=False):
	'''Simulates the multiplexed window problem (see Davies et al. 2023) to estimate
		the probability distribution of the number of batches until completion.
		To simulate the problem once and get a sample, set N_samples=1 and retrieve
		simulate_multiplexed_window_problem()[0].
	   ---Inputs---
		· n:	(int) total number of successes required.
		· m:	(int) number of attempts per batch (maximum number of successes per batch).
		· window_size:	(int) time window.
		· p:	(float) probability of success of each attempt.
		· N_samples:	(int) number of samples.
		· randomseed:	(int) random seed.
		· return_distribution:	(bool) if True, return the whole probability distribution of
								the number of batches until success. Otherwise, only return the
								sample average, 
	   
	   ---Outputs---
		· expected_num_batches:	(float) expected number of batches to completion (first moment).
		· expected_num_batches2:	(float) expected number of batches squared (second moment).'''
	assert p<=1, 'p should be between 0 and 1.'

	if m*window_size<n: # Impossible to complete
		return np.inf, 0, 0

	if randomseed is not None:
		np.random.seed(randomseed)
		random.seed(randomseed)

	num_batches_vec = [] # Number of batches until completion per sample

	# Run N_samples
	for sample in range(N_samples):
		# FINITE WINDOW
		if not window_size == np.inf:
			num_batches = 0
			total_successes = 0
			successes_window = np.zeros(window_size) # Store the number of successes in the last
													 # window_size batches
			while total_successes < n:
				new_batch = np.random.rand(m)
				new_successes = sum(new_batch<p)
				# Update count of successes
				total_successes += new_successes
				total_successes -= successes_window[num_batches%window_size]
				# Update window
				successes_window[num_batches%window_size] = new_successes
				# Increase number of batches
				num_batches += 1

		# INFINITE WINDOW
		else:
			num_batches = 0
			total_successes = 0

			while total_successes < n:
				new_batch = np.random.rand(m)
				new_successes = sum(new_batch<p)
				# Update count of successes
				total_successes += new_successes
				# Increase number of batches
				num_batches += 1

		num_batches_vec += [num_batches]

	avg_num_batches = np.mean(num_batches_vec)
	std_num_batches = np.std(num_batches_vec)
	stderr_num_batches = std_num_batches/np.sqrt(N_samples)

	return avg_num_batches, std_num_batches, stderr_num_batches


#---------------------------------------------------------------------
#---------------------------------------------------------------------
#----------------------- MISC -----------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
def a_eff(L_0, fitting='fourth-order'):
	if fitting=='third-order':
		a1 = 0.0024237022454138904 # [1/km]
		a2 = -0.001861344427541674 # [1/km^2]
		a3 = 0.000462010533071338 # [1/km^3]
		a4 = 0 # [1/km^4]
	elif fitting=='fourth-order':
		a1 = 0 # [1/km]
		a2 = 0.00027707970095919093 # [1/km^2]
		a3 = 0 # [1/km^3]
		a4 = 0.000028537199832870782 # [1/km^4]
	return a1*L_0 + a2*L_0**2 + a3*L_0**3 + a4*L_0**4

def find_critical_L(distribution, N, n_request, w_request, u, k, request_rate, speed_light, fwd_time_0, control_time, N_samples, randomseed, tolerance=1e-5):
	'''Finds the critical user-hub distance L in use case B, where we have N repeaters
		between each user and the hub, and L_0 = L/(N+1) and p = 10**(-a_eff(L_0)*(2*L)/10).
		The critical L is the one that makes the load equal to 1.
		We ASSUME that the travel_time is given by 2L/c.'''

	fwd_time = fwd_time_0

	# The load is 1 when the avg service time is service_time_CRIT
	# The minimum load happens for L=0 and p=1, in which case the avg service time is fwd_time * (2N+ceil(n/m))
	if distribution=='sequential':
		assert n_request<=w_request, 'Window too short for sequential scheme'
		service_time_CRIT = (2*k / (request_rate*u*(u-1)))
		min_load = request_rate*u*(u-1) * fwd_time*(2*N+np.ceil(n_request)) / (2*k)
	elif distribution=='parallel':
		assert n_request<=w_request*k, 'Window too short for parallel scheme'
		service_time_CRIT = (2 / (request_rate*u*(u-1)))
		min_load = request_rate*u*(u-1) * fwd_time*(2*N+np.ceil(n_request/k)) / 2
	# If the lower bound on the load is larger than 1, L_crit=0 (no service is possible)
	if min_load > 1:
		return 0

	L_old = 0 # We assume this minimum value for L [km]
	delta_L = 1 # Initial increase in L while exploring the parameter space

	def service_time_func(L):
		L_0 = L/(N+1)
		p = 10**(-a_eff(L_0)*(2*L)/10)

		if p < 1e-3: # This is to prevent very long calculations
			return np.inf

		if distribution=='sequential':
			m = 1
		elif distribution=='parallel':
			m = k

		if (p==1 or w_request==np.inf) and m==1: # and (not run_simulation):
			avg_B, _ = solve_multiplexed_window_problem(n_request, m, w_request, p)
		else:
			avg_B, _, _ = simulate_multiplexed_window_problem(n_request, m, w_request, p, N_samples, randomseed)

		assert avg_B >= np.ceil(n_request/m) # This minimum value happens for p=1
		travel_time = 2*L/speed_light
		service_time = travel_time + fwd_time * (2*N+avg_B)
		return service_time

	L = L_old
	error = -(tolerance+1)
	# We stop when the error in the load is very small or when the update in L is very small
	while np.abs(error) > tolerance and delta_L > tolerance:
		if error < 0:
			L_old = L
			L = L + delta_L
		else:
			L = (L+L_old)/2
			delta_L = delta_L/2
		error = service_time_func(L)-service_time_CRIT # error=0 for the critical L
	return L

def find_critical_u(distribution, n_request, w_request, p, k, request_rate, fwd_time, N, travel_time, N_samples=1000, randomseed=2, run_simulation=False):
	'''Finds the critical number of users.'''
	if distribution == 'sequential':
		s = k
	elif distribution == 'parallel':
		s = 1

	if p==1:
		avg_service_time = travel_time + fwd_time*( 2*N + np.ceil(n_request*s/k) )
	elif w_request==np.inf:
		avg_service_time = travel_time + fwd_time*(2*N+1) + fwd_time*(solve_multiplexed_window_problem(n_request, int(k/s), w_request, p,
																		validate_moments=False, plot_distribution=False)[0] - 1)
	elif run_simulation:
		#np.random.seed(randomseed)
		#random.seed(randomseed) # ISSUE: before, we had randomseed=None in the function below
		avg_service_time = travel_time + fwd_time*(2*N+1) + fwd_time*(simulate_multiplexed_window_problem(n_request, int(k/s), w_request, p,
																		N_samples=N_samples, randomseed=randomseed)[0] - 1)
	else:
		raise ValueError
	z = 8 * s / (request_rate * avg_service_time)
	crit_u = int(np.floor( 0.5 + 0.5 * np.sqrt(z+1) ))
	return crit_u


#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------- PLOTS AND DATA ANALYSIS ----------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
def MST_and_load_vs_param(varying_param, u, k, n_request, w_request, p, request_rate, travel_time, fwd_time, N, control_time, N_samples, randomseed, logscale=False, analytical_on=True, plot_rel_ratio=False, savefig=False):
	distribution_vec = ['sequential','parallel']
	
	if varying_param == 'u':
		varying_array = u
	elif varying_param == 'k':
		varying_array = k
	else:
		raise ValueError('Unknown varying_param')

	if analytical_on: # For some combinations of parameters, we have no analytics for the window problem
		MST_theory_vec = [[],[]]
	if N_samples is not None:
		MST_sim_vec = [[],[]]
		MST_err_sim_vec = [[],[]]
	if p==1: # Load formula only applies to p=1
		load_vec = [[],[]]

	for idx_distrib, distribution in enumerate(distribution_vec):
		for varying_value in tqdmn(varying_array, distribution,leave='False'):
			if varying_param == 'u':
				u = varying_value
			elif varying_param == 'k':
				k = varying_value
			if analytical_on:
				MST_theory = QCS_theory(distribution, n_request, w_request, u, k, p,
								   request_rate, travel_time, fwd_time, N, control_time)[0]
			if N_samples is not None:
				MST_sim, MST_err_sim = QCS_simulation(distribution, n_request, w_request, u, k, p,
								   request_rate, travel_time, fwd_time, N, control_time, N_samples, randomseed)

			if p==1: # Load formula only applies to p=1
				if distribution == 'sequential':
					load = request_rate*u*(u-1)*(travel_time + fwd_time*n_request)/(2*k)
				if distribution == 'parallel':
					load = request_rate*u*(u-1)*(travel_time + fwd_time*np.ceil(n_request/k))/2
			
			if analytical_on:
				MST_theory_vec[idx_distrib] += [MST_theory]
			if N_samples is not None:
				MST_sim_vec[idx_distrib] += [MST_sim]
				MST_err_sim_vec[idx_distrib] += [MST_err_sim]
			if p==1:
				load_vec[idx_distrib] += [load]

	# Create figure and axes
	fig, ax1 = plt.subplots()
	if plot_rel_ratio and not analytical_on:
		ax3 = ax1.twinx()
	elif p==1:
		ax2 = ax1.twinx()
	colors = ['tab:blue', 'tab:orange']
	markers = ['o','^']

	# Plot MST in primary axis
	for idx_distrib, distribution in enumerate(distribution_vec):
		if analytical_on:
			ax1.plot(varying_array, MST_theory_vec[idx_distrib], color=colors[idx_distrib], alpha=0.5, label='')
			ax1.scatter(varying_array, MST_theory_vec[idx_distrib], color=colors[idx_distrib],
						marker=markers[idx_distrib], label=distribution)
		if N_samples is not None:
			ax1.plot(varying_array, MST_sim_vec[idx_distrib], color=colors[idx_distrib], alpha=0.5, label='')
			ax1.errorbar(varying_array, MST_sim_vec[idx_distrib], yerr=2*np.array(MST_err_sim_vec[idx_distrib]),
							color=colors[idx_distrib], marker=markers[idx_distrib], linestyle='', capsize=5,
							label=distribution)
	if plot_rel_ratio and not analytical_on:
		rel_ratio = [(MST_sim_vec[0][i]-MST_sim_vec[1][i])/MST_sim_vec[1][i] for i in range(len(MST_sim_vec[0]))]
		ax3.scatter(varying_array, rel_ratio, color='k', marker='^', linestyle='')
		print(MST_sim_vec[0])
		print(MST_sim_vec[1])
		print(rel_ratio)
		
	# Plot load in secondary axis
	if p==1:
		for idx_distrib, distribution in enumerate(distribution_vec):
			ax2.plot(varying_array, load_vec[idx_distrib], '--', color=colors[idx_distrib], label=distribution)
		ax2.plot(varying_array, varying_array*0+1, '--', color='k', label='Unit load')

	# Plot specs
	plt.xlim(varying_array[0],varying_array[-1])
	ax1.legend()
	if varying_param == 'u':
		ax1.set_xlabel(r'Number of users $u$')
	elif varying_param == 'k':
		ax1.set_xlabel(r'Number of forwarding stations $k$')
	ax1.set_ylabel(r'Mean Sojourn Time ($\mu$s)', color='k') # Label
	if plot_rel_ratio and not analytical_on:
		ax3.set_ylabel(r'Rel. diff. (triangles)', color='k') # Label
		ax3.spines['right'].set_color('k') # Axis
		ax3.tick_params(axis='y', which='both', colors='k') # Ticks
	elif p==1:
		ax2.set_ylabel(r'Load (dashed lines)', color='k') # Label
		ax2.spines['right'].set_color('k') # Axis
		ax2.tick_params(axis='y', which='both', colors='k') # Ticks
	ax1.set_xticks(varying_array, minor=True)
	ax1.grid(which='both', axis='x', alpha=0.2)

	if logscale:
		ax1.set_yscale('log')

	if savefig:
		filename = 'figs/PAPER-MSTandLOADvs%s'%varying_param
		if varying_param == 'u':
			filename += '_k%d'%k
		elif varying_param == 'k':
			filename += '_u%d'%u
		filename += '_n%d_w%s_p%.3f_reqrate%.2e_tfly%s_tfwd%s_N%d_tcontrol%s'%(n_request,
								w_request, p, request_rate, travel_time, fwd_time, N, control_time)
		if N_samples is None:
			filename += '.pdf'
		else:
			filename += '_Nsamples%d_randomseed%s.pdf'%(N_samples, randomseed)
		plt.savefig(filename, dpi=300, bbox_inches='tight')
	else:
		plt.show()

	return

def MST_vs_param_many_p(varying_param, distribution, u, k, n_request, w_request, p_vec, request_rate, travel_time, fwd_time, control_time, N_samples, randomseed, logscale=False, analytical_on=True, savefig=False):
	
	if varying_param == 'u':
		varying_array = u
	elif varying_param == 'k':
		varying_array = k
	else:
		raise ValueError('Unknown varying_param')

	if analytical_on: # For some combinations of parameters, we have no analytics for the window problem
		MST_theory_vec = [[] for p in p_vec]
	MST_sim_vec = [[] for p in p_vec]
	MST_err_sim_vec = [[] for p in p_vec]

	for idx_p, p in enumerate(p_vec):
		for varying_value in tqdmn(varying_array, 'p = %.2f'%p,leave='False'):
			if varying_param == 'u':
				u = varying_value
			elif varying_param == 'k':
				k = varying_value

			if analytical_on:
				MST_theory = QCS_theory(distribution, n_request, w_request, u, k, p,
								   request_rate, travel_time, fwd_time, N, control_time)[0]
			MST_sim, MST_err_sim = QCS_simulation(distribution, n_request, w_request, u, k, p,
							   request_rate, travel_time, fwd_time, N, control_time, N_samples, randomseed,
							   check_load=True)

			if analytical_on:
				MST_theory_vec[idx_p] += [MST_theory]
			MST_sim_vec[idx_p] += [MST_sim]
			MST_err_sim_vec[idx_p] += [MST_err_sim]

	# Create figure and axes
	fig, ax1 = plt.subplots()
	if distribution=='sequential':
		cmap = plt.cm.get_cmap('inferno')
	elif distribution=='parallel':
		cmap = plt.cm.get_cmap('viridis')
	colors = [cmap(i/len(p_vec)) for i in range(len(p_vec))]

	# Plot MST in primary axis
	for idx_p, p in enumerate(p_vec):
		linestyle_sim = '-'
		if analytical_on:
			ax1.plot(varying_array, MST_theory_vec[idx_p], color=colors[idx_p], label='p = %.2f (theory)'%p)
			linestyle_sim = ''
		ax1.errorbar(varying_array, MST_sim_vec[idx_p], yerr=2*np.array(MST_err_sim_vec[idx_p]),
					 color=colors[idx_p], marker='o', linestyle=linestyle_sim, capsize=5, label='p = %.2f'%p)

	# Plot specs
	plt.xlim(varying_array[0],varying_array[-1])
	plt.legend()
	if varying_param == 'u':
		ax1.set_xlabel(r'Number of users $u$')
	elif varying_param == 'k':
		ax1.set_xlabel(r'Number of forwarding stations $k$')
	ax1.set_ylabel(r'Mean Sojourn Time', color='k') # Label
	ax1.set_xticks(varying_array, minor=True)
	ax1.grid(which='both', axis='x', alpha=0.2)

	if logscale:
		ax1.set_yscale('log')

	if savefig:
		filename = 'figs/RESULTS-C-%sMSTvspvs%s'%(distribution,varying_param)
		if varying_param == 'u':
			filename += '_k%d'%k
		elif varying_param == 'k':
			filename += '_u%d'%u
		filename += '_n%d_w%d_reqrate%.2e_tfly%s_tfwd%s_tcontrol%s_Nsamples%d_randomseed%s.pdf'%(n_request,
								w_request, request_rate, travel_time, fwd_time, control_time, N_samples, randomseed)
		plt.savefig(filename, dpi=300, bbox_inches='tight')
	else:
		plt.show()

	return

def calculate_MSTdiff_vs_u_and_k(u_vec, k_vec, n_request, w_request, p, request_rate, travel_time, fwd_time, N, control_time, N_samples=1000, randomseed=2, run_simulation=True):
	distribution_vec = ['sequential','parallel']

	# CALCULATE MST
	if not run_simulation:
		MST_theory_vec = [np.zeros((len(u_vec),len(k_vec))),np.zeros((len(u_vec),len(k_vec)))]
	else:
		MST_sim_vec = [np.zeros((len(u_vec),len(k_vec))),np.zeros((len(u_vec),len(k_vec)))]
		MST_err_sim_vec = [np.zeros((len(u_vec),len(k_vec))),np.zeros((len(u_vec),len(k_vec)))]

	for idx_distrib, distribution in enumerate(distribution_vec):
		for idx_u in tqdm(range(len(u_vec)),distribution,leave=False):
			u = u_vec[idx_u]
			for idx_k, k in enumerate(k_vec):
				if not run_simulation:
					MST_theory = QCS_theory(distribution, n_request, w_request, u, k, p,
									   request_rate, travel_time, fwd_time, N, control_time)[0]
					MST_theory_vec[idx_distrib][idx_u][idx_k] = MST_theory
				else:
					MST_sim, MST_err_sim = QCS_simulation(distribution, n_request, w_request, u, k, p,
							   request_rate, travel_time, fwd_time, N, control_time, N_samples, randomseed)
					MST_sim_vec[idx_distrib][idx_u][idx_k] = MST_sim
					MST_err_sim_vec[idx_distrib][idx_u][idx_k] = MST_err_sim

	# COMPUTE CRITICAL LOADS
	if True:
		u_crit_vec = [[], []] # Critical number of users for each value of k (first list for seq, second list for par)
		k_vec_critical = [[0.5], [0.5]] # (For plot purposes) First list for seq, second list for par
		u_vec_critical = [[1.5], [1.5]] # (For plot purposes) First list for seq, second list for par

		for idx_distrib, distribution in enumerate(['sequential', 'parallel']):
			if distribution == 'sequential':
				s = k
			elif distribution == 'parallel':
				s = 1
			for k in tqdm(k_vec,'Load (%s)'%distribution,leave=False):
				k_vec_critical[idx_distrib] += [k-0.5, k+0.5]
				crit_u = find_critical_u(distribution, n_request, w_request, p, k, request_rate, fwd_time, N, travel_time, N_samples=N_samples, randomseed=randomseed, run_simulation=run_simulation)
				u_crit_vec[idx_distrib] += [crit_u]
				u_vec_critical[idx_distrib] += [crit_u+0.5, crit_u+0.5]

	# RETURN DATA
	data = {'k_vec_critical': k_vec_critical,
			'u_vec_critical': u_vec_critical,
			'u_crit_vec': u_crit_vec}
	if not run_simulation:
		data['MST_theory_vec'] = MST_theory_vec
		return data
				
	else:
		data['MST_sim_vec'] = MST_sim_vec
		data['MST_err_sim_vec'] = MST_err_sim_vec
		return data

def plot_MSTdiff_vs_u_and_k(data, u_vec, k_vec, n_request, w_request, p, request_rate, travel_time, fwd_time, N, control_time, N_samples=1000, randomseed=2, run_simulation=False, savefig=False, dark=False):
	
	if dark:
		plt.style.use('dark_background')
	else:
		plt.rcParams.update(plt.rcParamsDefault)

	# Retrieve data
	k_vec_critical = data['k_vec_critical']
	u_vec_critical = data['u_vec_critical']
	u_crit_vec = data['u_crit_vec']

	if not run_simulation:
		MST_theory_vec = data['MST_theory_vec']
	else:
		MST_sim_vec = data['MST_sim_vec']
		MST_err_sim_vec = data['MST_err_sim_vec']

	# Define surface to plot
	if not run_simulation:
		surf = (MST_theory_vec[0]-MST_theory_vec[1])/(MST_theory_vec[1])
	else:
		surf = (MST_sim_vec[0]-MST_sim_vec[1])/(MST_sim_vec[1])
	# Delete cases with load>1
	for iix, uu in enumerate(u_vec):
		for jjx, kk in enumerate(k_vec):
			if run_simulation: # If we run the simulation, we need to use the estimates of the critical load since MST is not infinite
				if uu > u_crit_vec[0][jjx] or uu > u_crit_vec[1][jjx]:
					surf[iix][jjx] = np.nan

	cbar_min = -max(-np.floor(np.nanmin(surf)), np.ceil(np.nanmax(surf)))
	cbar_max = -cbar_min
	dx = 0.5
	dy = 0.5

	# Create figure and axes
	fig, ax = plt.subplots()

	# Intermediate variables
	surfmax = np.max(np.abs(surf))
	surfmin = np.min(np.abs(surf))
	cbar_mid = cbar_min+(cbar_max-cbar_min)/2
	cmap = plt.cm.get_cmap('bwr')
	cont = ax.imshow(np.flip(surf.T, 0), cmap=cmap,
					  extent=[u_vec[0]-dx, u_vec[-1]+dx,
							  k_vec[0]-dy, k_vec[-1]+dy],
					  vmin=cbar_min, vmax=cbar_max)
	ax.set_aspect(aspect="auto")

	# Colorbar #
	cbar_label = r'$\frac{\mathrm{MST}_\mathrm{seq}-\mathrm{MST}_\mathrm{par}}{\mathrm{MST}_\mathrm{par}}$' #(%)'
	cbar = fig.colorbar(cont, ax=ax, aspect=10)
	cbar.set_label(cbar_label)
	if surfmax == surfmin:
		cbar.set_ticks([0,surfmax])
		cbar.ax.set_yticklabels([r'${:.0f}\%$'.format(0),
								 r'${:.0f}\%$'.format(surfmax)])
	else:
		cbar.set_ticks([cbar_min,cbar_mid,cbar_max])
		cbar.ax.set_yticklabels([r'${:.1f}$'.format(cbar_min),
								 r'${:.0f}$'.format(cbar_mid),
								 r'${:.1f}$'.format(cbar_max)])

	# Plot critical loads
	colors_distrib = ['b', 'r']
	for idx_distrib, distribution in enumerate(['sequential', 'parallel']):
		plt.plot(u_vec_critical[idx_distrib], k_vec_critical[idx_distrib],'--', color=colors_distrib[idx_distrib])

	# Annotated colormap
	dpx = 0.03 # Extra patch size to cover blank space between patches
	for iix, uu in enumerate(u_vec):
		for jjx, kk in enumerate(k_vec):
			if not run_simulation:
				if np.isinf(MST_theory_vec[1][iix,jjx]) and (not np.isinf(MST_theory_vec[0][iix,jjx])):
					text = ax.text(uu, kk, 's', ha='center', va='center', color='b')
				elif np.isnan(surf[iix,jjx]):
					text = ax.text(uu, kk, 'x', ha='center', va='center', color='k')
					rect = patches.Rectangle((uu-0.5-dpx/2, kk-0.5-dpx/2), 1+dpx, 1+dpx, linewidth=0, facecolor=[0.3,0.3,0.3])
					ax.add_patch(rect)
			else: # If we run the simulation, we need to use the estimates of the critical load since MST is not infinite
				if uu > u_crit_vec[0][jjx] and uu > u_crit_vec[1][jjx]:
					text = ax.text(uu, kk, 'x', ha='center', va='center', color='k')
					rect = patches.Rectangle((uu-0.5-dpx/2, kk-0.5-dpx/2), 1+dpx, 1+dpx, linewidth=0, facecolor=[0.3,0.3,0.3])
					ax.add_patch(rect)
				elif uu > u_crit_vec[0][jjx] and uu <= u_crit_vec[1][jjx]:
					text = ax.text(uu, kk, 'p', ha='center', va='center', color='r')
				elif uu <= u_crit_vec[0][jjx] and uu > u_crit_vec[1][jjx]:
					text = ax.text(uu, kk, 's', ha='center', va='center', color='b')

	# Plot specs
	plt.xlabel(r'Number of users $u$')
	plt.ylabel(r'Number of forwarding stations $k$')
	plt.xlim((u_vec[0]-0.5,u_vec[-1]+0.5))
	plt.ylim((k_vec[0]-0.5,k_vec[-1]+0.5))

	# Major tick location
	ax.set_xticks([5,10,15])
	ax.set_yticks([2,4,6,8,10,12])

	# Minor tick frequency
	x_minor_intervals = 5 # Number of minor intervals between two major ticks               
	ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(x_minor_intervals))
	y_minor_intervals = 2 # Number of minor intervals between two major ticks
	ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(y_minor_intervals))

	# SAVE
	if savefig:
		if dark:
			filename = 'figs/DARK_'
		else:
			filename = 'figs/'
		filename += 'PAPER-MSTvsuvsk_n%d_w%s_p%.3f_reqrate%.2e_tfly%s_tfwd%s_N%d_tcontrol%s'%(n_request,
								w_request, p, request_rate, travel_time, fwd_time, N, control_time)
		if run_simulation:
			filename += '_Nsamples%d_randseed%d.pdf'%(N_samples, randomseed)
		else:
			filename += '.pdf'
		plt.savefig(filename, dpi=300, bbox_inches='tight')
	else:
		plt.show()

	return

def plot_MSTdiffERRORS_vs_u_and_k(data, u_vec, k_vec, n_request, w_request, p, request_rate, travel_time, fwd_time, control_time, N_samples=1000, randomseed=2, run_simulation=False, savefig=False, dark=False):
	
	if dark:
		plt.style.use('dark_background')
	else:
		plt.rcParams.update(plt.rcParamsDefault)

	# Retrieve data
	k_vec_critical = data['k_vec_critical']
	u_vec_critical = data['u_vec_critical']
	u_crit_vec = data['u_crit_vec']

	if not run_simulation:
		return
	else:
		MST_sim_vec = data['MST_sim_vec']
		MST_err_sim_vec = data['MST_err_sim_vec']

	# Define surface to plot
	A = data['MST_sim_vec'][0]
	B = data['MST_sim_vec'][1]
	eA = data['MST_err_sim_vec'][0]
	eB = data['MST_err_sim_vec'][1]
	# MST_diff = (A-B)/B
	MST_diff_error = np.sqrt( eA**2 / B**2 + eB**2 * A**2 / B**4 )
	surf = MST_diff_error

	# Delete cases with load>1
	for iix, uu in enumerate(u_vec):
		for jjx, kk in enumerate(k_vec):
			if run_simulation: # If we run the simulation, we need to use the estimates of the critical load since MST is not infinite
				if uu > u_crit_vec[0][jjx] or uu > u_crit_vec[1][jjx]:
					surf[iix][jjx] = np.nan

	cbar_min = 0
	cbar_max = np.ceil(np.nanmax(surf))
	dx = 0.5
	dy = 0.5

	# Create figure and axes
	fig, ax = plt.subplots()

	# Intermediate variables
	surfmax = np.max(np.abs(surf))
	surfmin = np.min(np.abs(surf))
	cbar_mid = cbar_min+(cbar_max-cbar_min)/2
	cmap = plt.cm.get_cmap('Reds')
	cont = ax.imshow(np.flip(surf.T, 0), cmap=cmap,
					  extent=[u_vec[0]-dx, u_vec[-1]+dx,
							  k_vec[0]-dy, k_vec[-1]+dy],
					  vmin=cbar_min, vmax=cbar_max)
	ax.set_aspect(aspect="auto")

	# Plot specs
	plt.xlabel(r'Number of users $u$')
	plt.ylabel(r'Number of forwarding stations $k$')
	plt.ylim((k_vec[0]-0.5,k_vec[-1]+0.5))

	# Colorbar #
	cbar_label = r'Error in relative difference'
	cbar = fig.colorbar(cont, ax=ax, aspect=10)
	cbar.set_label(cbar_label)
	if surfmax == surfmin:
		cbar.set_ticks([0,surfmax])
		cbar.ax.set_yticklabels([r'${:.0f}$'.format(0),
								 r'${:.0f}$'.format(surfmax)])
	else:
		cbar.set_ticks([cbar_min,cbar_mid,cbar_max])
		cbar.ax.set_yticklabels([r'${:.1f}$'.format(cbar_min),
								 r'${:.0f}$'.format(cbar_mid),
								 r'${:.1f}$'.format(cbar_max)])

	# Plot critical loads
	colors_distrib = ['b', 'r']
	for idx_distrib, distribution in enumerate(['sequential', 'parallel']):
		plt.plot(u_vec_critical[idx_distrib], k_vec_critical[idx_distrib],'--', color=colors_distrib[idx_distrib])

	# Annotated colormap
	for iix, uu in enumerate(u_vec):
		for jjx, kk in enumerate(k_vec):
			if not run_simulation:
				if np.isinf(MST_theory_vec[1][iix,jjx]) and (not np.isinf(MST_theory_vec[0][iix,jjx])):
					text = ax.text(uu, kk, 's', ha='center', va='center', color='b')
				elif np.isnan(surf[iix,jjx]):
					text = ax.text(uu, kk, 'x', ha='center', va='center', color='k')
					rect = patches.Rectangle((uu-0.5, kk-0.5), 1, 1, linewidth=0, facecolor=[0.3,0.3,0.3])
					ax.add_patch(rect)
			else: # If we run the simulation, we need to use the estimates of the critical load since MST is not infinite
				#if min_k_vec[1][iix] > kk and min_k_vec[0][iix] > kk:
				if uu > u_crit_vec[0][jjx] and uu > u_crit_vec[1][jjx]:
					text = ax.text(uu, kk, 'x', ha='center', va='center', color='k')
					rect = patches.Rectangle((uu-0.5, kk-0.5), 1, 1, linewidth=0, facecolor=[0.3,0.3,0.3])
					ax.add_patch(rect)
				#elif min_k_vec[1][iix] <= kk and min_k_vec[0][iix] > kk:
				elif uu > u_crit_vec[0][jjx] and uu <= u_crit_vec[1][jjx]:
					text = ax.text(uu, kk, 'p', ha='center', va='center', color='r')
				#elif min_k_vec[1][iix] > kk and min_k_vec[0][iix] <= kk:
				elif uu <= u_crit_vec[0][jjx] and uu > u_crit_vec[1][jjx]:
					text = ax.text(uu, kk, 's', ha='center', va='center', color='b')
	# SAVE
	if savefig:
		if dark:
			filename = 'figs/DARK_'
		else:
			filename = 'figs/'
		filename += 'PAPER-MSTERRORvsuvsk_n%d_w%s_p%.3f_reqrate%.2e_tfly%s_tfwd%s_tcontrol%s'%(n_request,
								w_request, p, request_rate, travel_time, fwd_time, control_time)
		if run_simulation:
			filename += '_Nsamples%d_randseed%d.pdf'%(N_samples, randomseed)
		else:
			filename += '.pdf'
		plt.savefig(filename, dpi=300, bbox_inches='tight')
	else:
		plt.show()

	return

def plot_critL_vs_u(crit_L_vec, distribution, N_vec, u_vec, n_request, w_request, k, request_rate, speed_light, fwd_time_0, control_time, N_samples, randomseed, plot_relative=False, savefig=False):
	## Plot params ##
	xlimvec = [u_vec[0], u_vec[-1]]
	y_upperlim = 150

	## Plot ##
	markers = ['o','x','^','v','s','d']

	if distribution=='sequential':
		cmap = plt.colormaps['inferno']
	elif distribution=='parallel':
		cmap = plt.colormaps['viridis']
	colors = [cmap(i/len(N_vec)) for i in range(len(N_vec))]

	fig, ax = plt.subplots()

	if N_vec[0] != 0:
		raise ValueError('We need N_vec[0]=0 for the baseline.')

	for idx_N, N in enumerate(N_vec):
		if plot_relative:
			increase_crit_L = [(crit_L_vec[idx_N][i] - crit_L_vec[0][i]) for i in range(len(u_vec))]
			plt.plot(u_vec, increase_crit_L, color=colors[idx_N],
					 label='', alpha=0.3)
			plt.scatter(u_vec, increase_crit_L, marker=markers[idx_N],
						color=colors[idx_N], label='N = %d'%N)
		else:
			plt.plot(u_vec, crit_L_vec[idx_N], color=colors[idx_N],
					 label='', alpha=0.3)
			plt.scatter(u_vec, crit_L_vec[idx_N], marker=markers[idx_N],
						color=colors[idx_N], label='N = %d'%N)

	for idx_N, N in enumerate(N_vec):
		if distribution == 'sequential':
			s = k
		elif distribution == 'parallel':
			s = 1
		fwd_time = fwd_time_0
		if plot_relative:
			max_increase = [(speed_light * ( s/(request_rate*u_vec[i]*(u_vec[i]-1)) - fwd_time*(2*N+np.ceil(n_request*s/k)/2 ))
						  - crit_L_vec[0][i]) for i in range(len(u_vec))]
			plt.plot(u_vec, max_increase, linestyle='-.', color=colors[idx_N])#, label=r'Upper bound')

	plt.xlabel(r'Number of users $u$')
	if plot_relative:
		plt.ylabel(r'$L_{\mathrm{crit}}(N) - L_{\mathrm{crit}}(0)$ (km)')
	else:
		plt.ylabel(r'Critical distance $L_{\mathrm{crit}}$ (km)')
	plt.xlim(xlimvec)
	plt.gca().set_ylim(bottom=0)
	if y_upperlim:
		plt.gca().set_ylim(top=y_upperlim)
	#plt.yscale('log')
	xticksvec = np.arange(max(5,xlimvec[0]),xlimvec[-1]+1,5)
	plt.xticks(ticks=xticksvec, labels=xticksvec)
	ax.set_xticks(np.arange(xlimvec[0],xlimvec[-1],1), minor=True)
	ax.grid(which='both', axis='x', alpha=0.2)
	plt.legend()

	if savefig:
		filename = 'figs/PAPER-critL_%s_n%d_w%s_k%d_reqrate%.2e_splight%.2e_tfwd%s_tcontrol%s'%(distribution,
					n_request, w_request, k, request_rate, speed_light, fwd_time_0, control_time)
		if not (w_request==np.inf): # Simulation was used
			filename += '_Nsamples%d_randomseed%s'%(N_samples, randomseed)
		plt.savefig(filename+'.pdf', dpi=300, bbox_inches='tight')

		# Save the critical distances for N=0 and different number of users:
		f = open(filename+'.txt', "w")
		f.write('users = %s'%u_vec)
		for idx, N in enumerate(N_vec):
			f.write('\nL_crit(N=%d) = %s'%(N,crit_L_vec[idx]))
		f.close()
	else:
		plt.show()

def calculate_MST_vs_L_many_N(L_vec, N_vec, u, k, n_request, w_request, request_rate, speed_light, fwd_time, control_time, N_samples, randomseed, calculation_timeout=60, analytical_on=False, ji=None):

	distribution_vec = ['sequential', 'parallel']

	if analytical_on: # For some combinations of parameters, we have no analytics for the window problem
		MST_theory_vec = [ [ [[] for L in L_vec] for N in N_vec] for distr in distribution_vec]
	MST_sim_vec = [ [ [[] for L in L_vec] for N in N_vec] for distr in distribution_vec]
	MST_err_sim_vec = [ [ [[] for L in L_vec] for N in N_vec] for distr in distribution_vec]

	# Set up alarm for simulation timeout (calculation_timeout must be in seconds)
	def handler(signum, frame):
		raise Exception('Simulation timeout')
	signal.signal(signal.SIGALRM, handler)

	# CALCULATIONS
	for idx_distrib, distribution in enumerate(distribution_vec):
		for idx_N, N in enumerate(N_vec):
			for idx_L in tqdmn(range(len(L_vec)), 'N = %d (%s)'%(N,distribution),leave='False'):

				# Compute p(L,L_0)
				L = L_vec[idx_L]
				L_0 = L/(N+1)
				p = 10**(-a_eff(L_0)*(2*L)/10)

				# Compute travel time
				travel_time = 2*L/speed_light

				# Calculate MST
				try:
					signal.alarm(calculation_timeout)
					if analytical_on:
						MST_theory = QCS_theory(distribution, n_request, w_request, u, k, p,
										   request_rate, travel_time, fwd_time, N, control_time)[0]
					MST_sim, MST_err_sim = QCS_simulation(distribution, n_request, w_request, u, k, p,
									   request_rate, travel_time, fwd_time, N, control_time, N_samples, randomseed)
				except:
					if analytical_on:
						MST_theory = np.inf
					MST_sim = np.inf
					MST_err_sim = np.inf

				signal.alarm(0) # Cancel timeout

				if analytical_on:
					MST_theory_vec[idx_distrib][idx_N][idx_L] = MST_theory
				MST_sim_vec[idx_distrib][idx_N][idx_L] = MST_sim
				MST_err_sim_vec[idx_distrib][idx_N][idx_L] = MST_err_sim

	data = {'MST_sim_vec': MST_sim_vec, 'MST_err_sim_vec': MST_err_sim_vec}
	if analytical_on:
		data['MST_theory_vec'] = MST_theory_vec
	return data
	
def plot_MST_vs_L_many_N(data, L_vec, N_vec, u, k, n_request, w_request, request_rate, speed_light, fwd_time, control_time, N_samples, randomseed, analytical_on=False, logscale=False, no_markers=False, savefig=False):

	distribution_vec = ['sequential', 'parallel']

	MST_sim_vec = data['MST_sim_vec']
	MST_err_sim_vec = data['MST_err_sim_vec']
	if analytical_on:
		MST_theory_vec = data['MST_theory_vec']

	# Create figure and axes
	fig, ax1 = plt.subplots()
	cmaps = [plt.cm.get_cmap('inferno'), plt.cm.get_cmap('inferno')]
	colors = [[cmaps[idx_distrib](i/len(N_vec)) for i in range(len(N_vec))] for idx_distrib in range(len(distribution_vec))]
	markers = ['o', 's']
	if analytical_on:
		linestyles_sim = ['','']
	else:
		linestyles_sim = ['-','--']

	# Plot MST
	for idx_distrib, distribution in enumerate(distribution_vec):
		for idx_N, N in enumerate(N_vec):
			if analytical_on:
				ax1.plot(L_vec, MST_theory_vec[idx_distrib][idx_N],
						 color=colors[idx_distrib][idx_N], label='N = %d (%s, theory)'%(N,distribution))
			if no_markers:
				ax1.plot(L_vec, MST_sim_vec[idx_distrib][idx_N],
						 color=colors[idx_distrib][idx_N], linestyle=linestyles_sim[idx_distrib],
						 label='N = %d (%s)'%(N,distribution))
			else:
				ax1.errorbar(L_vec, MST_sim_vec[idx_distrib][idx_N], yerr=2*np.array(MST_err_sim_vec[idx_distrib][idx_N]),
							 color=colors[idx_distrib][idx_N], marker=markers[idx_distrib],
							 linestyle=linestyles_sim[idx_distrib], capsize=5, label='N = %d (%s)'%(N,distribution))

	# Plot specs
	plt.xlim(L_vec[0],L_vec[-1])
	plt.legend()
	plt.xlabel(r'User-hub distance $L$ (km)')
	plt.ylabel(r'Mean Sojourn Time ($\mu$s)')
	plt.ylim([10,1e6])

	if logscale:
		ax1.set_yscale('log')

	if savefig:
		filename = 'figs/RESULTS-B-MSTvsLvsN_u%d_n%d_w%s_k%d_reqrate%.2e_splight%.2e_tfwd%s_tcontrol%s_Nsamples%d_randomseed%s'%(u,
					n_request, w_request, k, request_rate, speed_light, fwd_time, control_time, N_samples, randomseed)
		if no_markers:
			filename += 'nomarkers.pdf'
		else:
			filename += '.pdf'
		plt.savefig(filename, dpi=300, bbox_inches='tight')
	else:
		plt.show()

	return
















