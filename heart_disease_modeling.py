#!/usr/bin/env python
from brl.antecedent_mining import *
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from collections import defaultdict
from brl.mcmc import *
from brl.generative_model import *
from brl.utils import *
import time
import math
from operator import add

def find_brl():

    # *** Heart Disease-Specific Data Processing Start***

	def convert_age(age):
		if age < 40:
			return "Age < 40"
		elif age < 50:
			return "40 < Age < 50"
		elif age < 55:
			return "50 < Age < 55"    	
		elif age < 60:
			return "55 < Age < 60"
		elif age < 65:
			return "60 < Age < 65"
		else:
			return "Age > 65"

	def convert_sex(sex_num):
		if sex_num==1:
			return "Male"
		else:
			return "Female"

	def convert_chest_pain_type(type_num):
		if type_num==1:
			return "Typical Angina"
		elif type_num==2:
			return "Atypical Angina"
		elif type_num==3:
			return "Non-Anginal Pain "
		else:
			return "Asymptomatic"

	def convert_resting_blood_pressure(pressure):
		if pressure < 120:
			return "Blood Pressure < 120"
		elif pressure < 140:
			return "120 < Blood Pressure < 140"
		else:
			return "Blood Pressure > 140"   	

	def convert_cholesterol(cholesterol):

		if cholesterol < 200:
			return "Cholesterol < 200"
		elif cholesterol < 250:
			return "200 < Cholesterol < 250"
		elif cholesterol < 300:
			return "250 < Cholesterol < 300"
		else:
			return "Cholesterol > 300"

	def convert_fasting_blood_sugar(num):
		if num == 1:
			return "Fasting blood sugar MORE than 120 mg/dl"
		else:
			return "Fasting blood sugar LESS than 120 mg/dl"

	def convert_ecg(ecg_num):
		if ecg_num == 0:
			return "Normal ECG"
		elif ecg_num == 1:
			return "ECG: ST-T Wave Abnormality"
		else:
			return "ECG: Left Ventricular Hypertrophy"

	def convert_max_heartrate(heartrate):
		if heartrate < 130:
			return "Heartrate < 130"
		elif heartrate < 150:
			return "130 < Heartrate < 150"
		elif heartrate < 170:
			return "150 < Heartrate < 170"
		else:
			return "Heartrate > 170"

	def convert_exercise_induced_angina(num):
		if num==0:
			return "No Exercise Induced Angina"
		else:
			return "Exercise Induced Angina"

	def convert_st_depression_ecg(num):

		if num == 0:
			return "ST Depression == 0"
		elif num < 1:
			return "0 < ST Depression < 1"
		elif num < 2:
			return "1 < ST Depression < 2"
		else:
			return "ST Depression > 2"

	def convert_peak_st_slope(num):
		if num==1:
			return "Upsloping ST-segment"
		elif num==2:
			return "Flat ST-segment"
		else:
			return "Downsloping ST-segment"

	def convert_vessels_colored(num):
		return "{} Vessels Colored (Flourosopy)".format(num)

	def convert_thallium_scan(num):
		if num == 3:
			return "Normal Thallium Scan"
		elif num == 6:
			return "Fixed Defect Thallium Scan"
		else:
			return "Reversable Defect Thallium Scan"


	def convert_disease_status(num):
		if num > 0:
			return 1
		else:
			return 0


	data = pd.DataFrame.from_csv("data/uci_heartdisease_dataset/cleveland_data_14.csv")
	data = data.reset_index()

	# Turning data into processable antecedents 
	data["age"] = data["age"].apply(convert_age)
	data["sex"] = data["sex"].apply(convert_sex)
	data["pain_type"] = data["pain_type"].apply(convert_chest_pain_type)
	data["resting_blood_pressure"] = data["resting_blood_pressure"].apply(convert_resting_blood_pressure)
	data["cholesterol"] = data["cholesterol"].apply(convert_cholesterol)
	data["fasting_blood_sugar"] = data["fasting_blood_sugar"].apply(convert_fasting_blood_sugar)
	data["resting_ecg"] = data["resting_ecg"].apply(convert_ecg)
	data["max_hr"] = data["max_hr"].apply(convert_max_heartrate)
	data["exercise_induced_angina"] = data["exercise_induced_angina"].apply(convert_exercise_induced_angina)
	data["st_depression_ecg"] = data["st_depression_ecg"].apply(convert_st_depression_ecg)
	data["peak_st_slope"] = data["peak_st_slope"].apply(convert_peak_st_slope)
	data["vessels_colored"] = data["vessels_colored"].apply(convert_vessels_colored)
	data["thallium_scan"] = data["thallium_scan"].apply(convert_thallium_scan)
	data["disease_status"] = data["disease_status"].apply(convert_disease_status)

	# Getting the disease statuses
	outcomes = data['disease_status'].values.flatten()

	# *** Heart Disease-Specific Data Processing End***

	data_matrix = data.as_matrix()
	num_samples = len(data_matrix)

	# FP-Growth Parameters
	min_support_threshold = .1 # Elements that do not meet the support threshold are excluded
	max_antecedent_length = 3 # Max length of antecedent lists to retrieve
	number_of_possible_labels = 2

	# MCMC Parameters
	alpha = [1,1]
	lmda = 3
	eta = 1
	num_iterations = 2000
	burn_in = 0
	convergence_threshold = 1.05
	confidence_interval_width = 0.95

	# Frequent-Pattern (FP) Growth Algorithm: (brl.antecedent_mining)
	all_antecedents = generate_antecedent_list(data_matrix, num_samples, min_support_threshold, max_antecedent_length)

	# MCMC - Metropolis Hastings
	print("\nMCMC Parameters:")
	print("Alpha", alpha, "\n", "Lambda:", lmda, "\n", "Eta:", eta, "\n", "Min Number Iterations:", num_iterations, "\n", "Burn In", burn_in, "\n", "Convergence Threshold:", convergence_threshold, "\n")

	start = time.clock()
	generated_mcmc_samples = brl_metropolis_hastings(num_iterations, burn_in, convergence_threshold, data_matrix, outcomes, all_antecedents, alpha, lmda, eta)
	brl_point_list, highest_posterior_probability = find_brl_point(generated_mcmc_samples, data_matrix, outcomes, all_antecedents, alpha, lmda, eta)

	end = time.clock()
	print("Runtime in seconds:", end - start)

	# Generate the N's for each posterior using the data
	N_posterior = generate_N_bold_posterior(data_matrix, outcomes, brl_point_list, number_of_possible_labels)

	print("N_posterior:")
	print(N_posterior)
	print("\n")

	print_posterior_antecedent_list_results(N_posterior, brl_point_list, confidence_interval_width, alpha)

	# lower_bound, upper_bound = compute_dirichlet_confidence_interval(N_posterior[1], 0)
	# brl_point_predict(data_matrix[0], N_posterior, brl_point_list, alpha)

	return N_posterior, brl_point_list


if __name__=='__main__':
	N_posterior, brl_point_list = find_brl()
