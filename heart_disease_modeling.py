#!/usr/bin/env python

from collections import defaultdict
import time
import math
from operator import add

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


from brl.antecedent_mining import *
from brl.mcmc import *
from brl.generative_model import *
from brl.utils import *
from brl.brl_methods import *

def find_brl(train_on_all_data):

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

	def convert_thallium_scan(num_string):
		if num_string == '3.0':
			return "Normal Thallium Scan"
		elif num_string == '6.0':
			return "Fixed Defect Thallium Scan"
		else:
			return "Reversable Defect Thallium Scan"


	# def convert_disease_status(num):
	# 	if num > 0:
	# 		return min(num, 4)
	# 	else:
	# 		return 0
	def convert_disease_status(num):
		if num > 0:
			return 1
		else:
			return 0


	data = pd.DataFrame.from_csv("data/uci_heartdisease_dataset/cleveland_data_14.csv")
	data = data.reset_index()

	# Need to drop rows with ?
	data = data[data.vessels_colored != '?']
	data = data[data.thallium_scan != '?']


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

	outcomes_all = data['disease_status'].values.flatten()
	data = data.drop('disease_status', 1) # Remove true predictive value from training data
	data_matrix_all = data.as_matrix()

	# *** Heart Disease-Specific Data Processing End***

	# No K-Fold
	# # Divide into training and test set
	# data_matrix_train, data_test, outcomes_train, outcome_test = train_test_split(data_matrix_all, outcomes_all, test_size=.25)

	# # Variables used for remainder of code
	# outcomes = None
	# data_matrix = None

	# if train_on_all_data:
	# 	outcomes = outcomes_all
	# 	data_matrix = data_matrix_all
	# else:
	# 	# Train with only a subset and test on rest
	# 	data_matrix = data_matrix_train
	# 	outcomes = outcomes_train


	# num_samples = len(data_matrix)

	# # FP-Growth Parameters
	# min_support_threshold = .15 # Elements that do not meet the support threshold are excluded
	# max_antecedent_length = 5 # Max length of antecedent lists to retrieve
	# number_of_possible_labels = 2

	# print("\nTraining on all data:", train_on_all_data)

	# print("\nFP-Growth Parameters")
	# print("Number of Training Samples: {}".format(num_samples))
	# print("Minimum Support Threshold: {}".format(min_support_threshold))
	# print("Max Antecedent Length: {}".format(max_antecedent_length))

	# # MCMC Parameters
	# # alpha = [1, 1, 1, 1, 1]
	# alpha = [1,1]
	# lmda = 4
	# eta = 2
	# num_iterations = 2000
	# burn_in = 1000
	# convergence_threshold = 1.05
	# confidence_interval_width = 0.95

	# # Frequent-Pattern (FP) Growth Algorithm: (brl.antecedent_mining)
	# all_antecedents = generate_antecedent_list(data_matrix, num_samples, min_support_threshold, max_antecedent_length)

	# print("Number of Antecdents Mined: {}".format(len(all_antecedents.antecedents)))
	# # MCMC - Metropolis Hastings
	# print("\nMCMC Parameters:")
	# print("Alpha", alpha)
	# print("Lambda:", lmda)
	# print("Eta:", eta)
	# print("Min Number Iterations:", num_iterations)
	# print("Burn In", burn_in)
	# print("Convergence Threshold:", convergence_threshold, "\n")

	# input("Press Enter to Begin MCMC...")

	# start = time.clock()
	# generated_mcmc_samples = brl_metropolis_hastings(num_iterations, burn_in, convergence_threshold, data_matrix, outcomes, all_antecedents, alpha, lmda, eta)
	# brl_point_list, highest_posterior_probability = find_brl_point(generated_mcmc_samples, data_matrix, outcomes, all_antecedents, alpha, lmda, eta)

	# end = time.clock()
	# print("Runtime in seconds:", end - start)

	# print("\nFP-Growth Parameters")
	# print("Number of Training Samples: {}".format(num_samples))
	# print("Minimum Support Threshold: {}".format(min_support_threshold))
	# print("Max Antecedent Length: {}".format(max_antecedent_length))
	# print("Number of Antecdents Mined: {}".format(len(all_antecedents.antecedents)))
	# # MCMC - Metropolis Hastings
	# print("\nMCMC Parameters:")
	# print("Alpha", alpha)
	# print("Lambda:", lmda)
	# print("Eta:", eta)
	# print("Min Number Iterations:", num_iterations)
	# print("Burn In", burn_in)
	# print("Convergence Threshold:", convergence_threshold, "\n")

	# # Generate the N's for each posterior using the data
	# N_posterior = generate_N_bold_posterior(data_matrix, outcomes, brl_point_list, number_of_possible_labels)

	# print("N_posterior:")
	# print(N_posterior)
	# print("\n")

	# print_posterior_antecedent_list_results(N_posterior, brl_point_list, confidence_interval_width, alpha)


	# # Evaluate the BRL on the test set
	# make_brl_test_set_predictions(data_test, outcome_test, N_posterior, brl_point_list, alpha, 0.5)
	# find_auc(data_test, outcome_test, N_posterior, brl_point_list, alpha)

	# return N_posterior, brl_point_list

	# K-fold Stuff
	kf = KFold(4)
	scores = []
	aucs = []

	features = data_matrix_all

	for train_index, test_index in kf.split(features):

		# Divide into training and test set
		# data_matrix_train, data_test, outcomes_train, outcome_test = train_test_split(data_matrix_all, outcomes_all, test_size=.25)

		data_matrix, data_test = features[train_index], features[test_index]
		outcomes, outcome_test = outcomes_all[train_index], outcomes_all[test_index]

		num_samples = len(data_matrix)

		# FP-Growth Parameters
		min_support_threshold = .25 # Elements that do not meet the support threshold are excluded
		max_antecedent_length = 5 # Max length of antecedent lists to retrieve
		number_of_possible_labels = 2

		print("\nTraining on all data:", train_on_all_data)
		print("\nFP-Growth Parameters")
		print("Number of Training Samples: {}".format(num_samples))
		print("Minimum Support Threshold: {}".format(min_support_threshold))
		print("Max Antecedent Length: {}".format(max_antecedent_length))


		# MCMC Parameters
		# alpha = [1, 1, 1, 1, 1]
		alpha = [1,1]
		lmda = 5
		eta = 2
		num_iterations = 2000
		burn_in = 1000
		convergence_threshold = 1.05
		confidence_interval_width = 0.95

		# Frequent-Pattern (FP) Growth Algorithm: (brl.antecedent_mining)
		all_antecedents = generate_antecedent_list(data_matrix, num_samples, min_support_threshold, max_antecedent_length)

		print("Number of Antecdents Mined: {}".format(len(all_antecedents.antecedents)))
		# MCMC - Metropolis Hastings
		print("\nMCMC Parameters:")
		print("Alpha", alpha)
		print("Lambda:", lmda)
		print("Eta:", eta)
		print("Min Number Iterations:", num_iterations)
		print("Burn In", burn_in)
		print("Convergence Threshold:", convergence_threshold, "\n")

		# input("Press Enter to Begin MCMC...")

		start = time.clock()
		generated_mcmc_samples = brl_metropolis_hastings(num_iterations, burn_in, convergence_threshold, data_matrix, outcomes, all_antecedents, alpha, lmda, eta)
		brl_point_list, highest_posterior_probability = find_brl_point(generated_mcmc_samples, data_matrix, outcomes, all_antecedents, alpha, lmda, eta)

		end = time.clock()
		print("Runtime in seconds:", end - start)

		# print("\nFP-Growth Parameters")
		# print("Number of Training Samples: {}".format(num_samples))
		# print("Minimum Support Threshold: {}".format(min_support_threshold))
		# print("Max Antecedent Length: {}".format(max_antecedent_length))
		# print("Number of Antecdents Mined: {}".format(len(all_antecedents.antecedents)))
		# # MCMC - Metropolis Hastings
		# print("\nMCMC Parameters:")
		# print("Alpha", alpha)
		# print("Lambda:", lmda)
		# print("Eta:", eta)
		# print("Min Number Iterations:", num_iterations)
		# print("Burn In", burn_in)
		# print("Convergence Threshold:", convergence_threshold, "\n")

		# Generate the N's for each posterior using the data
		N_posterior = generate_N_bold_posterior(data_matrix, outcomes, brl_point_list, number_of_possible_labels)

		print("N_posterior:")
		print(N_posterior)
		print("\n")

		print_posterior_antecedent_list_results(N_posterior, brl_point_list, confidence_interval_width, alpha)


		# Evaluate the BRL on the test set
		fpr, tpr, accuracy = make_brl_test_set_predictions(data_test, outcome_test, N_posterior, brl_point_list, alpha, 0.5)
		auc = find_auc(data_test, outcome_test, N_posterior, brl_point_list, alpha)

		scores.append(accuracy)
		aucs.append(auc)

	print("Accuracy Average:", sum(scores) / len(scores))
	print("AUC Average:", sum(aucs) / len(aucs))

	return scores, aucs

def dummy():
	make_brl_test_set_predictions(data_matrix_all, outcomes_all, N_posterior, brl_point_list, [1,1], 0.5)
	find_auc(data_matrix_all, outcomes_all, N_posterior, brl_point_list, [1,1])


if __name__=='__main__':

	train_on_all_data = False
	N_posterior, brl_point_list = find_brl(train_on_all_data)


