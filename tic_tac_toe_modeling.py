from brl.antecedent_mining import *
from brl.mcmc import *
from brl.generative_model import *
from brl.utils import *
from brl.brl_methods import *

def convert_top_left(val):
    if val=='x':
        return "top_left_x"
    elif val=='o':
        return "top_left_o"
    else:
        return "top_left_blank"
    
def convert_top_middle(val):
    if val=='x':
        return "top_middle_x"
    elif val=='o':
        return "top_middle_o"
    else:
        return "top_middle_blank"

def convert_top_right(val):
    if val=='x':
        return "top_right_x"
    elif val=='o':
        return "top_right_o"
    else:
        return "top_right_blank"
    
def convert_middle_left(val):
    if val=='x':
        return "middle_left_x"
    elif val=='o':
        return "middle_left_o"
    else:
        return "middle_left_blank"

def convert_middle_middle(val):
    if val=='x':
        return "middle_middle_x"
    elif val=='o':
        return "middle_middle_o"
    else:
        return "middle_middle_blank"

def convert_middle_right(val):
    if val=='x':
        return "middle_right_x"
    elif val=='o':
        return "middle_right_o"
    else:
        return "middle_right_blank"

def convert_bottom_left(val):
    if val=='x':
        return "bottom_left_x"
    elif val=='o':
        return "bottom_left_o"
    else:
        return "bottom_left_blank"

def convert_bottom_middle(val):
    if val=='x':
        return "bottom_middle_x"
    elif val=='o':
        return "bottom_middle_o"
    else:
        return "bottom_middle_blank"

def convert_bottom_right(val):
    if val=='x':
        return "bottom_right_x"
    elif val=='o':
        return "bottom_right_o"
    else:
        return "bottom_right_blank"
    
def convert_result(val):
    if val=='positive':
        return 1
    else:
        return 0


def find_brl():

    data = pd.DataFrame.from_csv("data/tictactoe_dataset/tic_tac_toe.csv")
    data = data.reset_index()

    # Convert features to distinguishing ones, result to numerical label
    data['top_left'] = data['top_left'].apply(convert_top_left)
    data['top_middle'] = data['top_middle'].apply(convert_top_middle)
    data['top_right'] = data['top_right'].apply(convert_top_right)
    data['middle_left'] = data['middle_left'].apply(convert_middle_left)
    data['middle_middle'] = data['middle_middle'].apply(convert_middle_middle)
    data['middle_right'] = data['middle_right'].apply(convert_middle_right)
    data['bottom_left'] = data['bottom_left'].apply(convert_bottom_left)
    data['bottom_middle'] = data['bottom_middle'].apply(convert_bottom_middle)
    data['bottom_right'] = data['bottom_right'].apply(convert_bottom_right)
    data['result'] = data['result'].apply(convert_result)


    outcomes = data['result'].values.flatten()
    data = data.drop('result', 1) # Remove true predictive value from training data
    data_matrix = data.as_matrix()

    num_samples = len(data_matrix)

    # FP-Growth Parameters
    min_support_threshold = .08 # Elements that do not meet the support threshold are excluded
    max_antecedent_length = 3 # Max length of antecedent lists to retrieve
    number_of_possible_labels = 2

    print("\nTraining on all data")

    print("\nFP-Growth Parameters")
    print("Number of Training Samples: {}".format(num_samples))
    print("Minimum Support Threshold: {}".format(min_support_threshold))
    print("Max Antecedent Length: {}".format(max_antecedent_length))

    # MCMC Parameters
    alpha = [1,1]
    lmda = 8
    eta = 3
    num_iterations = 2000
    burn_in = 1000
    convergence_threshold = 1.08
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

    input("Press Enter to Begin MCMC...")

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

    print_posterior_antecedent_list_results(N_posterior, brl_point_list, confidence_interval_width, [0,0])


    # Evaluate the BRL on the test set
    make_brl_test_set_predictions(data_matrix, outcomes, N_posterior, brl_point_list, [0,0])


    return N_posterior, brl_point_list




if __name__=='__main__':

	N_posterior, brl_point_list = find_brl()



