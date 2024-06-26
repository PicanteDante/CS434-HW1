import numpy as np
import time

def main():
	#############################################################
	# These first bits are just to help you develop your code
	# and have expected ouputs given. All asserts should pass.
	############################################################


	#######################################
	# Now on to the real data!
	#######################################

	# Load training and test data as numpy matrices
	train_X, train_y, test_X = load_data()


	#######################################
	# Q9 Hyperparmeter Search
	#######################################

	results = []

	# Search over possible settings of k
	print("Performing 4-fold cross validation")

	best_k = 0
	best_acc = 0.0
     
	for k in [1,3,5,7,9,99,999,8000]:
		t0 = time.time()
		print("k: {}".format(k))
		#######################################
		# TODO Compute train accuracy using whole set
		#######################################
		train_y_pred = []
		for i in range(train_X.shape[0]):
			query = train_X[i]
			predicted_label = knn_classify_point(train_X, train_y, query, k)
			train_y_pred.append(predicted_label)

		train_correct = 0
		# Evaluate correctness
		for i in range(0, len(train_y)):
			if (train_y_pred[i] == train_y[i]):
				train_correct += 1
		#train_correct = np.sum(train_y_pred == train_y)
		train_acc = train_correct / len(train_y)

		print("train correct: {}".format(train_correct))
		print("len y: {}".format(len(train_y)))
		print("len trainypred: {}".format(len(train_y_pred)))

		#######################################
		# TODO Compute 4-fold cross validation accuracy
		#######################################
		val_acc, val_acc_var = cross_validation(train_X, train_y, num_folds=4, k=k)

		t1 = time.time()
		print("k = {} -- train acc = {:.2f}%  val acc = {:.2f}% ({:.4f})\t[exe_time = {:.3f}]".format(k, train_acc*100, val_acc*100, val_acc_var*100, t1-t0))
		results.append((k, train_acc, val_acc, val_acc_var, t1-t0))

		if (val_acc > best_acc):
			best_k = k
          
	for k, train_acc, val_acc, val_acc_var, exec_time in results:
          print("k = {} -- train acc = {:.2f}%  val acc = {:.2f}% ({:.4f})\t[exe_time = {:.3f}]".format(k, train_acc*100, val_acc*100, val_acc_var*100, exec_time))


	#######################################
	# Q10 Kaggle Submission
	#######################################


	# TODO set your best k value and then run on the test set
	best_k = 1

	# Make predictions on test set
	pred_test_y = predict(train_X, train_y, test_X, best_k)

	# add index and header then save to file
	test_out = np.concatenate((np.expand_dims(np.array(range(2000),dtype=int), axis=1), pred_test_y), axis=1)
	header = np.array([["id", "income"]])
	test_out = np.concatenate((header, test_out))
	np.savetxt('test_predicted.csv', test_out, fmt='%s', delimiter=',')

######################################################################
# Q7 get_nearest_neighbors 
######################################################################
# Finds and returns the index of the k examples nearest to
# the query point. Here, nearest is defined as having the 
# lowest Euclidean distance. This function does the bulk of the
# computation in kNN. As described in the homework, you'll want
# to use efficient computation to get this done. Check out 
# the documentaiton for np.linalg.norm (with axis=1) and broadcasting
# in numpy. 
#
# Input: 
#   example_set --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   query --    a 1-by-d vector representing a single example
#
#   k --        the number of neighbors to return
#
# Output:
#   idx_of_nearest --   a k-by- list of indices for the nearest k
#                       neighbors of the query point
######################################################################

def get_nearest_neighbors(example_set, query, k):
    # Compute Euclidean distances between the query point and all examples
    distances = np.linalg.norm(example_set - query, axis=1)

    # Get indices of the k nearest neighbors
    idx_of_nearest = np.argsort(distances)[:k]
    return idx_of_nearest  


######################################################################
# Q7 knn_classify_point 
######################################################################
# Runs a kNN classifier on the query point
#
# Input: 
#   examples_X --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   examples_Y --  a n-by-1 vector of example class labels
#
#   query --    a 1-by-d vector representing a single example
#
#   k --        the number of neighbors to return
#
# Output:
#   predicted_label --   either 0 or 1 corresponding to the predicted
#                        class of the query based on the neighbors
######################################################################

def knn_classify_point(examples_X, examples_y, query, k):
    # Get indices of the k nearest neighbors
    nearest_indices = get_nearest_neighbors(examples_X, query, k)
    
    # Get the labels of the nearest neighbors
    nearest_labels = examples_y[nearest_indices]
    
    # Count the occurrences of each class in the nearest neighbors
    unique_classes, counts = np.unique(nearest_labels, return_counts=True)
    
    # Find the class with the highest count
    predicted_label = unique_classes[np.argmax(counts)]
    
    return predicted_label





######################################################################
# Q8 cross_validation 
######################################################################
# Runs K-fold cross validation on our training data.
#
# Input: 
#   train_X --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   train_Y --  a n-by-1 vector of example class labels
#
# Output:
#   avg_val_acc --      the average validation accuracy across the folds
#   var_val_acc --      the variance of validation accuracy across the folds
######################################################################

def cross_validation(train_X, train_y, num_folds=4, k=1):
     n = train_X.shape[0]
     fold_size = n // num_folds
     accuracies = []

     for fold in range(num_folds):
          # Split the data into training and validation sets
          validation_start = fold * fold_size
          validation_end = (fold + 1) * fold_size
          validation_X = train_X[validation_start:validation_end]
          validation_y = train_y[validation_start:validation_end]

          #train_X_parts = [train_X[:validation_start], train_X[validation_end:]]
          #train_y_parts = [train_y[:validation_start], train_y[validation_end:]]
          # Use numpy.split to split the training data
          train_X_parts = np.split(train_X, [validation_start, validation_end])
          train_y_parts = np.split(train_y, [validation_start, validation_end])
        
          # Combine the training parts
          train_X_fold = np.vstack(train_X_parts)
          train_y_fold = np.vstack(train_y_parts)

          # Classify using k-NN
          predictions = []
          for i in range(len(validation_X)):
               query = validation_X[i]
               predicted_label = knn_classify_point(train_X_fold, train_y_fold, query, k)
               predictions.append(predicted_label)

          # Calculate accuracy
          correct = 0
          for i in range(0, len(validation_y)):
               if (predictions[i] == validation_y[i]):
                    correct += 1
          #train_correct = np.sum(train_y_pred == train_y)
          accuracy = correct / len(validation_y)

          #correct = np.sum(predictions == validation_y)
          #accuracy = correct / len(validation_y)
          accuracies.append(accuracy)
     
     avg_val_acc = np.mean(accuracies)
     var_val_acc = np.var(accuracies)
     return avg_val_acc, var_val_acc



##################################################################
# Instructor Provided Code, Don't need to modify but should read
##################################################################


######################################################################
# compute_accuracy 
######################################################################
# Runs a kNN classifier on the query point
#
# Input: 
#   true_y --  a n-by-1 vector where each value corresponds to 
#              the true label of an example
#
#   predicted_y --  a n-by-1 vector where each value corresponds
#                to the predicted label of an example
#
# Output:
#   predicted_label --   the fraction of predicted labels that match 
#                        the true labels
######################################################################

def compute_accuracy(true_y, predicted_y):
    accuracy = np.mean(true_y == predicted_y)
    return accuracy

######################################################################
# Runs a kNN classifier on every query in a matrix of queries
#
# Input: 
#   examples_X --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   examples_Y --  a n-by-1 vector of example class labels
#
#   queries_X --    a m-by-d matrix representing a set of queries 
#
#   k --        the number of neighbors to return
#
# Output:
#   predicted_y --   a m-by-1 vector of predicted class labels
######################################################################

def predict(examples_X, examples_y, queries_X, k): 
    # For each query, run a knn classifier
    predicted_y = [knn_classify_point(examples_X, examples_y, query, k) for query in queries_X]

    return np.array(predicted_y,dtype=int)[:,np.newaxis]

# Load data
def load_data():
    traindata = np.genfromtxt('train.csv', delimiter=',')[:, 1:]
    train_X = traindata[1:, :-1]
    train_y = traindata[1:, -1]
    train_y = train_y[:,np.newaxis]
    
    test_X = np.genfromtxt('test_pub.csv', delimiter=',')[1:, 1:]

    return train_X, train_y, test_X


    
if __name__ == "__main__":
    main()
