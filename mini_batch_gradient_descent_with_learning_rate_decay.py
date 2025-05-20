from dnn_lib import *

# Load data
train_X, train_Y = load_dataset_moons()

# Set the model layers dimensions
layers_dims = [train_X.shape[0], 5, 2, 1]

# Train the model
parameters, costs = train_deep_fully_connected_model(train_X, train_Y, layers_dims, learning_rate=0.07, num_iterations=5000, mini_batch_size=64, learning_rate_decay_rate=0.0001, print_cost=True)
plot_costs(costs, learning_rate=0.07)

# Check the train results
train_predictions = predict(train_X, parameters, 0.5)
train_accuracy = calculate_accuracy(train_predictions, train_Y)
print ("Training set accuracy: {}".format(train_accuracy))

# Visualise train predictions
plot_decision_boundary(parameters, train_X, train_Y, padding=0.1)