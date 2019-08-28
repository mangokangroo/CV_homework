import numpy as np
import random


# Generate random test data
def gen_sample_data():
    w = random.randint(0, 10) + random.random()
    b = random.randint(0, 5) + random.random()
    num_samples = 100
    x_list = []
    y_list = []
    for i in range(num_samples):
        x = random.randint(0, 100) * random.random()
        y = w * x + b + random.random() * random.randint(-1, 1)
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list, w, b


# Preprocess data before gradient descent: add a column of 1, and convert to ndarray
def process_train_data(inX_list, inY_list):
    x_out = np.ones((len(inX_list), 2))
    x_out[:, 1] = inX_list
    x_out = np.mat(x_out)
    y_out = np.mat([inY_list]).transpose()
    return x_out, y_out


# Gradient descent -- Use VECTORIZATION to reduce the use of "for loop"
def gradient_descent(x_train, gt_y_list, lr, max_iter):
    theta = np.ones((2, 1))
    x, gt_y = process_train_data(x_train, gt_y_list)
    # Iterate to perform gradient descent
    for i in range(max_iter):
        # Evaluation of the model at each point, in VECTOR form
        eval_y_vector = x * theta
        # Deviation of the evaluation from the ground-truth value at each point
        cost = eval_y_vector - gt_y
        # Loss
        loss = np.sum(cost)
        gradient = np.dot(np.transpose(x), cost)
        theta = theta - lr * gradient
        print('w:{0}, b:{1}'.format(theta[1], theta[0]))
        print('loss is {0}'.format(loss))
        if abs(loss) <= 0.01:
            print("\nTotal number of iteration is: " + str(i) + '\n')
            break
    return theta, loss



def run():
    x_list, y_list, w, b = gen_sample_data()
    lr = 0.00001
    max_iter = 30000
    model, loss = gradient_descent(x_list, y_list, lr, max_iter)
    w_train = model[1]
    b_train = model[0]
    # Output to monitor
    print('Sample values of w, b are:')
    print('w:{0}, b:{1}'.format(w, b))
    print('Trained values of w, b are')
    print('w_train:{0}, b_train:{1}'.format(w_train, b_train))
    print('Final loss is: ' + str(loss))


if __name__ == '__main__':
    run()
