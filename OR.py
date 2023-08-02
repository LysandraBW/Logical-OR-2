from Activation import _Sigmoid
from Loss import _MSE
from Network import *

data = [
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0]
]

decision = input("Train and Test New Network [A] or Test Pre-Loaded [B] Network: ")

network = None

if decision == "A":
    network = Network([2, 2, 1])
    network.set_activation(all=_Sigmoid)
    network.set_cost(_MSE)
    network.train(data, lambda a: a[0:2], lambda a: [a[2]], 1000, 0.1)
    network.test(data, lambda a: a[0:2], lambda a: [a[2]], lambda predicted, target: round(predicted[0]) == target[0])

elif decision == "B":
    network = Network(None, "LogicalORTuner.txt")
    network.test(data, lambda a: a[0:2], lambda a: [a[2]], lambda predicted, target: round(predicted[0]) == target[0])

else:
    exit(0)

# This may not be the most readable and clean, but it's largely unimportant in the grand scheme of the multiverse.
while True:
    operands = input("\nEnter 2 Digits: ")
    inputs = operands.split(" ")
    for index in range(len(inputs)):
        inputs[index] = float(inputs[index])
    print(f"Output: {network.get_output(inputs)}")
    exit_test = input("Enter Any Character to Exit: ")
    if exit_test != "":
        break