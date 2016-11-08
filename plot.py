import matplotlib.pyplot as plt
import numpy as np
import re

with open('logs.txt') as f:
    content = f.readlines()
train_loss = []
validation_loss = []

for i in content:
    if 'training loss:' in i:
        train_loss.append(re.findall("\d+\.\d+", i))
    elif 'validation loss:' in i:
        validation_loss.append(re.findall("\d+\.\d+", i))

x = range(1,201)
labels = ['100', '1000', '10000', 'All(~97000)']



plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Effect of epochs on shallow depth CNN')

plt.plot(x, train_loss, color='r', label='training_loss')
plt.plot(x, validation_loss, color='b', label='validation_loss')
plt.legend(loc=1)

plt.savefig("plots/shallow_depth_model.png")
plt.show()

