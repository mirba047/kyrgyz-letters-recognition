import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image

images = pd.read_csv("kg_alphabet.csv")
labels = pd.read_csv("kg_alphabet_labels.csv")

print("images: ", len(images))
print("labels: ", len(labels))

b = ["Ң", "Ө", "Ү"]
d = [0, 1, 2]

a = int(input("датасеттин ичинен канчанчы сүрөттү көргүңүз келет? "))

first_image_flat = images.iloc[a].values
first_image = first_image_flat.reshape(28, 28)

plt.imshow(first_image, cmap='gray')
plt.axis('off')
plt.title(f"Label: {labels.iloc[a].values[0]}, Name: {b[d[labels.iloc[a].values[0]]]}", fontsize=20)
plt.show()

images = images.astype("float32") / 255
labels = np.eye(len(d))[labels]

weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (72, 784))
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (3, 72))

bias_input_to_hidden = np.zeros((72, 1))
bias_hidden_to_output = np.zeros((3, 1))

epochs = 5
e_loss = 0
e_correct = 0
learning_rate = 0.01

x_train, x_test, y_train, y_test = train_test_split(images.values, labels, test_size=0.1, random_state=42)

choice = int(input("ЖИни окутууну каалайсызбы? ооба: 1, окутулган салмактарды жүктөө: 0: "))

if choice == 1:
    for epoch in range(epochs):
        print(f"Epoch №{epoch}")

        shuffle_indices = np.arange(len(x_train))
        np.random.shuffle(shuffle_indices)
        x_train = x_train[shuffle_indices]
        y_train = y_train[shuffle_indices]

        for image, label in zip(x_train, y_train):
            image = np.reshape(image, (-1, 1))
            label = np.reshape(label, (-1, 1))

            #Forward propagation (to hidden layer)
            hidden_raw = np.dot(weights_input_to_hidden, image) + bias_input_to_hidden
            hidden = 1 / (1 + np.exp(-hidden_raw))

            #Forward propagation (to output layer)
            output_raw = np.dot(weights_hidden_to_output, hidden) + bias_hidden_to_output
            output = 1 / (1 + np.exp(-output_raw))

            #Loss / Error calculation
            e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
            e_correct += int(np.argmax(output) == np.argmax(label))

            #Backpropagation (output layer)
            delta_output = output - label
            weights_hidden_to_output += -learning_rate * np.dot(delta_output, np.transpose(hidden))
            bias_hidden_to_output += -learning_rate * delta_output

            #Backpropagation (hidden layer)
            delta_hidden = np.dot(np.transpose(weights_hidden_to_output), delta_output) * (hidden * (1 - hidden))
            weights_input_to_hidden += -learning_rate * np.dot(delta_hidden, np.transpose(image))
            bias_input_to_hidden += -learning_rate * delta_hidden

            #DONE
        print(f"Loss: {round((e_loss[0] / len(x_train)) * 100, 3)}%")
        print(f"Accuracy: {round((e_correct / len(x_train)) * 100, 3)}%")
        e_loss = 0
        e_correct = 0

e_correct = 0
for image, label in zip(x_test, y_test):
    image = np.reshape(image, (-1, 1))
    label = np.reshape(label, (-1, 1))

    hidden_raw = bias_input_to_hidden +np.dot(weights_input_to_hidden, image)
    hidden = 1 / (1 + np.exp(-hidden_raw))

    output_raw = bias_hidden_to_output + np.dot(weights_hidden_to_output, hidden)
    output = 1 / (1 + np.exp(-output_raw))

    e_correct += int(np.argmax(output) == np.argmax(label))

if choice == 1:
    print(f"Test Accuracy:  {round((e_correct / len(x_test)) * 100, 3)}%")

if choice == 0:
    path = "./img/self_imgs/"

    weights_input_to_hidden = np.loadtxt("./weights/weights_input_to_hidden.csv", delimiter=",")
    weights_hidden_to_output = np.loadtxt('./weights/weights_hidden_to_output.csv', delimiter=',')
    bias_input_to_hidden = np.loadtxt('./weights/bias_input_to_hidden.csv', delimiter=',').reshape((72, 1))
    bias_hidden_to_output = np.loadtxt('./weights/bias_hidden_to_output.csv', delimiter=',').reshape((3, 1))

    run = True
    while run:
        #CHECK CUSTOM
        name_img = input(f"ЖИ-аркылуу {path} - тин ичиндеги текшерип көргүңүз келген сүрөттүн аталышын жазыңыз: ")
        test_image = plt.imread(f"{path}{name_img}", format="jpg, png")

        #Grayscale + Unit RGB + inverse colors
        gray  = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])
        test_image = 1 - (gray(test_image).astype("float32") / 255)

        

        #Reshape
        #test_image = np.reshape(test_image, (test_image.shape[0] * test_image.shape[1]))

        #Predict
        image = np.reshape(test_image, (-1, 1))

        #Forward propagation (to hidden layer)
        hidden_raw = bias_input_to_hidden + np.dot(weights_input_to_hidden, image)
        hidden = 1 / (1 + np.exp(-hidden_raw))

        #Forward propagation (to output layer)
        output_raw = bias_hidden_to_output + np.dot(weights_hidden_to_output, hidden)
        output = 1 / (1 + np.exp(-output_raw))

        plt.imshow((test_image.reshape(28, 28) * 255).astype(np.uint8), cmap="gray")
        plt.title(f"ЖИ муну: {b[output.argmax()]} тамгасы деп ойлойт")
        plt.show()

if choice == 1:

    run2 = True
    while run2:

        #CHECK CUSTOM
        name_img2 = input("ЖИ-аркылуу текшерип көргүңүз келген сүрөттүн аталышын жазыңыз: ")
        test_image = plt.imread(f"./img/self_imgs/{name_img2}", format="jpg, png")

        #Grayscale + Unit RGB + inverse colors
        gray  = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])
        test_image = 1 - (gray(test_image).astype("float32") / 255)

        

        #Reshape
        #test_image = np.reshape(test_image, (test_image.shape[0] * test_image.shape[1]))

        #Predict
        image = np.reshape(test_image, (-1, 1))

        #Forward propagation (to hidden layer)
        hidden_raw = bias_input_to_hidden + np.dot(weights_input_to_hidden, image)
        hidden = 1 / (1 + np.exp(-hidden_raw)) # sigmoid

        #Forward propagation (to output layer)
        output_raw = bias_hidden_to_output + np.dot(weights_hidden_to_output, hidden)
        output = 1 / (1 + np.exp(-output_raw))

        plt.imshow((test_image.reshape(28, 28) * 255).astype(np.uint8), cmap="gray")
        plt.title(f"ЖИ-муну: {output.argmax()} тамгасы деп ойлойт")
        plt.show()

        
        again2 = int(input("Дагы башка сүрөттөрдү текшерип көргүңүз келеби?: <<1>>-ооба, \n<<0>>-жок жана андан көрө окутулган ЖИ-нин \nсалмактарынан маанилерин сактоого өтө берейин: "))
        if again2 == 0:
            run2 = False

if choice == 1:
    save = int(input("Окутулган ЖИ-нин салмактарынын маанилерин сактоону каалайсызбы. <<1>>-ооба, <<0>>-жок: "))

    if save == 1:
        # Салмактардын CSV файлга сакталышы
        np.savetxt('./weights/weights_input_to_hidden.csv', weights_input_to_hidden, delimiter=',')
        np.savetxt('./weights/weights_hidden_to_output.csv', weights_hidden_to_output, delimiter=',')
        np.savetxt('./weights/bias_input_to_hidden.csv', bias_input_to_hidden, delimiter=',')
        np.savetxt('./weights/bias_hidden_to_output.csv', bias_hidden_to_output, delimiter=',')