import os
import torch
import torchvision
from torchvision import transforms
from codecarbon import EmissionsTracker
import sys
import numpy as np
from torch.optim import lr_scheduler
from tqdm import tqdm
import numpy as np

current_path = os.getcwd()
#Takes as input the first argument given when running the script, which is an ID that will define the threshold and the number of the experiment for a given threshold
id = int(sys.argv[1])
#Generation of 30 thresholds, linearly distributed
thresholds = np.linspace(0.1, 0.95, num=30)
#Each tuple is a threshold and the number of the experiment associated to this threshold (1 to 5)
threshold_tuples = [(th, exp_num) for th in thresholds for exp_num in range(1, 6)]
id_tuple = threshold_tuples[id]
threshold_value = id_tuple[0]
exp_num = id_tuple[1]

#Preparation of dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.CIFAR10(root = './',train=True, download=True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size= 64, shuffle=True, num_workers=4)#data loader pour les batch, multiples de 8 pour le batch size

testset = torchvision.datasets.CIFAR10(root = './', train=False, download=True, transform = transform)#transform pour data augmentation
testloader = torch.utils.data.DataLoader(testset, batch_size= 64, shuffle=False, num_workers=4)#num_workers jamais utiliser sur Windows

#Go to the GitHub repository to import the models
os.chdir("./senet")

from senet.baseline import resnet20
from senet.se_resnet import se_resnet20
from senet.se_module import SELayer

#Instantiate the model (add "reduction = 8" as second argument to change the reduction ratio)
model = se_resnet20(num_classes=10, threshold = threshold_value).to("cuda")

# Instantiating the optimizer and scheduler
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1 , momentum=0.9, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.33)

# Loss function
loss_fcn = torch.nn.CrossEntropyLoss()

# Metrics
def torch_acc(y_pred, y_true):
    train_acc = (torch.argmax(y_pred, dim=1) == y_true).float().mean()
    return train_acc

device = "cuda"
#Initialisation of lists containing training and testing accuracies
list_acc_train = []
list_acc_test = []
#Setting the output files names (according to the id given as input)
file_name_carbon = "emissions_"+str(id)+".csv"
file_name_results = current_path +"/Output_"+str(id)+".txt"
#Opening the output file and writing the threshold and experiment number for this threshold
fh = open(file_name_results, 'a')
fh.write("Threshold: %f\n" % threshold_value)
fh.write("Experience number: %d\n\n" % exp_num)

#Running 100 epochs
for epoch in range (100):
    #Giving a name to each epoch to log energy consumption by epoch
    pjct_name = "Epoch_"+str(epoch)

    #Instantiate and start the energy tracker for this epoch
    if epoch == 0:
        tracker = EmissionsTracker(project_name=pjct_name, output_dir=current_path, output_file=file_name_carbon, on_csv_write = "update", allow_multiple_runs=True)
    else:
        tracker = EmissionsTracker(project_name=pjct_name, output_dir=current_path, output_file=file_name_carbon, on_csv_write = "append", allow_multiple_runs=True)

    tracker.start()

    #Training phase
    model.train()

    train_loss = 0
    train_acc = 0
    layers_training = []
    fh.write("Epoch:%d\n"%epoch)
    for x, y in trainloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fcn(pred, y)
        train_loss += loss.item()
        train_acc += torch_acc(pred, y)
        loss.backward()
        optimizer.step()

    #For each SE layer, get the history of percentages of switched off layers across all images of the training dataset and reset the history for the next epoch
    for m in model.modules():
        if isinstance(m, SELayer):
            layers_training.append(m.history)
            m.history = []

    #For each SE layer, average the percentages over all training images to get a global percentage of switched off channels for each SE layer. Results for each SE layer are written in the output file.
    nb_layer = 0
    for layer in layers_training:
        nb_layer += 1
        layer_score = np.average(layer)
        fh.write("Percentage of switched off channels for layer %d at training: %.3f%%\n"%(nb_layer, layer_score*100))

    train_acc /= len(trainloader)

    #Testing phase
    model.eval()
    test_loss = 0
    test_acc = 0
    layers_testing = []
    for x, y in testloader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fcn(pred, y)
        test_loss += loss.item()
        test_acc += torch_acc(pred, y)

    #For each SE layer, get the history of percentages of switched off layers across all images of the testing dataset and reset the history for the next epoch
    for m in model.modules():
        if isinstance(m, SELayer):
            layers_testing.append(m.history)
            m.history = []

    #For each SE layer, average the percentages over all testing images to get a global percentage of switched off channels for each SE layer. Results for each SE layer are written in the output file.
    nb_layer = 0
    for layer in layers_testing:
        nb_layer += 1
        layer_score = np.average(layer)
        fh.write("Percentage of switched off channels for layer %d at testing: %.3f%%\n"%(nb_layer, layer_score*100))
        
    test_acc /= len(testloader)
    scheduler.step()

    #Writing training and testing loss and accuracy in the output file
    fh.write("Training loss:%f\n"% train_loss)
    fh.write("Training accuracy:%f\n"% train_acc)
    fh.write("Test loss:%f\n"% test_loss)
    fh.write("Test accuracy:%f\n\n"% test_acc)
    list_acc_train.append(train_acc)
    list_acc_test.append(test_acc)

    #Stop the energy consumption tracker for the epoch. The results will be logged in the emissions output file.
    tracker.stop()

#Write maximum training and testing accuracies across all epochs in the output file
fh.write("Max training accuracy:%f\n"% max(list_acc_train))
fh.write("Max testing accuracy:%f\n"% max(list_acc_test))
fh.close()



