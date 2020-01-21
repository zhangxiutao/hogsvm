import torch
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn
import os.path
import cnn
import numpy as np
import matplotlib.pyplot as plt

datasetPath = './dataSet/nndataset_splitted'
classes = [0,1]
# how many samples per batch to load
batch_size = 2
# percentage of training set to use as validation
test_size = 0.3
valid_size = 0.1
showBatchImages = False
train_on_gpu = torch.cuda.is_available()
# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

if __name__ == '__main__':



    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.Resize(size=(35,95)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    testData = datasets.ImageFolder(os.path.join('.',datasetPath,'test'),transform=transform)
    print(testData.size())
    test_loader = torch.utils.data.DataLoader(testData, batch_size=batch_size,
        shuffle=False, num_workers=1)

    if showBatchImages:
        # obtain one batch of training images
        dataiter = iter(test_loader)
        images, labels = dataiter.next()
        images = images.numpy() # convert images to numpy for display

        # plot the images in the batch, along with the corresponding labels
        fig = plt.figure(figsize=(10, 4))
        # display 20 images
        for idx in np.arange(2):
            ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])
            imshow(images[idx])
            ax.set_title(classes[labels[idx]])
        plt.show()

    criterion = torch.nn.CrossEntropyLoss()

    # track test loss
    test_loss = 0.0
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))


    model = cnn.Net()
    model.load_state_dict(torch.load("model_cifar.pt"))
    if train_on_gpu:      
        model.cuda()
    model.eval()

    i=1
    # iterate over test data
    #print(len(test_loader)) #how many batches
    for data, target in test_loader:
        print("target",target)
        i=i+1
        if len(target)!=batch_size:
            continue
            
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        print(data)
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        
        _, pred = torch.max(output, 1)    
        print("output",pred)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        #print(target)
        
        for i in range(batch_size):       
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # average test loss
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(2):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))