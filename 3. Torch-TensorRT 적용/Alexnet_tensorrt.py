# Import libraries
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms

# gpu available
torch.cuda.is_available()

#Check device: GPU(cuda) or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

# dataset upload
# data augmentation
# data normalization
transform_train = transforms.Compose([ # train set loader
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_val = transforms.Compose([ # validation set loader
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


transform_test = transforms.Compose([ # test set loader
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
val_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_val)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# split the training set into training and validation sets
train_size = int(0.9 * len(train_set)) # use only 5000 images for validation
val_size = len(train_set) - train_size
train_set, valset = torch.utils.data.random_split(train_set, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_set, batch_size= 128, shuffle=True) # use shuffle only for train
val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

# sample image plot
def im_convert(tensor):
  image = tensor.clone().detach().numpy()
  image = image.transpose(1, 2, 0)
  image = image * np.array([0.5, 0.5, 0.5] + np.array([0.5, 0.5, 0.5]))
  image = image.clip(0, 1)
  return image


CLASS_NAMES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# sample test image plot (normalized ones)
data_iter = iter(test_loader)
images, labels = next(data_iter)

fig = plt.figure(figsize=(25, 4))

for i in np.arange(20): # normalized test images
  # row 2 column 10
  ax = fig.add_subplot(2, 10, i+1, xticks=[], yticks=[])
  plt.imshow(im_convert(images[i]))
  ax.set_title(CLASS_NAMES[labels[i].item()])


# Define the AlexNet model
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


model = AlexNet().to(device) # use adam
optimizer = optim.Adam(model.parameters(), lr=0.001) # set learning rate as 0.001

# Train the model
criterion = nn.CrossEntropyLoss()
num_epoch = 10
#PATH =  "/content/gdrive/MyDrive/alexNet_pytorch.pt"
import math
total_batches = math.ceil(len(train_set) / 128)
total_steps = num_epoch * total_batches


best_val_acc = 0.0  # initialize to lowest possible value
for epoch in range(num_epoch):
    # Train the modela
    train_loss = 0.0
    train_acc = 0.0
    model.train()  # set model to training mode

    for i, (images, labels) in enumerate(train_loader):
        # Move images and labels to device (GPU)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() # zero out the gradients
        outputs = model(images) # forward pass
        loss = criterion(outputs, labels) # compute loss

        loss.backward() # backward pass
        optimizer.step() # update weights

        train_loss += loss.item() * images.size(0) # calculate training loss and accuracy
        _, preds = torch.max(outputs, 1)
        train_acc += torch.sum(preds == labels.data)

        if (i + 1) % 100 == 0: # print status update every 100 batches
            print(f'Epoch [{epoch+1}/{num_epoch}], Step [{i+1}/{total_steps}], '
                  f'Training Loss: {train_loss / ((i+1)*128):.4f}, Training Accuracy: {train_acc / ((i+1)*128):.4f}')

    # Validate the model
    val_loss = 0.0
    val_acc = 0.0
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            # Move images and labels to device (GPU)
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Calculate validation loss and accuracy
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_acc += torch.sum(preds == labels.data)

    # Calculate average losses and accuracies
    train_loss = train_loss / len(train_set)
    train_acc = train_acc / len(train_set)
    val_loss = val_loss / len(val_set)
    val_acc = val_acc / len(val_set)

    # Print epoch summary
    print(f'Epoch [{epoch+1}/{num_epoch}], '
          f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}, '
          f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

    # Save the best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

# Function to plot images along with their predictions
def plot_predictions(images, labels, predictions):
    fig, axs = plt.subplots(1, len(images), figsize=(25, 4))
    for i in range(len(images)):
        image = images[i] / 2 + 0.5     # Unnormalize the image
        image = image.permute(1, 2, 0)   # Transpose to (height, width, channels)
        axs[i].imshow(image)
        axs[i].set_title(f"True: {CLASS_NAMES[labels[i]]}\nPred: {CLASS_NAMES[predictions[i]]}")
        axs[i].axis('off')
    plt.show()


# Get a batch of test images and their labels
data_iter = iter(test_loader)
images, labels = next(data_iter)

# Move the images and labels to the device
images, labels = images.to(device), labels.to(device)
sample_images = images[:10] # 10 sample image
sample_labels = labels[:10]
# Make predictions on the batch of images
outputs = test_model(sample_images)
_, predicted = torch.max(outputs, 1)

# Plot the images and their predictions
plot_predictions(sample_images.cpu(), sample_labels.cpu(), predicted.cpu())

# Using Torch-TensorRT
import torch_tensorrt
import torchvision.models as models

original_model = AlexNet().half().eval().to("cuda")

optimized_model = torch_tensorrt.compile(original_model,
    inputs = [torch_tensorrt.Input(
            min_shape=[1, 3, 112, 112],
            opt_shape=[1, 3, 224, 224],
            max_shape=[1, 3, 448, 448],
            dtype=torch.half)
    ],
    enabled_precisions = {torch.half}, # Run with FP16
)
torch.jit.save(optimized_model, "trt_ts_module.ts") # Save the Model

correct = 0
total = 0
test_model = AlexNet().to(device)
test_model.load_state_dict(torch.load("best_model.pth"))

model.eval()
model = model.cuda()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
f32_data = torch.ones([1, 3, 224, 224]).cuda()
f16_data = torch.ones([1, 3, 224, 224], dtype=torch.half).cuda()
inference_time_total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)

        start.record()
        # outputs = test_model(f32_data)
        outputs = optimized_model(f16_data)
        end.record()

        print(outputs)
        
        torch.cuda.synchronize()    

        inference_time = start.elapsed_time(end)
        inference_time_total += inference_time

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Total inference time for the test set: %.5f seconds' % (inference_time_total / 1000))

