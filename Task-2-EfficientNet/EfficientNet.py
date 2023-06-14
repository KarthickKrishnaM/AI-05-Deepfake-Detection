import torch
import torchvision



efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)


for param in efficientnet.parameters():
    param.requires_grad = False

efficientnet.eval()

criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():
        for inputs, labels in testset:
            outputs = efficientnet(inputs)
            loss = criterion(outputs, labels)

            print(loss)

