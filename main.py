import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features) - 2)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores * 2)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)
      
class PrunableNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(32*32*3, 512)
        self.relu = nn.ReLU()
        self.fc2 = PrunableLinear(512, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def sparsity_loss(self):
        g1 = torch.sigmoid(self.fc1.gate_scores * 2)
        g2 = torch.sigmoid(self.fc2.gate_scores * 2)

        return (g1.mean() + g2.mean()) * 20
      
def train_model(lam, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64)

    model = PrunableNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
  
    for epoch in range(epochs):
        model.train()

        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            out = model(x)
            loss_cls = criterion(out, y)
            loss_sparse = model.sparsity_loss()

            loss = loss_cls + lam * loss_sparse

            loss.backward()
            optimizer.step()

        print(f"Lambda {lam} | Epoch {epoch+1} | Loss: {loss.item():.4f}")

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            out = model(x)

            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()

    g1 = torch.sigmoid(model.fc1.gate_scores * 2).detach().cpu().numpy().flatten()
    g2 = torch.sigmoid(model.fc2.gate_scores * 2).detach().cpu().numpy().flatten()
    all_gates = np.concatenate([g1, g2])

    accuracy = 100 * correct / total
    sparsity = np.mean(all_gates < 0.05) * 100

  print("\nGate Stats:")
    print("Min:", all_gates.min())
    print("Max:", all_gates.max())
    print("Mean:", all_gates.mean())

    return accuracy, sparsity, all_gates

lambdas = [0.01, 0.1, 0.5]
results = []

for lam in lambdas:
    acc, sp, gates = train_model(lam)
    results.append((lam, acc, sp))

    print(f"\nLambda: {lam}")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Sparsity: {sp:.2f}%")

print("\nFinal Results:")
print("Lambda\tAccuracy\tSparsity")
for lam, acc, sp in results:
    print(f"{lam}\t{acc:.2f}%\t\t{sp:.2f}%")
  
plt.hist(gates, bins=50)
plt.title("Gate Value Distribution")
plt.xlabel("Gate Value")
plt.ylabel("Count")
plt.show()
