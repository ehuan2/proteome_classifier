import torch
import json
# Load the tensor from the file
loaded_tensor = torch.load('example.pt')

k = 5
# Print the shape of the tensor
print(loaded_tensor.shape, 20**k)
print(torch.sum(loaded_tensor))

seq = 'MAFSAEDVLKEYDRRRRMEALLLSLYYPND'
print(len(seq) - k + 1)

# Read the JSON file and create a PyTorch tensor
with open('mapping.json', 'r') as f:
    data = json.load(f)

indices = set()

def get_index(partial_seq):
    index = 0
    for i in range(k):
        index += data[partial_seq[i]] * 20**(k - i - 1)
    print(f'Index: {index}')
    print(loaded_tensor[index])

    if index in indices:
        return 0
    
    indices.add(index)

    return loaded_tensor[index]

count = 0

for i in range(len(seq) - k + 1):
    count += get_index(seq[i:])

print(count)
