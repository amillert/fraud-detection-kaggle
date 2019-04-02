from preprocessing.preprocess_data import CustomDataset
import numpy as np
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __name__ == "__main__":
    dataset = CustomDataset()

    # TODO:
    #  - implement arg parser and obtain all the consts from console

    DATA_SIZE = len(dataset)
    BATCH_SIZE = 512

    print(f"Amount of data read: {DATA_SIZE}")
    batches = DataLoader(dataset=dataset, batch_size=512, shuffle=True, num_workers=4)
    print('Creating batches done')

    # TODO:
    #  - implement naive model
    #  - implement negative sampling model and loss function

    model = ...
    EPOCHS = 100
    ETA = 0.001
    MOMENTUM = 0.9

    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=ETA, momentum=MOMENTUM)

    num_batches = np.ceil(DATA_SIZE / BATCH_SIZE)
    print('Upcoming batches: {num_batches}')

    for epoch in range(EPOCHS):
        tick = time.time()
        loss_total = 0.0
        batch_count = 0
        for x, y in batches:
            batch_count += 1
            naive_loss = criterion(model(x), y)
            loss_total += naive_loss.item()

            # TODO:
            #  - negative sampling

            optimizer.zero_grad()
            naive_loss.backward()
            optimizer.step()

        print("~~~~~~~~~~~~~~~~~~")
        print(f"epoch: {epoch + 1} out of {EPOCHS}")
        print(f"time per epoch: {time.time() - tick}")
        print(f"mean loss: {loss_total / num_batches}")
        print(f"total loss: {loss_total}")
        print("~~~~~~~~~~~~~~~~~~")
