import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.naive_nn import NaiveNN
from preprocessing.preprocess_data import CustomDataset


def learning():
    ## save data
    tick = time.time()
    DATASET = CustomDataset()
    print(time.time() - tick)
    # 284 s
    # exit(14)

    ## load data
    tick = time.time()
    DATASET = CustomDataset(load_x=True, load_y=True)
    print(time.time() - tick)
    # 0.003 s
    exit(12)

    # TODO:
    #  - implement arg parser and obtain all the consts from console

    DATA_SIZE, FEATURES_SIZE = DATASET.shape
    BATCH_SIZE = 500

    assert not DATA_SIZE % BATCH_SIZE, f"BATCH_SIZE must be a divisor of the DATASIZE, " \
            f"else the model will not have a proper layer size set. Given BATCH_SIZE: {BATCH_SIZE} " \
            f"and DATA_SIZE: {DATA_SIZE}. " \
            f"For Your reference, use one of these: " \
            f"{[x for x in range(DATA_SIZE) if 1 < x <= 1024 and not DATA_SIZE % x]}"

    H_DIMS = 200

    print(f"Amount of data read: {DATA_SIZE}")
    BATCHES = DataLoader(dataset=DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    print('Creating batches done')

    MODEL = NaiveNN(DATA_SIZE, BATCH_SIZE, FEATURES_SIZE, H_DIMS)
    EPOCHS = 100
    # ETA = 0.001
    ETA = 10
    # MOMENTUM = 0.9
    MOMENTUM = 0

    # CRITERION = nn.NLLLoss()
    # CRITERION = nn.L1Loss()
    CRITERION = nn.MSELoss()
    # CRITERION = nn.CrossEntropyLoss()
    OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=ETA, momentum=MOMENTUM)

    NUM_BATCHES = np.ceil(DATA_SIZE / BATCH_SIZE)
    print(f"Upcoming batches: {NUM_BATCHES}")

    
    # MODEL.train()
    for epoch in range(EPOCHS):
        correctly_classified, total_classified = 0.0, 0.0
        tick = time.time()
        loss_total = 0.0
        batch_count = 0

        for X, y in BATCHES:
            batch_count += 1
            outputs = MODEL(X)
            naive_loss = CRITERION(outputs, y)
            print(f"Batch {batch_count} loss: {naive_loss}, mean loss: {loss_total / batch_count}")
            loss_total += naive_loss.item()
            predicted = torch.argmax(outputs.reshape(len(outputs), -1).data, 1).float()
            total_classified += outputs.size(0)
            correctly_classified += (predicted == outputs).sum().item()
            OPTIMIZER.zero_grad()
            naive_loss.backward()
            OPTIMIZER.step()

            if not (batch_count) % 10 and batch_count > 0:
                print(f"Batch {batch_count} of epoch {epoch + 1}")
                print('mean loss {0:.4f}'.format(loss_total / batch_count))

        tock = time.time()
        print("~~~~~~~~~~~~~~~~~~")
        print(f"Epoch: {epoch + 1} out of {EPOCHS}")
        print(f"Time per epoch: {tock- tick}")
        print(f"Mean loss: {loss_total / NUM_BATCHES}")
        print(f"Total loss: {loss_total}")
        print("~~~~~~~~~~~~~~~~~~")
        print("@@@@@@@@@@@@@@@@@@")
        print(f"Accuracy of the model: {correctly_classified / total_classified * 100.0}%")
        print("@@@@@@@@@@@@@@@@@@")


        #  zapisywanie wag do pliku


if __name__ == "__main__":
    learning()