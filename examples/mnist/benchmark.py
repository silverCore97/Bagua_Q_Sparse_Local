from __future__ import print_function
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import logging
import bagua.torch_api as bagua
import time
import sys

#Benchmark
#import timeit
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
#from torchvision import models


# Model for Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

#------Training Model
def train(args, model, train_loader, optimizer, epoch):
    #Sets model to training mode 
    model.train()
    # Benchmark
    logging.info("Running benchmark...")
    img_secs = []
    for batch_idx, (data, target) in enumerate(train_loader):
        s=time.time()
        #data: features , target: label
        data, target = data.cuda(), target.cuda()
        #Gradients are set back to zero here to avoid gradient accumulation
        optimizer.zero_grad()
        # Calculates predicted labels by using the model
        output = model(data)
        # Loss function using predicted labels and actual labels
        loss = F.nll_loss(output, target)
        # Backwards propagation    !!!calculates tensor loss gradient 
        loss.backward()
        # Optimizer step selection
        if args.fuse_optimizer:
            optimizer.fuse_step()
        else:
            optimizer.step()
        e = time.time()
        timer= e-s
        #time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
        #img_sec = args.batch_size * args.num_batches_per_iter / time
        img_sec = len(data)/timer
        logging.info(
           "Iter #%d: %.1f img/sec %s" % (batch_idx, img_sec * bagua.get_world_size(),
                                          "GPU")
        )
        img_secs.append(img_sec)
        if batch_idx % args.log_interval == 0:
            logging.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    return img_secs



def test(model, test_loader, train_loader):
    model.eval()
    test_loss = 0
    correct = 0


    train_loss = 0
    train_correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            train_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            train_correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)

    logging.info(
        "\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            train_loss,
            train_correct,
            len(train_loader.dataset),
            100.0 * train_correct / len(train_loader.dataset),
        )
    )

    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logging.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return test_loss, correct, train_loss, train_correct


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )

# set number of epochs here
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="qsparselocal",
        help="gradient_allreduce, bytegrad, decentralized, low_precision_decentralized, qadam, async",
        #Add new algorithm for testing------------------
    )

    parser.add_argument(
        "--async-sync-interval",
        default=500,
        type=int,
        help="Model synchronization interval(ms) for async algorithm",
    )
    parser.add_argument(
        "--set-deterministic",
        action="store_true",
        default=False,
        help="set deterministic or not",
    )
    parser.add_argument(
        "--fuse-optimizer",
        action="store_true",
        default=False,
        help="fuse optimizer or not",
    )
    parser.add_argument(
        "--gap",
        default=3,
        type=int,
        help="gap between synchronization rounds",
    )


    #args = parser.parse_args() 
    # New line below solves ipykernel_launcher.py: error: unrecognized arguments
    args, unknown = parser.parse_known_args()
    
    if args.set_deterministic:
        print("set_deterministic: True")
        np.random.seed(666)
        random.seed(666)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(666)
        torch.cuda.manual_seed_all(666 + int(bagua.get_rank()))
        torch.set_printoptions(precision=10)

    torch.cuda.set_device(bagua.get_local_rank())
    bagua.init_process_group()


    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)
    if bagua.get_rank() == 0:
        logging.getLogger().setLevel(logging.INFO)

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    if bagua.get_local_rank() == 0:
        dataset1 = datasets.MNIST(
            "../data", train=True, download=True, transform=transform
        )
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        dataset1 = datasets.MNIST(
            "../data", train=True, download=True, transform=transform
        )

    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset1, num_replicas=bagua.get_world_size(), rank=bagua.get_rank()
    )
    train_kwargs.update(
        {
            "sampler": train_sampler,
            "batch_size": args.batch_size // bagua.get_world_size(),
            "shuffle": False,
        }
    )
    # Train and Test dataset
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Throw the instantiation of the network onto the cuda dvice
    model = Net().cuda()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)


    if args.algorithm == "gradient_allreduce":
        from bagua.torch_api.algorithms import gradient_allreduce

        algorithm = gradient_allreduce.GradientAllReduceAlgorithm()
    elif args.algorithm == "decentralized":
        from bagua.torch_api.algorithms import decentralized

        algorithm = decentralized.DecentralizedAlgorithm()
    elif args.algorithm == "low_precision_decentralized":
        from bagua.torch_api.algorithms import decentralized

        algorithm = decentralized.LowPrecisionDecentralizedAlgorithm()
    elif args.algorithm == "bytegrad":
        from bagua.torch_api.algorithms import bytegrad

        algorithm = bytegrad.ByteGradAlgorithm()
    elif args.algorithm == "qadam":
        from bagua.torch_api.algorithms import q_adam

        optimizer = q_adam.QAdamOptimizer(
            model.parameters(), lr=args.lr, warmup_steps=100
        )
        algorithm = q_adam.QAdamAlgorithm(optimizer)
    #################################################################    
    elif args.algorithm == "qsparselocal":
        import qsparselocal
        # Set lower learning rate, no convergence for lr = 1
        optimizer = qsparselocal.QSparseLocalOptimizer(
            model.parameters(), lr=args.lr, schedule = args.gap+1
        )
        algorithm = qsparselocal.QSparseLocalAlgorithm(optimizer)
    elif args.algorithm == "async":
        from bagua.torch_api.algorithms import async_model_average

        algorithm = async_model_average.AsyncModelAverageAlgorithm(
            sync_interval_ms=args.async_sync_interval,
        )
    else:
        raise NotImplementedError

    #  Model von Bagua
    model = model.with_bagua(
        [optimizer],
        algorithm,
        do_flatten=not args.fuse_optimizer,
    )

    # Optimizer from Bagua if args.fuse_optimizer==True
    if args.fuse_optimizer:
        optimizer = bagua.contrib.fuse_optimizer(optimizer)

    #------------ Loss, accuracy
    loss_list =[]
    acc_list = []
    train_loss_list =[]
    train_acc_list = []
    
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    start = time.time()
    img_secs_total=[]
    for epoch in range(1, args.epochs + 1):
        if args.algorithm == "async":
            model.bagua_algorithm.resume(model)

        #Benchmark
        img_epoch = train(args, model, train_loader, optimizer, epoch)
        img_secs_total.append(img_epoch)
        if args.algorithm == "async":
            model.bagua_algorithm.abort(model)

        new_loss,new_acc, new_train_loss, new_train_acc =test(model, test_loader, train_loader)
        loss_list.append(new_loss)
        acc_list.append(new_acc*100/len(test_loader))
        train_loss_list.append(new_train_loss)
        train_acc_list.append(new_train_acc*100/len(train_loader))
        scheduler.step()
        ####
        torch.cuda.empty_cache()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    # Benchmark
    if args.algorithm == "async":
      model.bagua_algorithm.abort(model)

    # Used for measuring the time taken for the epochs themselves
    end = time.time()

    # Results Benchmark
    i=1
    for img_secs in img_secs_total:
        logging.info("Epoch %s" % (i))
        i+=1
        img_sec_mean = np.mean(img_secs)
        img_sec_conf = 1.96 * np.std(img_secs)
        logging.info("Img/sec per %s: %.1f +-%.1f" % ("GPU", img_sec_mean, img_sec_conf))
        logging.info(
            "Total img/sec on %d %s(s): %.1f +-%.1f"
            % (
                bagua.get_world_size(),
                "GPU",
                bagua.get_world_size() * img_sec_mean,
                bagua.get_world_size() * img_sec_conf,
            )
        )


    print("Elapsed time:",end-start)

    import matplotlib.pyplot as plt

    ep =[i for i in range(1, args.epochs + 1)]


    # Those three values only exist for 
    if args.algorithm == 'qsparselocal':
        print("Current quantization method:",qsparselocal.quantization_scheme)
        print("Gap:",gap)
    print("Learning rate:",args.lr)
    print("Train Loss:",train_loss_list)
    print("Train Accuracy:",train_acc_list)   
    print("Test Loss:",loss_list)
    print("Test Accuracy:",acc_list)
     
    """
    plt.figure(1)
    plt.subplot(211)
    plt.plot(ep,loss_list)
    plt.subplot(212)
    plt.plot(ep,acc_list)

    plt.show()  
    """
    


if __name__ == "__main__":
    main()
