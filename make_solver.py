#import os
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from utils import read_json

def make_solver(criterion, opt, lr, w_d, net, train_loader, val_loader, epoc, sch):  # TODO: save path 만들기
    print("[Making solver...]")
    # Set criterion
    criterion = set_criterion(criterion)
    # Set optimizer
    optimizer, scheduler = set_optimizer(opt, lr, w_d, net, sch)

    # Set metrics
    # log_dict = set_metrics(args)

    # Set solver
    # 아래껀 고동희 방식, 각각 model 마다 solver 파일을 만들어서 실행하는 방식임
    # module = importlib.import_module(f"Solver.{net}_solver")
    # solver = module.Solver(net, train_loader, val_loader, criterion, optimizer) # 이후에 scheduler와 log_dict를 추가해야 함

    solver = Solver(net, train_loader, val_loader, criterion, optimizer, epoc, scheduler)

    return solver

class Solver:
    def __init__(self, net, train_loader, val_loader, criterion, optimizer, epoc, scheduler): #scheduler, log_dict를 추후에 추가해야 함
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoc = epoc
        self.scheduler = scheduler

    def train(self):

        self.net.train()
        for i, data in enumerate(self.train_loader):

            inputs, labels = data[0].cuda(), data[1].cuda()

            # Feed-forward
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)

            # Backward
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Print

        print("")

    def val(self):

        self.net.eval()
        loss = 0
        global correct
        correct = 0

        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                # Load batch data
                inputs, labels = data[0].cuda(), data[1].cuda()

                # Feed-forward
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)

                # 여기 아래부터는 내가 가진 쥬피터에 있는 코드 보고 따라하였음
                pred = outputs.argmax(dim=1)
                correct += pred.eq(labels.view_as(pred)).sum().item()

        print('\nTest set: Average Loss {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            loss, correct, len(self.val_loader.dataset), 100 * correct / len(self.val_loader.dataset)))

    def experiment(self):

        print("[Start experiment]")

        test_graph_acc = []
        graph_epo = []
        graph_epoch = 1

        for epoch in range(self.epoc):

            print(f"Epoch {epoch + 1}/{self.epoc}")
            # Train
            self.train()

            # Validation
            self.val()

            # 그래프를 표현하기 위해서 갖다 놓은것
            graph_epo.append(graph_epoch)
            graph_epoch += 1
            test_graph_acc.append(100 * correct / len(self.val_loader.dataset))


        # 최종 그래프 결과와 max값 확인
        plt.plot(graph_epo, test_graph_acc, 'b')
        plt.show()
        print("last_acc : ", test_graph_acc[-1])
        print("max_acc : ", max(test_graph_acc))

        return test_graph_acc



def set_criterion(criterion):
    if criterion == "MSE":
        criterion = nn.MSELoss()
    elif criterion == "CEE":
        criterion = nn.CrossEntropyLoss()
    elif criterion == 'NLL':
        criterion = nn.NLLLoss()
    return criterion


def set_optimizer(opt, lr, w_d, net, sch):
    if opt == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=w_d)
    elif opt == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=lr)

    if sch[0] == "exp":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=sch[1])

    # # Load optimizer  NOTE: 코드 맞는지 확인, scheduler 다르면 어떻게 되는지 확인 # 추가로 확인하기로 (스케줄러 확인하거나 불러와야 하는 상황에서)
    # if args.train_cont:
    #     print("[Loading optimizer...]")
    #     optimizer.load_state_dict(torch.load(args.train_cont)['optimizer_state_dict'])
    # # Set scheduler
    # if args.scheduler:
    #     if args.scheduler == 'exp':
    #         scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    #     elif args.scheduler == 'multi_step':
    #         scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.learning_gamma)
    #     elif args.scheduler == 'plateau':
    #         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20,
    #                                                          threshold=0.1, threshold_mode='abs', verbose=True)
    #     return optimizer, scheduler

    return optimizer, scheduler


# def set_metrics(args):
#     if args.train_cont:
#         print("[Loading metrics log...]")
#         log_dict = read_json(os.path.join(args.train_cont, "log_dict.json"))
#     else:
#         log_dict = {f"{phase}_{metric}": [] for phase in ["train", "val"] for metric in args.metrics}
#     return log_dict