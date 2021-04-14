#from config import arg
from data_loader import data_loader
from build_net import build_net
from make_solver import make_solver
from utils import control_random, timeit
import numpy as np
import time






#@timeit
def main():

    first_sub = 1
    last_sub = 9

    for i in range(first_sub, last_sub+1):
    # # Parsing
    # args = arg()
    #
    # # Control randomness
    # if args.seed:
    #     control_random(args)

        # Setting parameters
        mode = 'train'
        sub_ind = i
        batch_size = 72
        seed = 42
        gpu = 'multi'
        net = 'EEGNet2D' # 'CNN', 'EEGNet2D', 'Seo_CNN'
        load_path = None
        criterion = 'NLL'
        opt = 'Adam'
        lr = 0.001
        w_d = 2e-3
        epoc = 5000
        cfg = [6, 8] # filter_n, filter_size in WKN
        CSP = None # True or else
        sch = ['exp', 0.999] # 0 : 스케줄러 종류('exp') , 1 : gamma

        print(f'''
        sub_ind : {sub_ind}
        batch_size : {batch_size}
        seed : {seed}
        gpu : {gpu}
        net : {net} 
        criterion : {criterion}
        opt : {opt}
        lr : {lr}
        w_d : {2e-3}
        epoc : {epoc}
        cfg : {cfg}
        CSP : {CSP}
        sch : {sch}
        ----Args are completed!!!----
        '''
        )
        print('')

        # Control randomness
        if seed:
            control_random(seed, gpu)
        print('----Randomness is completed!!!----')
        print('')

        # Load data
        train_loader, val_loader = data_loader(sub_ind, batch_size, CSP) # NOTE: data shpae = [trials, segment, band, scale, chans, times]
        print('----Data loading is completed!!!----')
        print('')

        # Build net
        net = build_net(net, cfg, train_loader.dataset.X.shape, load_path, gpu)
        print('----Building net is completed!!!----')
        print('')

        # Make solver
        solver = make_solver(criterion, opt, lr, w_d, net, train_loader, val_loader, epoc, sch)
        print('----Making solver is completed!!!----')
        print('')

        #
        # Run
        if mode != 'test':
            test_acc = solver.experiment()
            print("sub_ind : ", sub_ind)
            dir = 'C:/Users/MAI Lab/PycharmProjects/BCI_seo/sub' + f'{i}' + '_acc.npy'
            np.save(dir, test_acc)
        else:  # NOTE: test 만들어야 함
            pass

if __name__ == '__main__':
    start = time.time()
    main()
    print("time :", time.time() - start)