import argparse




if __name__ == '__main__':
    print()

    parser = argparse.ArgumentParser(description='select args')
    parser.add_argument('--task', type=int, default=2, help='choose task2 or task3')
    parser.add_argument('--tVec_dim', '-t', type=int, default=16)
    parser.add_argument('--hid_siz', '-hid', type=int, default=128)

    args = parser.parse_args()

    if args.task == 2:
        from task2_utils.train_time2vec2 import main
        main(tVec_dim=args.tVec_dim, hid_siz=args.hid_siz)
        # rmse:97.8813598195819,smape:50.781584439212324
    if args.task == 3:
        from task3_utils.train_time2vec3 import main
        main(tVec_dim=args.tVec_dim, hid_siz=args.hid_siz)
        # test_acc:0.7048


