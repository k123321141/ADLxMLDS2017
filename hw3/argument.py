def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    from os.path import expanduser,join
    home = expanduser("~")

    MODEL_PATH      =   join(home,'rl','breakout_dqn.h5')
    SUMMARY_PATH    =   join(home,'rl','summary','breakout_dqn_')

    parser.add_argument('--test', action='store_true', help='test dqn')
    parser.add_argument('--keep_train', action='store_true', default=True, help='load trained model')
    #
    #dqn setting 
    parser.add_argument('--dqn_model', default=MODEL_PATH, help='path to save model for trainging')
    parser.add_argument('--dqn_summary', default=SUMMARY_PATH, help='path to save summary for training')
    parser.add_argument('--dqn_summary_name', default='default', help='summary version')
    parser.add_argument('--dqn_epsilon', type=float, default=0.2, help='start epsilon')
    parser.add_argument('--dqn_epsilon_end', type=float, default=0.001, help='end epsilon')
    parser.add_argument('--dqn_exploration_steps', type=float, default=1000000, help='how many step in env per epsilon decay')
    parser.add_argument('--dqn_batch', type=int, default=32, help='batch size')
    parser.add_argument('--dqn_train_start', type=int, default=3000, help='random action before start training')
    parser.add_argument('--dqn_update_target', type=int, default=10000, help='frequency of updating target network')
    parser.add_argument('--dqn_discount_factor', type=float, default=0.99, help='discount factor')
    parser.add_argument('--dqn_memory', type=int, default=100000, help='memory size for reply experience')
    parser.add_argument('--dqn_no_ops', type=int, default=10, help='do not action for init env')
    parser.add_argument('--dqn_save_interval', type=int, default=1000, help='do not action for init env')
    #bonus
    parser.add_argument('--dqn_dueling', action='store_true', help='dqn dueling network bonus')
    parser.add_argument('--dqn_double_dqn', action='store_true', help='double dqn bonus')
    return parser
