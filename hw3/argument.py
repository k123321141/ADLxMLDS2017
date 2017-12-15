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


    parser.add_argument('--test', action='store_true', help='test dqn')
    parser.add_argument('--keep_train', action='store_true', default=True, help='load trained model')
    #
    #dqn setting 
    parser.add_argument('--dqn_model', default=join(home,'rl','breakout_dqn.h5'), help='path to save model for trainging')
    parser.add_argument('--dqn_duel_model', default=join(home,'rl','breakout_duel_dqn.h5'), 
            help='path to save duel network model for trainging')
    parser.add_argument('--dqn_summary', default=join(home,'rl','summary','breakout_dqn'), help='path to save summary for training')
    parser.add_argument('--dqn_epsilon', type=float, default=0.99, help='start epsilon')
    parser.add_argument('--dqn_epsilon_end', type=float, default=0.1, help='end epsilon')
    parser.add_argument('--dqn_exploration_steps', type=float, default=600000, help='how many step in env per epsilon decay')
    parser.add_argument('--dqn_batch', type=int, default=32, help='batch size')
    parser.add_argument('--dqn_train_start', type=int, default=3000, help='random action before start training')
    parser.add_argument('--dqn_update_target', type=int, default=10000, help='frequency of updating target network')
    parser.add_argument('--dqn_discount_factor', type=float, default=0.99, help='discount factor')
    parser.add_argument('--dqn_memory', type=int, default=40000, help='memory size for reply experience')
    parser.add_argument('--dqn_no_ops', type=int, default=10, help='do not action for init env')
    parser.add_argument('--dqn_save_interval', type=int, default=30, help='how many episodes per saving')
    parser.add_argument('--dqn_max_spisode', type=int, default=100000, help='maximum iteration')
    #bonus
    parser.add_argument('--dqn_dueling', action='store_true',default=False, help='dqn dueling network bonus')
    parser.add_argument('--dqn_double_dqn', action='store_true',default=False, help='double dqn bonus')
    #pg setting 
    parser.add_argument('--pg_model', default=join('rl','pong_pg.h5'), help='path to save model for trainging')
    parser.add_argument('--pg_summary', default=join('rl','summary','pong_pg'), help='path to save summary for training')
    parser.add_argument('--pg_batch', type=int, default=32, help='batch size')
    parser.add_argument('--pg_discount_factor', type=float, default=1., help='discount factor')
    parser.add_argument('--pg_baseline', type=int, default=0, help='baseline info')
    parser.add_argument('--pg_max_spisode', type=int, default=100000, help='maximum iteration')
    #bonus
    return parser
