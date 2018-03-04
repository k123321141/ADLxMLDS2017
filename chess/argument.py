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
    #parser.add_argument('--dqn_model', default=join(home,'rl','breakout_dqn.h5'), help='path to save model for trainging')
    parser.add_argument('--dqn_model', default=join('.','saved_breakout_dqn.h5'), help='path to save model for trainging')
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
    #parser.add_argument('--pg_model', default=join('rl','pong_pg.h5'), help='path to save model for trainging')
    parser.add_argument('--pg_model', default=join('.','saved_pong_pg.h5'), help='path to save model for trainging')
    parser.add_argument('--pg_summary', default=join('rl','summary','pong_pg'), help='path to save summary for training')
    parser.add_argument('--pg_batch', type=int, default=32, help='batch size')
    parser.add_argument('--pg_discount_factor', type=float, default=1., help='discount factor')
    parser.add_argument('--pg_baseline', type=int, default=0, help='baseline info')
    parser.add_argument('--pg_max_spisode', type=int, default=100000, help='maximum iteration')
    parser.add_argument('--pg_save_interval', type=int, default=10, help='how many episodes per saving')
    #parser.add_argument('--pg_old_model', action='store_true', default=False, help='load old model with 6 action')
    parser.add_argument('--pg_old_model', action='store_true', default=True, help='load old model with 6 action')
    
    #ac setting 
    parser.add_argument('--ac_model', default=join('.','models','pong_ac.h5'), help='path to save model for trainging')
    parser.add_argument('--ac_summary', default=join('.','summary','pong_ac'), help='path to save summary for training')
    parser.add_argument('--ac_discount_factor', type=float, default=0.9, help='discount factor')
    parser.add_argument('--ac_baseline', type=int, default=0, help='baseline info')
    parser.add_argument('--ac_max_spisode', type=int, default=100000, help='maximum iteration')
    parser.add_argument('--ac_save_interval', type=int, default=1, help='how many episodes per saving')
    parser.add_argument('--ac_update_target_frequency', type=int, default=1, help='frequency of updating target network per episode')
    
    #a3c setting 
    parser.add_argument('--a3c_model', default=join('.','models','pong_a3c.h5'), help='path to save model for trainging')
    parser.add_argument('--a3c_worker_count', type=int, default=16, help='due to cpu count')
    parser.add_argument('--a3c_summary', default=join('.','summary','pong_a3c'), help='path to save summary for training')
    parser.add_argument('--a3c_discount_factor', type=float, default=0.99, help='discount factor')
    parser.add_argument('--a3c_worker_num', type=int, default=100000, help='maximum iteration')
    parser.add_argument('--a3c_max_spisode', type=int, default=100000, help='maximum iteration')
    parser.add_argument('--a3c_save_interval', type=int, default=3, help='how many episodes per saving')
    parser.add_argument('--a3c_update_target_frequency', type=int, default=1000, help='frequency of updating target network per episode')
    parser.add_argument('--a3c_train_frequency', type=int, default=10, help='how many steps per update')
    
    #ddpg setting 
    parser.add_argument('--ddpg_model', default=join('.','models','pong_ddpg.h5'), help='path to save model for trainging')
    parser.add_argument('--ddpg_summary', default=join('.','summary','pong_ddpg'), help='path to save summary for training')
    parser.add_argument('--ddpg_discount_factor', type=float, default=0.9, help='discount factor')
    parser.add_argument('--ddpg_baseline', type=int, default=0, help='baseline info')
    parser.add_argument('--ddpg_max_spisode', type=int, default=100000, help='maximum iteration')
    parser.add_argument('--ddpg_save_interval', type=int, default=4, help='how many episodes per saving')
    parser.add_argument('--ddpg_train_start', type=int, default=10000, help='random action before start training')
    parser.add_argument('--ddpg_epsilon', type=float, default=0.9, help='start epsilon')
    parser.add_argument('--ddpg_epsilon_end', type=float, default=0.1, help='end epsilon')
    parser.add_argument('--ddpg_exploration_steps', type=float, default=500000, help='how many step in env per epsilon decay')
    parser.add_argument('--ddpg_update_target_frequency', type=int, default=1, help='frequency of updating target network per episode')
    parser.add_argument('--ddpg_train_frequency', type=int, default=1, help='how many steps per update')
    parser.add_argument('--ddpg_batch_size', type=int, default=32, help='batch size per update')

    parser.add_argument('--TAU', type=int, default=0.1, help='the rate of target networks updating')
    parser.add_argument('--reply_buffer', type=int, default=100000, help='memory size for reply experience')
    #bonus
    return parser
