"""

### NOTICE ###
You DO NOT need to upload this file

"""
import argparse
from test import test
from environment import Environment


def parse():
    parser = argparse.ArgumentParser(description="MLDS&ADL HW3")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_pg', action='store_true', help='whether train policy gradient')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--train_ac', action='store_true', help='whether train actor-critic')
    parser.add_argument('--train_ddpg', action='store_true', help='whether train deep deterministic policy gradient')
    parser.add_argument('--train_a3c', action='store_true', help='whether train actor-critic')
    parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--test_ac', action='store_true', help='whether test actor-critic')
    parser.add_argument('--test_a3c', action='store_true', help='whether test actor-critic')
    parser.add_argument('--test_ddpg', action='store_true', help='whether test deep deterministic policy gradient')
    parser.add_argument('--video_dir', default=None, help='output video directory')
    parser.add_argument('--do_render', action='store_true', help='whether render environment')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        print('except')
        pass
    args = parser.parse_args()
    return args


def run(args):
    if args.train_pg:
        env_name = args.env_name or 'PongDeterministic-v4'
        env = Environment(env_name, args)
        from agent_dir.agent_pg import Agent_PG
        agent = Agent_PG(env, args)
        agent.train()
    if args.train_ac:
        env_name = args.env_name or 'PongDeterministic-v4'
        env = Environment(env_name, args)
        from agent_dir.agent_ac import Agent_AC
        agent = Agent_AC(env, args)
        agent.train()
    if args.train_a3c:
        env_name = args.env_name or 'PongDeterministic-v4'
        env = Environment(env_name, args)
        from agent_dir.agent_a3c import Agent_A3C
        agent = Agent_A3C(env, args)
        agent.train()

    if args.train_ddpg:
        env_name = args.env_name or 'Pong-v0'
        env = Environment(env_name, args)
        from agent_dir.agent_ddpg import Agent_DDPG
        agent = Agent_DDPG(env, args)
        agent.train()

    if args.train_dqn:
        env_name = args.env_name or 'BreakoutNoFrameskip-v4'
        env = Environment(env_name, args, atari_wrapper=True)
        
        from agent_dir.agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        agent.train()

    if args.test_pg:
        env = Environment('Pong-v0', args, test=True)
        from agent_dir.agent_pg import Agent_PG
        agent = Agent_PG(env, args)
        test(agent, env)
    if args.test_ac:
        env = Environment('Pong-v0', args, test=True)
        from agent_dir.agent_ac import Agent_AC
        agent = Agent_AC(env, args)
        test(agent, env)
    if args.test_a3c:
        env = Environment('PongDeterministic-v4', args, test=True)
        from agent_dir.agent_a3c import Agent_A3C
        agent = Agent_A3C(env, args)
        test(agent, env)
    if args.test_ddpg:
        env = Environment('Pong-v0', args, test=True)
        from agent_dir.agent_ddpg import Agent_DDPG
        agent = Agent_DDPG(env, args)
        test(agent, env)

    if args.test_dqn:
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
        from agent_dir.agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=100)

if __name__ == '__main__':
    args = parse()
    run(args)
