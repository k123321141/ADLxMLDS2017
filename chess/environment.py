"""

### NOTICE ###
You DO NOT need to upload this file

"""
import gym
import numpy as np
from atari_wrapper import make_wrap_atari
from gym.spaces.box import Box
from cv2 import resize

class Environment(object):
    def __init__(self, env_name, args, atari_wrapper=False, test=False):
        if atari_wrapper:
            clip_rewards = not test
            self.env = make_wrap_atari(env_name, clip_rewards)
        else:
            assert 'Pong' in env_name
            self.env = getPongEnv(env_name)

        
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.do_render = args.do_render

        if args.video_dir:
            self.env = gym.wrappers.Monitor(self.env, args.video_dir, force=True)

    def seed(self, seed):
        '''
        Control the randomness of the environment
        '''
        self.env.seed(seed)

    def reset(self):
        '''
        When running dqn:
            observation: np.array
                stack 4 last frames, shape: (84, 84, 4)

        When running pg:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)
        '''
        observation = self.env.reset()

        return np.array(observation)


    def step(self,action):
        '''
        When running dqn:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
            reward: int
                wrapper clips the reward to {-1, 0, 1} by its sign
                we don't clip the reward when testing
            done: bool
                whether reach the end of the episode?

        When running pg:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)
            reward: int
                if opponent wins, reward = +1 else -1
            done: bool
                whether reach the end of the episode?
        '''
        if not self.env.action_space.contains(action):
            raise ValueError('Ivalid action!!')

        if self.do_render:
            self.env.render()

        observation, reward, done, info = self.env.step(action)

        return np.array(observation), reward, done, info


    def get_action_space(self):
        return self.action_space


    def get_observation_space(self):
        return self.observation_space


    def get_random_action(self):
        return self.action_space.sample()
def getPongEnv(env_id):

    if 'ram' in env_id:
        env = gym.make(env_id)
        env = RamNormalizedEnv(env)
        return env

    else:

        env = gym.make(env_id)
        if 'NoFrameskip' in env_id:
            assert 'NoFrameskip' in env.spec.id
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=args.skip_rate)
        env = EpisodicLifeEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        if 'Pong' in env_id:
            env = PongActionResetEnv(env)
        env = AtariRescale(env)
        env = NormalizedEnv(env)
        return env

def process_frame(frame):
    #frame = frame[conf["crop1"]:conf["crop2"] + 160, :160]
    frame = frame[34:34 + 160, :160]
    #frame = frame.mean(2)
    frame = frame[:,:,0]
    frame = frame.astype(np.float32)
    #frame *= (1.0 / 255.0)
    frame = resize(frame, (80, 80))
    frame = np.reshape(frame, [1, 80, 80])
    return frame


class AtariRescale(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(0.0, 1.0, [1, 80, 80], dtype='uint8')

    def observation(self, observation):
        return process_frame(observation)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        '''
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0
        '''
    def observation(self, observation):
        '''
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)
        '''
        I = observation.astype(np.uint8) 
        #print I
        I[I == 144] = 0
        I[I == 109] = 0
        I[I != 0] = 1
        return I.astype(np.float)

class RamNormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0
    def observation(self, observation):
        '''
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)
        '''
        #return (observation-127)/255.
        return observation 
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        noops = random.randrange(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

    def step(self, ac):
        return self.env.step(ac)

class PongActionResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        self.env.action_space.n = 3
    def step(self, ac):
        if ac == 0:
            ac = 0
        elif ac == 1:
            ac = 2
        elif ac == 2:
            ac = 3
        else:
            print('error')
            import sys
            sys.exit()
        return self.env.step(ac)

    def reset(self):
        return self.env.reset()


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = True
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = info['ale.lives']
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
            self.was_real_done = False
        self.lives = lives
        return obs, reward, done, info

    def reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
            self.lives = 0
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, info = self.env.step(0)
            self.lives = info['ale.lives']
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs
