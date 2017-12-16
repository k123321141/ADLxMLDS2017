import numpy as np
import json
from collections import deque
import matplotlib.pyplot as plt

def plt_file(path,color):
    with open(path, 'r') as f:
        data = f.read()
    j_data = json.loads(data)
    q = deque()
    scores = []
    episodes = []
    for buf in j_data:
        _,e,r = buf
        q.append(r)
        if len(q) > 30:
            q.popleft()
        scores.append(np.mean(q))
        episodes.append(e)
    plt.plot(episodes,scores,color)
    plt.ylabel('episodes')
    plt.ylabel('scores')
plt_file('/Users/payo_mac/Downloads/run_DQN_duel_summary-tag-Total_Reward_Episode.json','r-')
plt_file('/Users/payo_mac/Downloads/run_DQN_summary-tag-Total_Reward_Episode.json','b-')
plt_file('/Users/payo_mac/Downloads/run_double_DQN_summary-tag-Total_Reward_Episode.json','g-')

plt.show()
