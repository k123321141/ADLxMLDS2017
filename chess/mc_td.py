import random


d = 0.9
class state(object):
    def __init__(self, v):
        self.v = v
        self.n = 1
def r(a, b):
    if a == 0:
        return 0
    elif a == 1:
        return -1
    elif a == 2:
        return 1 if b == 5 else -1
def show(states, i):
    assert len(states) == 7
    s0,s1,s2,s3,s4,s5,s6 = states
    print '%s%3d%s' % ('-'*27, i, '-'*30) 
    print '%s%2.2f%s' % (' '*20, s0.v ,' '*15)
    print '%s%2.2f%s%2.2f%s' % (' '*10, s1.v,' '*17, s2.v,' '*5)
    print '%2.2f%s%2.2f%s%2.2f%s%2.2f' % (s3.v,' '*14, s4.v,' '*2,s5.v, ' '*12, s6.v)
    print '-'*60
def mc_demo():

    s = []
    for i in range(7):
        s.append(state(0.))
    idx = 1
    show(s,idx)
    p1 = [0,1,3]
    p2 = [0,2,5]
    p3 = [0,1,4]
    p4 = [0,2,6]
    ps = [p1,p2,p3,p4]
    idx += 1
    for i in range(100000):
        p = random.choice(ps)
        a,b,c = p
        dr = discounted_rewards = [ d*r(b,c), r(b,c) ]
        s[b].v = (s[b].v * s[b].n + dr[1])/ (s[b].n+1)
        s[a].v = (s[a].v * s[a].n + dr[0])/ (s[a].n+1)
        s[a].n +=1
        s[b].n +=1

        show(s, idx)
        idx += 1



mc_demo()



