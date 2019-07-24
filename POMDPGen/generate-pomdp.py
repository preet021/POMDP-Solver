import random
import sys

states, actions, observations = map(int, input().strip().split())

discount = random.randint(75, 95) / 100

f = open("../examples/" + sys.argv[1], "w+")

f.write("discount: %.2f\n" % (discount))

f.write("values: reward\n")

s = "states: "
for i in range(states):
	s = s + "s{} ".format(i)
s = s + "\n"
f.write(s)

s = "actions: "
for i in range(actions):
	s = s + "a{} ".format(i)
s = s + "\n"
f.write(s)

s = "observations: "
for i in range(observations):
	s = s + "p{} ".format(i)
s = s + "\n"
f.write(s)

s = "start: "
for i in range(states):
	s = s + str(1/states) + " "
s = s + "\n"
f.write(s)
f.write("\n")

# writing the reward function
for i in range(states):
	for j in range(actions):
		s = "R: a{} : s{} : * : * {}\n".format(j, i, random.randint(0,10))
		f.write(s)
f.write("\n")

# writing the observation function
for i in range(actions):
	for j in range(states):
		obs = random.randint(0, observations - 1)
		for k in range(observations):
			if k == obs:
				s = "O : a{} : s{} : p{} 1.0\n".format(i, j, k)
			else:
				s = "O : a{} : s{} : p{} 0.0\n".format(i, j, k)
			f.write(s)
f.write("\n")

# writing the transition function
for i in range(states):
	for j in range(actions):
		t = [0 for s in range(states)]
		l = 0 if ((i - 5) < 0) else (i - 5)
		r = (states - 1) if ((i + 5) >= states) else (i + 5)
		pr_sum = 0
		for s in range(l, r):
			p = random.randint(0,9)/10
			while (pr_sum + p > 1):
				p = random.randint(0,9)/10
			pr_sum += p
			t[s] = p
		t[r] = 1 - pr_sum
		pr_sum = 0
		for s in range(l, r + 1):
			pr_sum += t[s]
		assert pr_sum == 1
		for k in range(states):
			s = "T : a{} : s{} : s{} {}\n".format(j, i, k, round(t[k], 1))
			f.write(s)
f.write("\n")

f.close()
