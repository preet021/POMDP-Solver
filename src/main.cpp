#include <stdio.h>
#include <ctype.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <sys/types.h>
#include <assert.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <queue>
#include <map>
#include <set>

#define sz(a) (int)(a).size()

using namespace std;
#include "witness.h"

const double INF = 1e18;
const int TIME_HORIZON = 3;
bool has_discount = false, has_states = false, has_actions = false, has_observations = false, has_start = false;
double **R, ***T, ***O;
double discount, almost_zero = 1e-5;
int num_of_states = 0, num_of_actions = 0, num_of_observations = 0, time_horizon = 0;
map <string, int> state_map, action_map, obs_map;
map <int, string> inv_state_map, inv_action_map, inv_obs_map;
vector <ptree> V[TIME_HORIZON+1]; // Value Function
queue <re> Q_R;
queue <te> Q_T;
queue <oe> Q_O;
oe obs;
te trns;
re rew;
bstate cur_b, next_b;

// Function declarations
void store_reward_func();
void store_transition_func();
void store_obs_func();

double difference(vector <ptree>& a, vector <ptree>& b);
double weakbound(vector <ptree>& X, vector <ptree>& Y);

void solvePOMDP();
void witness(int t, int a);
void prune(int t, vector<ptree>& X);
bool findb(int a, int t, vector<ptree>& Q, bstate& b);
double check_pnew(vector<ptree>& Q, ptree& pnew, bstate& b);

ptree besttree(bstate& b, int a, vector<ptree>& X);
void back(vector <double>& alpha, int a, int o, vector<double>& _alpha);

void print_tree(ptree& p);

int main(int argc, char* argv[]) {

	// argv[1] should be the path to input file

	// Checking whether input file is provided
	if (argc != 2) {
		fprintf(stderr, "Usage:\n./binary <path_to_input_file>\n");
		exit(EXIT_FAILURE);
	}

	// Checking whether path to input file is a valid path
	struct stat st;
	stat(argv[1], &st);
	bool is_reg_file = S_ISREG(st.st_mode);
	if (!is_reg_file) {
		fprintf(stderr, "ERROR: file %s is not a regular file\n", argv[1]);
		exit(EXIT_FAILURE);
	}

	// Opening file
	char *line = NULL, *trimmed;
	size_t len = 0;
	ssize_t read;
	FILE *fp = fopen(argv[1], "r");
	if (!fp) {
		fprintf(stderr, "ERROR: unable to open file %s\n", argv[1]);
		exit(EXIT_FAILURE);
	}

	// Reading file line by line and storing the POMDP variables
	char *s, *t;
	int it;
	double sm;
	string str;
	while ((read = getline(&line, &len, fp)) != -1) {
		line = trim(line, (int)read);
		if (!strncmp("discount", line, 8)) {
			has_discount = true;
			for (int i=0; line[i]; ++i)
				if (line[i] == ':') {
					sscanf(line+i+1, "%lf", &discount);
					break;
				}
		}
		else if (!strncmp("states", line, 6)) {
			has_states = true;
			for (int i=0; line[i]; ++i)
				if (line[i] == ':') {
					s = line + i + 1;
					break;
				}
			t = strtok(s, " \n\t\v\f\r");
			while (t != NULL) {
				str = "";
				for (int i=0; t[i]; ++i)
					str += t[i];
				state_map[str] = num_of_states;
				++num_of_states;
				t = strtok(NULL, " \n\t\v\f\r");
			}
		}
		else if (!strncmp("actions", line, 7)) {
			has_actions = true;
			for (int i=0; line[i]; ++i)
				if (line[i] == ':') {
					s = line + i + 1;
					break;
				}
			t = strtok(s, " \n\t\v\f\r");
			while (t != NULL) {
				str = "";
				for (int i=0; t[i]; ++i)
					str += t[i];
				action_map[str] = num_of_actions;
				++num_of_actions;
				t = strtok(NULL, " \n\t\v\f\r");
			}
		}
		else if (!strncmp("observations", line, 12)) {
			has_observations = true;
			for (int i=0; line[i]; ++i)
				if (line[i] == ':') {
					s = line + i + 1;
					break;
				}
			t = strtok(s, " \n\t\v\f\r");
			while (t != NULL) {
				str = "";
				for (int i=0; t[i]; ++i)
					str += t[i];
				obs_map[str] = num_of_observations;
				++num_of_observations;
				t = strtok(NULL, " \n\t\v\f\r");
			}
		}
		else if (!strncmp("start", line, 5)) {
			has_start = true;
			cur_b.b.assign(num_of_states, 0);
			t = strtok(line+5, ": \n\t\v\r\f");
			it = 0;
			while (t != NULL) {
				sscanf(t, "%lf", &cur_b.b[it]);
				++it;
				t = strtok(NULL, ": \n\t\v\r\f");
			}
			sm = 1;
			for (int i=0; i<num_of_states; ++i)
				sm -= cur_b.b[i];
			// Checking if sum of probabilities is 1
			if (sm > almost_zero || sm < -almost_zero) {
				fprintf(stderr, "ERROR: start belief state probabilities inconsistent\n");
    			exit(EXIT_FAILURE);
			}
		}
		else if (!strncmp("R", line, 1)) {
			t = strtok(line+1, ": \n\t\v\r\f");
			it = 0;
			while (t != NULL) {
				str = "";
				for (int i=0; t[i]; ++i)
					str += t[i];
				if (it == 0) {
					if (str == "*")
						rew.action = -1;
					else
						rew.action = action_map[str];
				}
				else if (it == 1) {
					if (str == "*")
						rew.state = -1;
					else
						rew.state = state_map[str];
				}
				else if (it == 4) {
					sscanf(t, "%lf", &rew.value);
				}
				it++;
				t = strtok(NULL, ": \n\t\v\f\r");
			}
			Q_R.push(rew);
		}
		else if (!strncmp("T", line, 1)) {
			t = strtok(line+1, ": \n\t\v\r\f");
			int it = 0;
			while (t != NULL) {
				str = "";
				for (int i=0; t[i]; ++i)
					str += t[i];
				if (it == 0) {
					if (str == "*")
						trns.action = -1;
					else
						trns.action = action_map[str];
				}
				else if (it == 1) {
					if (str == "*")
						trns.start_state = -1;
					else
						trns.start_state = state_map[str];
				}
				else if (it == 2) {
					if (str == "*")
						trns.end_state = -1;
					else
						trns.end_state = state_map[str];
				}
				else if (it == 3) {
					sscanf(t, "%lf", &trns.value);
				}
				it++;
				t = strtok(NULL, ": \n\t\v\f\r");
			}
			Q_T.push(trns);
		}
		else if (!strncmp("O", line, 1)) {
			t = strtok(line+1, ": \n\t\v\r\f");
			int it = 0;
			while (t != NULL) {
				str = "";
				for (int i=0; t[i]; ++i)
					str += t[i];
				if (it == 0) {
					if (str == "*")
						obs.action = -1;
					else
						obs.action = action_map[str];
				}
				else if (it == 2) {
					if (str == "*")
						obs.obs = -1;
					else
						obs.obs = obs_map[str];
				}
				else if (it == 1) {
					if (str == "*")
						obs.end_state = -1;
					else
						obs.end_state = state_map[str];
				}
				else if (it == 3) {
					sscanf(t, "%lf", &obs.value);
				}
				it++;
				t = strtok(NULL, ": \n\t\v\f\r");
			}
			Q_O.push(obs);
		}
	}
    if (!has_discount) {
    	fprintf(stderr, "ERROR: discount factor missing in input file %s\n", argv[1]);
    	exit(EXIT_FAILURE);
    }
    else if (!has_states) {
    	fprintf(stderr, "ERROR: set of stated missing in input file %s\n", argv[1]);
    	exit(EXIT_FAILURE);
    }
    else if (!has_start) {
    	// agent can be in any state with equal probability
    	cur_b.b.assign(num_of_states, (double)1 / num_of_states);
    }
    else if (!has_actions) {
    	fprintf(stderr, "ERROR: set of actions missing in input file %s\n", argv[1]);
    	exit(EXIT_FAILURE);
    }
    else if (!has_observations) {
    	fprintf(stderr, "ERROR: set of observations missing in input file %s\n", argv[1]);
    	exit(EXIT_FAILURE);
    }

    // Creating the inverese state, action and obs map
    for (auto it: state_map) {
    	inv_state_map[it.second] = it.first; 
    }
    for (auto it: action_map) {
    	inv_action_map[it.second] = it.first;
    }
    for (auto it: obs_map) {
    	inv_obs_map[it.second] = it.first;
    }
    
    // Storing the Reward function   R(s, a)
    store_reward_func();

    // Storing the Transition function   T(s, a, s')
    store_transition_func();

    // Storing the Observation function   O(a, s', o)
    store_obs_func();

    // Printing the input POMDP
	cout << "Number of States: " << num_of_states << '\n';
	cout << "Number of Actions: " << num_of_actions << '\n';
	cout << "Number of Observations: " << num_of_observations << '\n';
	cout << "Discount Factor: " << discount << '\n';
	cout << "Initial Belief State: ";
	for (int i=0; i<num_of_states; ++i)
		cout << cur_b.b[i] << " ";
	cout << "\n\n";

	// Calling the solver
	solvePOMDP();

	// Removing the intermediate auxilary files
	system("rm model.lp out.lp");

	// Printing the best action after taking the observation as input
	cout << "\n----------------------------------------------------\nPOMDP Solved" << endl;
	// for (int i=0; i<sz(V[TIME_HORIZON]); ++i)
		// print_tree(V[TIME_HORIZON][i]);
	while (1) {

		// Finding best action for current belief state
		int best_action = -1;
		double bestval = -INF;
		for (int i=0; i<sz(V[time_horizon]); ++i) {
			if (dot_product(cur_b.b, V[time_horizon][i].value) > bestval) {
				bestval = dot_product(cur_b.b, V[time_horizon][i].value);
				best_action = V[time_horizon][i].action;
			}
		}
		cout << "\ncurrent belief state:\n";
		for (int s=0; s<num_of_states; ++s)
			cout << cur_b.b[s] << " ";
		cout << "\nbest_action: " << inv_action_map[best_action] << endl;

		// Getting the evidence
		cout << "Observation: ";
		string obs;
		cin >> obs;
		int o = obs_map[obs];
		assert(o >= 0 && o < num_of_observations);

		// Computing the next belief state
		next_b.b.assign(num_of_states, 0);
		bestval = 0;
		for (int i=0; i<num_of_states; ++i) {
			if (O[best_action][i][o]) {
				for (int s=0; s<num_of_states; ++s)
					if (T[s][best_action][i])
						next_b.b[i] += T[s][best_action][i] * cur_b.b[s];
				next_b.b[i] *= O[best_action][i][o];
			}
			bestval += next_b.b[i];
		}
		for (int i=0; i<num_of_states; ++i)
			cur_b.b[i] = next_b.b[i] / bestval;
		
		// time_horizon--;
	}
	exit(EXIT_SUCCESS);
}

void store_reward_func() {
	R = new double*[num_of_states];
    re tp, rew;
    while (!Q_R.empty()) {
    	tp = Q_R.front();
    	Q_R.pop();
    	if (tp.action == -1) {
    		for (int i=0; i<num_of_actions; ++i) {
    			rew = tp;
    			rew.action = i;
    			Q_R.push(rew);
    		}
    	}
    	else if (tp.state == -1) {
    		for (int i=0; i<num_of_states; ++i) {
    			rew = tp;
    			rew.state = i;
    			Q_R.push(rew);
    		}
    	}
    	else {
    		if (R[tp.state] == NULL) {
    			R[tp.state] = new double[num_of_actions];
    			memset(R[tp.state], 0, sizeof(R[tp.state]));
    		}
    		R[tp.state][tp.action] += tp.value;
    	}
    }
}

void store_transition_func() {
	T = new double**[num_of_states];
    for (int i=0; i<num_of_states; ++i)
    	T[i] = new double*[num_of_actions];
    te tp, trns;
    while (!Q_T.empty()) {
    	tp = Q_T.front();
    	Q_T.pop();
    	if (tp.action == -1) {
    		for (int i=0; i<num_of_actions; ++i) {
    			trns = tp;
    			trns.action = i;
    			Q_T.push(trns);
    		}
    	}
    	else if (tp.start_state == -1) {
    		for (int i=0; i<num_of_states; ++i) {
    			trns = tp;
    			trns.start_state = i;
    			Q_T.push(trns);
    		}
    	}
    	else if (tp.end_state == -1) {
    		for (int i=0; i<num_of_states; ++i) {
    			trns = tp;
    			trns.end_state = i;
    			Q_T.push(trns);
    		}
    	}
    	else {
    		if (T[tp.start_state][tp.action] == NULL) {
    			T[tp.start_state][tp.action] = new double[num_of_states];
    			memset(T[tp.start_state][tp.action], 0, sizeof(T[tp.start_state][tp.action]));
    		}
    		T[tp.start_state][tp.action][tp.end_state] += tp.value;
    	}
    }
}

void store_obs_func() {
	O = new double**[num_of_actions];
    for (int i=0; i<num_of_actions; ++i)
    	O[i] = new double*[num_of_states];
    oe tp, obs;
    while (!Q_O.empty()) {
    	tp = Q_O.front();
    	Q_O.pop();
    	if (tp.action == -1) {
    		for (int i=0; i<num_of_actions; ++i) {
    			obs = tp;
    			obs.action = i;
    			Q_O.push(obs);
    		}
    	}
    	else if (tp.obs == -1) {
    		for (int i=0; i<num_of_observations; ++i) {
    			obs = tp;
    			obs.obs = i;
    			Q_O.push(obs);
    		}
    	}
    	else if (tp.end_state == -1) {
    		for (int i=0; i<num_of_states; ++i) {
    			obs = tp;
    			obs.end_state = i;
    			Q_O.push(obs);
    		}
    	}
    	else {
    		if (O[tp.action][tp.end_state] == NULL) {
    			O[tp.action][tp.end_state] = new double[num_of_observations];
    			memset(O[tp.action][tp.end_state], 0, sizeof(O[tp.action][tp.end_state]));
    		}
    		O[tp.action][tp.end_state][tp.obs] += tp.value;
		}
	}
}

void back(vector <double>& alpha, int a, int o, vector<double>& _alpha) {
	/*
		back[alpha,a,o](s) is the expected reward received by the agent
		that takes action a starting in s, observer o and then follows the
		policy tree corresponding to alpha. It does not include the reward
		of taking action a in s (i.e. R(s, a)).
	*/
	_alpha.clear();
	_alpha.assign(num_of_states, 0);
	for (int s=0; s<num_of_states; ++s)
		for (int _s=0; _s<num_of_states; _s++)
			if (O[a][_s][o] && T[s][a][_s]) {
				_alpha[s] += alpha[_s] * T[s][a][_s] * O[a][_s][o];
			}
}

ptree besttree(bstate& b, int a, vector<ptree>& X) {
	ptree p;
	p.action = a;
	// cout << "\nbesttree" << endl;
	// for (int i=0; i<num_of_states; ++i)
	// 	cout << b.b[i] << " ";
	// cout << endl;
	p.value.assign(num_of_states, 0);
	for (int s=0; s<num_of_states; ++s)
		if (R[s][a]) p.value[s] += R[s][a];

	if (time_horizon > 1) {
		// Finding the best t-1 step policy tree for each observation
		p.choice.resize(num_of_observations);
		vector <double> _vec;
		for (int o=0; o<num_of_observations; ++o) {
			int bestpol = -1;
			double bestval = -INF, val;
			for (int i=0; i<sz(X); ++i) {
				back(X[i].value, a, o, _vec);
				val = dot_product(b.b, _vec);
				if (val > bestval) {
					bestval = val;
					bestpol = i;
				}
			}
			assert(bestpol != -1);
			p.choice[o] = bestpol;
		}
		// Calculating the value vector
		vector <double> _alpha;
		for (int s=0; s<num_of_states; ++s) {
			for (int o=0; o<num_of_observations; ++o) {
				back(X[p.choice[o]].value, a, o, _alpha);
				p.value[s] += discount * _alpha[s];
			}
		}
	}
	// print_tree(p);
	// cout << "besttree returning" << endl;
	return p;
}

void print_tree(ptree& p) {
	cout << "\nAction: " << inv_action_map[p.action] << "\nValue: ";
	for (int s=0; s<num_of_states; ++s)
		cout << p.value[s] << " ";
	if (time_horizon > 1) {
		cout << "\nChoice: ";
		for (int o=0; o<num_of_observations; ++o)
			cout << inv_action_map[V[time_horizon-1][p.choice[o]].action] << " ";
	}
	cout << endl;
}

double weakbound(vector <ptree>& X, vector <ptree>& Y) {
	double delta = -INF;
	for (int i=0; i<sz(X); ++i) {
		double mindiff = INF;
		for (int j=0; j<sz(Y); ++j) {
			double maxcomponent = -INF;
			for (int s=0; s<num_of_states; ++s)
				maxcomponent = max(maxcomponent, X[i].value[s] - Y[j].value[s]);
			mindiff = min(mindiff, maxcomponent);
		}
		delta = max(delta, mindiff);
	}
	return delta;
}

double difference(vector <ptree>& a, vector <ptree>& b) {
	// cout << sz(a) << " diff " << sz(b) << endl;
	double p1 = weakbound(a, b), p2 = weakbound(b, a);
	if (p1 > p2) return p1;
	return p2;
}

void prune(int t, vector<ptree>& X) {
	
	V[t].clear();
	set <int> sX;
	for (int i=0; i<sz(X); ++i)
		sX.insert(i);
	
	double delta;
	// cout << "\nentered pruned with TH " << t << " and size of Qa " << sz(X) << endl;
	FILE *fp;
	for (int i=0; i<sz(X); ++i) {
		// Opening the input file to LP solver
		fp = fopen("model.lp", "w");

		// Maximize the objective function
		fprintf(fp, "max: 1 x0;\n\n");

		// Constraint that all b[s] >= 0
		for (int s=1; s<=num_of_states; ++s)
			fprintf(fp, "x%d >= 0;\n", s);

		// Constraint that sum{b[s]} = 1
		fprintf(fp, "x1");
		for (int s=2; s<=num_of_states; ++s)
			fprintf(fp, " + x%d", s);
		fprintf(fp, " = 1.0;\n");

		// Constraint b.p >= delta + b.p' for all p' in X
		for (int j=0; j<sz(X); ++j) {
			if (sX.find(j) == sX.end()) continue;
			if (j == i) continue;
			fprintf(fp, "%f x1", X[i].value[1-1] - X[j].value[1-1]);
			for (int s=2; s<=num_of_states; ++s)
				fprintf(fp, " + %f x%d", X[i].value[s-1] - X[j].value[s-1], s);
			fprintf(fp, " - x0 >= 0;\n");
		}

		// x0 is a free variable
		fprintf(fp, "free x0;");

		// Closing the file
		fclose(fp);

		// Calling the solver
		system("lp_solve model.lp > out.lp");

		// getting the output of LP solver
		fp = fopen("out.lp", "r");
		int it;
		char *line, *obj_value;
		size_t len = 0;
		ssize_t read = getline(&line, &len, fp);
		if ((read = getline(&line, &len, fp)) != -1) {
			line = trim(line, (int)read);
			for (it=0; line[it]; it++)
				if (line[it] == ':')
					break;
			obj_value = line + it + 1;
			sscanf(obj_value, "%lf", &delta);
		}
		fclose(fp);
		if (delta > 0)
			V[t].push_back(X[i]);
		else
			sX.erase(i);
	}
	// cout << "returning from prune with size " << sz(V[t]) << endl;
}

double check_pnew(vector<ptree>& X, ptree& pnew, bstate& b) {
	// cout << "\nentered check pnew sz " << sz(X) << endl;
	// print_tree(pnew);
	FILE *fp = fopen("model.lp", "w");
	fprintf(fp, "max: 1 x0;\n\n");

	// Constraint that all b[s] >= 0
	for (int s=1; s<=num_of_states; ++s)
		fprintf(fp, "x%d >= 0;\n", s);

	// Constraint that sum{b[s]} = 1
	fprintf(fp, "x1");
	for (int s=2; s<=num_of_states; ++s)
		fprintf(fp, " + x%d", s);
	fprintf(fp, " = 1.0;\n");

	// Constraint b.pnew >= delta + b.p' for all p' in X
	for (int j=0; j<sz(X); ++j) {
		fprintf(fp, "%f x1", pnew.value[1-1] - X[j].value[1-1]);
		for (int s=2; s<=num_of_states; ++s)
			fprintf(fp, " + %f x%d", pnew.value[s-1] - X[j].value[s-1], s);
		fprintf(fp, " - x0 >= 0;\n");
	}

	// x0 is a free variable
	fprintf(fp, "free x0;\n");

	// Closing the file
	fclose(fp);

	// Calling the solver
	system("lp_solve model.lp > out.lp");

	// getting the output of LP solver
	fp = fopen("out.lp", "r");
	double ret;
	char *line, *obj_value;
	size_t len = 0;
	int it;

	// getting the value of objective function
	ssize_t read = getline(&line, &len, fp);
	if (!strncmp(line, "This problem is", 15)) {
		// cout << "check pnew failure" << endl;
		return -1; // Failure
	}

	if ((read = getline(&line, &len, fp)) != -1) {
		line = trim(line, (int)read);
		for (it=0; line[it]; it++) {
			if (line[it] == ':')
				break;
		}
		obj_value = line + it + 1;
		sscanf(obj_value, "%lf", &ret);
	}

	// getting the value of 'b'
	read = getline(&line, &len, fp);
	read = getline(&line, &len, fp);
	read = getline(&line, &len, fp);
	it = 0;
	while ((read = getline(&line, &len, fp)) != -1) {
		line = trim(line, (int)read);
		for (int itr=0; line[itr]; itr++)
			if (isspace(line[itr])) {
				obj_value = line + itr + 1;
				break;
			}
		sscanf(obj_value, "%lf", &b.b[it++]);
	}
	fclose(fp);
	return ret;
}

bool findb(int a, int t, vector<ptree>& Q, bstate& b) {
	vector <double> _alpha;
	bool ret = false;
	// cout << "\nentering findb with action " << inv_action_map[a] << " and TH " << t << endl;
	for (int i=0; !ret && i<sz(Q); ++i) {
		
		for (int o=0; !ret && o<num_of_observations; ++o) {
			
			for (int j=0; !ret && j<sz(V[t-1]); ++j) {
				
				ptree pnew = Q[i];
				
				// Altering one of the subtrees of Q[i] to make pnew
				if (pnew.choice[o] == j)
					continue;
				else
					pnew.choice[o] = j;

				// Updating the value vector of pnew
				pnew.value.assign(num_of_states, 0);
				for (int s=0; s<num_of_states; ++s) {
					if (R[s][a]) pnew.value[s] += R[s][a];
					for (int ob=0; ob<num_of_observations; ++ob) {
						back(V[t-1][pnew.choice[ob]].value, a, ob, _alpha);
						pnew.value[s] += discount * _alpha[s];
					}
				}

				// Checking if pnew is an improvement over Qa
				double delta = check_pnew(Q, pnew, b);
				if (delta > almost_zero) {
					// cout << "\ncheck_pnew returned " << delta << endl;
					// cout << "\nimprovement for bstate: ";
					// for (int s=0; s<num_of_states; ++s)
						// cout << b.b[s] << " "; cout << endl;
					ret = true;
				}
			}
		}
	}
	// cout << "\nfindb returning " << ret << endl;
	return ret;
}

void witness(int t, int a, vector<ptree>& Q) {
	bstate b;
	b.b.assign(num_of_states, 0);
	b.b[0] = 1;
	Q.push_back(besttree(b, a, V[t-1]));
	bool has_witness = findb(a, t, Q, b);
	ptree p;
	while (has_witness) {
		// cout << "has_witness " << t << endl;
		p = besttree(b, a, V[t-1]);
		Q.push_back(p);
		// cout << sz(Q) << endl;
		// cout << "\ncurrent set of policy trees:\n";
		// for (int i=0; i<sz(Q); ++i)
		// 	print_tree(Q[i]);
		has_witness = findb(a, t, Q, b);
		// cout << "findb has returned " << has_witness << endl;
	}
}

void solvePOMDP() {
	cout << "POMDP solver initiated...\n\n";
	time_horizon = 0;
	vector<ptree> Q, X;
	do {
		++time_horizon;
		X.clear();
		for (int a=0; a<num_of_actions; ++a) {
			Q.clear();
			witness(time_horizon, a, Q);
			X.insert(X.end(), Q.begin(), Q.end());
		}
		// V[time_horizon].insert(V[time_horizon].end(), X.begin(), X.end());
		prune(time_horizon, X);
		cout << "\nTime Horizon: " << time_horizon << "  Size of Value Function: " << sz(V[time_horizon]) << endl;
	} while ( (time_horizon < TIME_HORIZON) && !(difference(V[time_horizon-1], V[time_horizon]) <= 0) );
}
