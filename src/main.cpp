#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <ctype.h>
#include <string.h>
#include <vector>
#include <queue>
#include <map>

#define sz(a) (int)(a).size()

using namespace std;
#include "witness.h"

const double INF = 1e18;
const int TIME_HORIZON = 100;
bool has_discount, has_states, has_actions, has_observations, has_start;
double **R, ***T, ***O;
double discount;
int num_of_states = 0, num_of_actions = 0, num_of_observations = 0, time_horizon = 0;
map <string, int> state_map, action_map, obs_map;
re rew;
te trns;
oe obs;
char *s, *t;
string str;
vector <ptree> V[TIME_HORIZON];
queue <re> Q_R;
queue <te> Q_T;
queue <oe> Q_O;
bstate cur_b;
ptree choice (ptree p, int o, int a);

double dot_product(vector<double>& a, vector<double>& b) {
	assert(sz(a) == sz(b));
	double result = 0;
	for (int i=0; i<sz(a); ++i)
		result += a[i] * b[i];
	return result;
}

ptree best(bstate& b, vector<ptree>& X) {
	ptree bestpol;
	bestpol.value.assign(num_of_states, -INF);
	double bestval = -INF, val;
	for (int pol=0; pol<sz(X); pol++) {
		val = dot_product(b.b, X[pol].value);
		if ((val > bestval) or ((val == bestval) /*and (X[pol] > bestpol)*/)) {
			bestval = val;
			bestpol = X[pol];
		}
	}
	return bestpol;
}

vector <double>& back(vector <double>& alpha, int a, int o, vector<double>& _alpha) {
	_alpha.clear();
	_alpha.assign(num_of_states, 0);
	for (int s=0; s<num_of_states; ++s)
		for (int _s=0; _s<num_of_states; _s++)
			_alpha[s] += alpha[_s] * T[s][a][_s] * O[_s][a][o];
	return _alpha;
}

vector <double>& value(ptree& p) {
	if (!sz(p.value)) {
		vector <double> _alpha;
		p.value.assign(num_of_states, 0);
		int a = p.action;
		for (int s=0; s<num_of_states; ++s) {
			p.value[s] = R[s][a];
			for (int o=0; o<num_of_observations; ++o) {
				ptree ret = choice(p, o, a);
				p.value[s] += discount * back(ret.value, a, o, _alpha)[s];
			}
		}
	}
	return p.value;
}

ptree choice(ptree p, int o, int a) {
	vector <ptree> S;
	ptree temp;
	vector <double> _alpha;
	for (int i=0; i<sz(V[time_horizon-1]); ++i) {
		temp.value = back(V[time_horizon-1][i].value, a, o, _alpha);
		S.push_back(temp);
	}
	return best(cur_b, S);
}

char* trim(char *s, int len) {
	int i = 0;
	for (; i<len && isspace(s[i]); ++i);
	int j = len - 1;
	for (; j>i && isspace(s[j]); j--);
	for (int k=i; k<=j; ++k) s[k-i] = s[k];
	s[j-i+1] = '\0';
	return s;
}

int main(int argc, char* argv[]) {
	
	// argv[1] should be the path to input file

	// Checking whether input file is provided
	if (argc != 2) {
		fprintf(stderr, "Usage:\n./binary path_to_input_file%s\n", argv[1]);
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
		}
		else if (!strncmp("R", line, 1)) {
			t = strtok(line+1, ": \n\t\v\r\f");
			int it = 0;
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
    	fprintf(stderr, "ERROR: initial belief state missing in input file %s\n", argv[1]);
    	// exit(EXIT_FAILURE);
    }
    else if (!has_actions) {
    	fprintf(stderr, "ERROR: set of actions missing in input file %s\n", argv[1]);
    	exit(EXIT_FAILURE);
    }
    else if (!has_observations) {
    	fprintf(stderr, "ERROR: set of observations missing in input file %s\n", argv[1]);
    	exit(EXIT_FAILURE);
    }
    
    // Storing the Reward function   R(s, a)
	R = new double*[num_of_states];
    while (!Q_R.empty()) {
    	re tp = Q_R.front();
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

    // Storing the Transition function   T(s, a, s')
	T = new double**[num_of_states];
    for (int i=0; i<num_of_states; ++i)
    	T[i] = new double*[num_of_actions];
    while (!Q_T.empty()) {
    	te tp = Q_T.front();
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

    // Storing the Observation function   O(a, s', o)
	O = new double**[num_of_actions];
    for (int i=0; i<num_of_actions; ++i)
    	O[i] = new double*[num_of_states];
    while (!Q_O.empty()) {
    	oe tp = Q_O.front();
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

	exit(EXIT_SUCCESS);
}
