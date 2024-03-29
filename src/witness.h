#ifndef WITNESS_POMDP
	
	#define WITNESS_POMDP
	using namespace std;
	typedef struct Reward_Entry {
		int action;
		int state;
		double value;
	}re;

	typedef struct Transition_Entry {
		int action;
		int start_state;
		int end_state;
		double value;
	}te;

	typedef struct Observation_Entry {
		int action;
		int end_state;
		int obs;
		double value;
	}oe;

	typedef struct Belief_State {
		vector <double> b;
	}bstate;

	typedef struct Policy_Tree {
		int action;
		vector <double> value;
		vector <int> choice;
	}ptree;

	char* trim(char *s, int len);
	double dot_product(vector<double>& a, vector<double>& b);
	int get_zmdp_policy (char* policy);
	extern vector <struct Policy_Tree> zV;
	extern int num_of_states;
	extern int num_of_actions;
	extern int num_of_observations;
	extern int num_of_zmdp_policy_trees;

#endif