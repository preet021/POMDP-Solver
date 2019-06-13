#ifndef WITNESS_POMDP
	
	#define WITNESS_POMDP
	
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

#endif