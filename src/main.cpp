#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <ctype.h>
#include <string.h>
#include <queue>
#include <map>

using namespace std;

char *trim (char *s, int len) {
	// cout << "aya" << endl;
	int i = 0;
	for (; i<len && isspace(s[i]); ++i);
	int j = len - 1;
	for (; j>i && isspace(s[j]); j--);
	for (int k=i; k<=j; ++k) s[k-i] = s[k];
	s[j-i+1] = '\0';
	return s;
}

int main (int argc, char* argv[]) {
	
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
	bool has_discount, has_states, has_actions, has_observations, has_start;
	double discount;
	int num_of_states = 0, num_of_actions = 0, num_of_observations = 0;
	map <string, int> state_map, action_map, obs_map;
	char *s, *t;
	string str;
	queue <char*> Q;
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
		else if ((!strncmp("R", line, 1)) or (!strncmp("T", line, 1)) or (!strncmp("O", line, 1))) {
			char* p = new char[(int)strlen(line)];
			strcpy(p, (const char*)line);
			Q.push(p);
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
	exit(EXIT_SUCCESS);
}
