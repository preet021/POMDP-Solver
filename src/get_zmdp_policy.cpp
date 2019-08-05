#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>

using namespace std;
#include "witness.h"

vector <ptree> zV;
int num_of_zmdp_policy_trees;

int get_zmdp_policy (char* policy) {
	// policy should be the path to output policy file

	// Checking whether path to output policy file is a valid path
	struct stat st;
	stat(policy, &st);
	bool is_reg_file = S_ISREG(st.st_mode);
	if (!is_reg_file) {
		fprintf(stderr, "ERROR: file %s is not a regular file\n", policy);
		exit(EXIT_FAILURE);
	}

	// Opening file
	cout << "Opening file..." << endl;
	FILE *fp = fopen(policy, "r");
	if (!fp) {
		fprintf(stderr, "ERROR: unable to open file %s\n", policy);
		exit(EXIT_FAILURE);
	}
	
	cout << "Reading file..." << endl;
	size_t len = 0;
	char *line = NULL;
	ssize_t read;
	while ((read = getline(&line, &len, fp)) != -1) {
		line = trim(line, (int)read);
		if (line[0] == '#') continue;
		if ((int)strlen(line) == 0) continue;
		if (!strncmp("numPlanes", line, 9)) {
			sscanf(line+12, "%d", &num_of_zmdp_policy_trees);
			break;
		}
	}
	read = getline(&line, &len, fp);
	zV.resize(num_of_zmdp_policy_trees);
	for (int i=0, o; i<num_of_zmdp_policy_trees; ++i) {
		zV[i].value.assign(num_of_states, 0);
		read = getline(&line, &len, fp); // {
		
		read = getline(&line, &len, fp); // action
		line = trim(line, (int)read);
		line[(int)strlen(line)-1] = '\0';
		sscanf(line+9, "%d", &zV[i].action);

		read = getline(&line, &len, fp); // numEntries
		line = trim(line, (int)read);
		line[(int)strlen(line)-1] = '\0';
		sscanf(line+13, "%d", &o);

		read = getline(&line, &len, fp);
		while (o--) {
			read = getline(&line, &len, fp);
			line = trim(line, (int)read);
			if (line[strlen(line)-1] == ',') 
				line[strlen(line)-1] = ' ';
			line = trim(line, (int)read);
			int s;
			double r;
			for (int it=0; line[it]; ++it)
				if (line[it] == ',') {
					sscanf(line+it+1, "%lf", &r);
					break;
				}
			for (int it=0; line[it]; ++it)
				if (line[it] == ',') {
					line[it] = '\0';
					break;
				}
			sscanf(line, "%d", &s);
			zV[i].value[s] = r;
		}
		read = getline(&line, &len, fp);
		read = getline(&line, &len, fp);
	}
	fclose(fp);
	cout << "File read." << endl;
}
