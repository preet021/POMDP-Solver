#include <bits/stdc++.h>

using namespace std;

typedef struct Policy_Tree {
	int action;
	vector <double> value;
}ptree;

char* trim(char *s, int len) {
	int i = 0;
	for (; i<len && isspace(s[i]); ++i);
	int j = len - 1;
	for (; j>i && isspace(s[j]); j--);
	for (int k=i; k<=j; ++k) s[k-i] = s[k];
	s[j-i+1] = '\0';
	return s;
}

int main () {
	FILE *fp = fopen("../test/zmdp-master/results/out.policy", "r");
	size_t len = 0;
	char *line = NULL;
	ssize_t read;
	int num_of_policy_trees, num_of_states, num_of_actions, num_of_observations;
	cin >> num_of_states >> num_of_actions >> num_of_observations;
	while ((read = getline(&line, &len, fp)) != -1) {
		line = trim(line, (int)read);
		if (line[0] == '#') continue;
		if ((int)strlen(line) == 0) continue;
		if (!strncmp("numPlanes", line, 9)) {
			sscanf(line+12, "%d", &num_of_policy_trees);
			break;
		}
	}
	read = getline(&line, &len, fp);
	vector <ptree> V(num_of_policy_trees);
	for (int i=0, o; i<num_of_policy_trees; ++i) {
		V[i].value.assign(num_of_states, 0);
		read = getline(&line, &len, fp); // {
		
		read = getline(&line, &len, fp); // action
		line = trim(line, (int)read);
		line[(int)strlen(line)-1] = '\0';
		sscanf(line+9, "%d", &V[i].action);

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
			V[i].value[s] = r;
		}
		read = getline(&line, &len, fp);
		read = getline(&line, &len, fp);
	}
	fclose(fp);
	std::vector<double> cur_b;
}
