#include <vector>
using namespace std;
#include "witness.h"

ptree choice(ptree p, int o, int a) {
	std::vector <ptree> S;
	ptree temp;
	std::vector <double> _alpha;
	for (int i=0; i<sz(V[time_horizon-1]); ++i) {
		temp.value = back(V[time_horizon-1][i].value, a, o, _alpha);
		S.push_back(temp);
	}
	return best(cur_b, S);
}
