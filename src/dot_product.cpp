#include <assert.h>
#include <vector>
#define sz(a) (int)(a).size()

double dot_product(std::vector<double>& a, std::vector<double>& b) {
	assert(sz(a) == sz(b));
	double result = 0;
	for (int i=0; i<sz(a); ++i)
		result += b[i] * a[i];
	return result;
}