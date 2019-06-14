#include <ctype.h>

char* trim(char *s, int len) {
	int i = 0;
	for (; i<len && isspace(s[i]); ++i);
	int j = len - 1;
	for (; j>i && isspace(s[j]); j--);
	for (int k=i; k<=j; ++k) s[k-i] = s[k];
	s[j-i+1] = '\0';
	return s;
}
