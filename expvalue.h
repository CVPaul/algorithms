#ifndef EXPVALUE
#define EXPVALUE

#define MAX_SIZE 10005
#define MOD 10000

int cmp(char a, char b);
void infix_to_suffix(char* infix, char* suffix);
long long suffix_value(char* suffix);

#endif/*expvalue.h*/