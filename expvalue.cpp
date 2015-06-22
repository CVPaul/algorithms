#include "expvalue.h"
int cmp(char a, char b) {
	/*if (b == '(')
		return 1;
	else if ((b == '*' || b == '/') && (a == '+' || a == '-' || a == '('))
		return 1;
	else if ((b == '+' || b == '-') && (a == '('))
		return 1;
	else
		return 0;*/
	if (b == '*'&&a == '+')
		return 1;
	return 0;
}
void infix_to_suffix(char* infix, char* suffix) {
	int i, k, j = 0, top = 0;
	char stack[MAX_SIZE];

	for (i = 0; infix[i] != '\0'; i++) {
		if (infix[i] >= '0' && infix[i] <= '9') {
			suffix[j++] = infix[i];
		}
		else {
			if (i != 0 && infix[i - 1] >= '0' && infix[i - 1] <= '9') {
				suffix[j++] = ' ';
			}
			if (infix[i] == ')') {
				while (stack[top - 1] != '(') {
					suffix[j++] = stack[--top];
					suffix[j++] = ' ';
				}
				top--;
			}
			else if (top == 0 || cmp(stack[top - 1], infix[i])) {
				stack[top++] = infix[i];
			}
			else {
				while (!cmp(stack[top - 1], infix[i])) {
					suffix[j++] = stack[--top];
					suffix[j++] = ' ';
					if (top == 0)
						break;
				}
				stack[top++] = infix[i];
			}
		}
	}
	if (suffix[j - 1] != ' ') {
		suffix[j++] = ' ';
	}
	while (top != 0) {
		suffix[j++] = stack[--top];
		suffix[j++] = ' ';
	}
	suffix[j] = '\0';
}
long long suffix_value(char* suffix) {
	int i, j;
	char op;
	long long stack[MAX_SIZE];
	int top = 0;
	long long value = 0;
	for (i = 0; suffix[i] != '\0'; i++) {
		if (suffix[i] >= '0' && suffix[i] <= '9') {
			value = value * 10 + suffix[i] - '0';
		}
		else if (suffix[i] == ' ') {
			stack[top++] = (value%MOD);
			value = 0;
		}
		else {
			switch (suffix[i]) {
			case '+': value = stack[top - 2] + stack[top - 1]; break;
			//case '-': value = stack[top - 2] - stack[top - 1]; break;
			case '*': value = stack[top - 2] * stack[top - 1]; break;
			//case '/': value = stack[top - 2] / stack[top - 1]; break;
			default: break;
			}
			top -= 2;
		}
	}
	return stack[0]%MOD;
}