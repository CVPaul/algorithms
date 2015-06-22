#include "leetcode.h"

string Solution::num2str(int num)
{
	string re;

	if (num == 0) return "0";

	while (num > 0)
	{
		char ch = '0' + num % 10;
		num /= 10;
		re.insert(re.begin(), ch);
	}
	return re;
}
bool Solution::cmp_str(string s1, string s2)
{
	string t1 = s1; t1.append(s2);// be careful!!!!
	string t2 = s2; t2.append(s1);
	return (t1.compare(t2) < 0);
}
string Solution::largestNumber(vector<int> &num)
{
	vector<string> strs(num.size());
	for (int k = 0; k < num.size(); k++)
	{
		strs[k] = num2str(num[k]);
	}
	sort(strs.begin(), strs.end(), cmp_str);
	string result;
	for (int k = num.size() - 1; k >= 0; k--)
	{
		result.append(strs[k]);
	}
	int pos = 0;
	while (pos < result.length() && result[pos] == '0')pos++;
	if (pos >= result.length()) return "0";
	else
	{
		return result.substr(0, result.length() - pos);
	}
}
vector<vector<int> > Solution::threeSum(vector<int>& num)
{
	sort(num.begin(), num.end());
	vector<vector<int> > res;
	int size = num.size();
	vector<int> temp(3);
	for (int i = 0; i < size - 2; i++) {
		if (i == 0 || (i > 0 && num[i] != num[i - 1])) {
			int lo = i + 1, hi = size - 1, sum = 0 - num[i];
			while (lo < hi) {
				if (num[lo] + num[hi] == sum) {
					temp[0] = num[i]; temp[1] = num[lo]; temp[2] = num[hi];
					res.push_back(temp);
					while (lo < hi && num[lo] == num[lo + 1]) lo++;
					while (lo < hi && num[hi] == num[hi - 1]) hi--;
					lo++; hi--;
				}
				else if (num[lo] + num[hi] < sum) lo++;
				else hi--;
			}
		}
	}
	return res;
}
void Solution::reorderList(ListNode *head)
{
	if (!head || !head->next || !head->next->next) return;
	ListNode* fast = head->next;
	ListNode* slow = head;
	while (fast&&fast->next) // split
	{
		slow = slow->next;
		fast = fast->next->next;
	}
	fast = slow->next;
	slow->next = NULL; // head is the first half fast is the second half
	ListNode* temp, reverse(0);
	while (fast) // reverse the second half
	{
		if (reverse.next)
		{
			temp = fast;
			fast = fast->next;
			temp->next = reverse.next;
			reverse.next = temp;
		}
		else
		{
			reverse.next = fast;
			fast = fast->next;
			reverse.next->next = NULL;
		}
	}
	slow = reverse.next;// the second reversed half
	reverse.next = head;
	fast = head;
	head = head->next;
	while (head&&slow) // merge
	{
		fast->next = slow;
		slow = slow->next;
		fast = fast->next;

		fast->next = head;
		head = head->next;
		fast = fast->next;
	}
	if (head)fast->next = head;
	if (slow)fast->next = slow;
	head = reverse.next; // return;
}
ListNode *Solution::detectCycle(ListNode *head)
{
	ListNode *fast = head;
	ListNode *slow = head;
	while (fast&&fast->next)
	{
		fast = fast->next->next;
		slow = slow->next;
		if (fast == slow) break;
	}
	if (!fast || !fast->next) return NULL;
	/**
	当fast若与slow相遇时，slow肯定没有走遍历完链表，而fast已经在环内循环了n圈(1<=n)。假设slow走了s步，
	则fast走了2s步（fast步数还等于s 加上在环上多转的n圈），设环长为r，则：
	2s = s + nr
	s= nr

	设整个链表长L，入口环与相遇点距离为x，起点到环入口点的距离为a。
	a + x = nr
	a + x = (n – 1)r +r = (n-1)r + L - a
	a = (n-1)r + (L – a – x)

	(L – a – x)为相遇点到环入口点的距离，由此可知，从链表头到环入口点等于(n-1)循环内环+相遇点到环入口点，
	于是我们从链表头、与相遇点分别设一个指针，每次各走一步，两个指针必定相遇，且相遇第一点为环入口点
	*/
	while (head != slow)
	{
		head = head->next;
		slow = slow->next;
	}
}
bool Solution::hasCycle(ListNode *head)
{
	ListNode *fast = head;
	ListNode *slow = head;
	while (fast&&fast->next)
	{
		fast = fast->next->next;
		slow = slow->next;
		if (fast == slow) break;
	}
	if (!fast || !fast->next) return false;
	return true;
}
// this answer is not accept by the OJ I don't know why
//bool Solution::wordBreak(string s, unordered_set<string> &dict)
//{
//	unordered_set<string>::iterator iter = dict.begin();
//	while (iter != dict.end())
//	{
//		string patent = *iter;
//		string::size_type pos = s.find(patent);
//		while (pos!=string::npos)
//		{
//			s.erase(pos, patent.length());
//			pos = s.find(patent);
//		}
//		iter++;
//	}
//	if (s.empty()) return true;
//	return false;
//}
bool Solution::wordBreak(string s, unordered_set<string> &dict)
{
	int len = s.length();
	vector<bool> good(len + 1, false);
	good[len] = true;
	for (int i = len - 1; i >= 0; i--)
	{
		for (int j = i + 1; j <= len; j++)
		{
			if (good[j])
			{
				if (dict.find(s.substr(i, j - i)) != dict.end())
				{
					good[i] = true;
					break;
				}
			}
		}
	}
	return good[0];
}
int Solution::singleNumber(int A[], int n)
{
	sort(A, A + n);
	int pre = 0, cur = 0;
	while (cur < n)
	{
		if (A[cur] == A[pre])cur++;
		else
		{
			if ((cur - pre) != 3)
				break;
			pre = cur;
		}
	}
	return A[pre];
}
/*
The basic idea is to implement a modulo-3 counter (to count how many times "1" occurs)
for each bit position.
Such modulo-3 counter needs two bits (B1,B0) to represent.
(B1,B0):
(0, 0) : '1' occurs zero times after last resetting,
(0, 1) : '1' occurs one times after last resetting,
(1, 0) : '1' occurs two times after last resetting,
(1, 1) : '1' occurs three times after last resetting, then we need to reset the counter to (0,0)
So to implement such modulo-3 counters, we need three variables (b0, b1, reset)
The n-th bit of b0 is the B0 bit of the modulo-3 counter for the n-th bit (n=0..31 assuming int
is 32 bit)
The n-th bit of b1 is the B1 bit of the modulo-3 counter for the n-th bit (n=0..31 assuming int
is 32 bit)
The n-th bit of reset is the reset flag of the modulo-3 counter for the n-th bit (n=0..31 assuming
int is 32 bit),

- b0: can be easily implemented with XOR bit operation,  as b0 = b0^ A[i]
- b1: B1 will only be set to 1, when B0 (of the n-th bit counter) =1 and the n-th bit of A[i] = 1,
and stay '1' until it is reseted. So b1 |=  b0 & A[i];
- The reset flag is set when (B1, B0) = (1,1). So, reset = b0 & b1;
- The reset operation can be done by b0 = b0 ^ reset and b1 = b1 ^ reset;

After updating the b0, b1, reset with all A[], the b0 will be the final result since if the n-th bit
of the to-be-found element is 1, then the times of '1' occurs on the n-th bit is 3*x+1, which is 1
after the modulo 3 opertation.
*/
/*int singleNumber(int A[], int n) {
int b0 = 0, b1 = 0, reset = 0;
int i;

if (n>0)
{
for (i = 0; i<n; i++)
{
b1 |= (b0 & A[i]);
b0 = b0 ^ A[i];
reset = b1 & b0;
b1 = b1 ^ reset;
b0 = b0 ^ reset;
}

return b0;
}
}*/

//------------------------------------------------------------------------------------------------------------
/*
//--  Another version with explicit modulo-3 counters implemented, just for reference-----//
const int int_bitwidth = 32;
int singleNumber(int A[], int n) {
int mod3Counter[int_bitwidth];
int i,j;
unsigned int temp = 0;

if(n>0)
{
for(i=0; i<int_bitwidth; i++) mod3Counter[i] = 0;

for(i=0; i<n; i++)
{
temp = (unsigned int) A[i];
for(j=0; j<int_bitwidth; j++)
{
if(temp & 0x1)
{
mod3Counter[j] = (mod3Counter[j] + 1) % 3;
}
temp = temp>>1;
}

temp =0;
for(j=0; j<int_bitwidth; j++)
{
temp = temp << 1;
if(mod3Counter[int_bitwidth - 1 -j])
{
temp = temp | 0x1;
}
}
}
return (int)temp;

}
}
*/
ListNode *Solution::addTwoNumbers(ListNode *l1, ListNode *l2)
{
	if (!l1) return l2;
	if (!l2) return l1;

	ListNode* p1 = l1;
	ListNode* p2 = l2;
	while (p1&&p2)
	{
		p1->val += p2->val;
		p1 = p1->next;
		p2 = p2->next;
	}
	int base = 0, temp;
	if (p1)
	{
		p1 = l1;
		base = 0, temp;
		while (p1->next)
		{
			temp = (p1->val + base);
			base = temp / 10;
			p1->val = temp % 10;
			p1 = p1->next;
		}
		temp = (p1->val + base); // last one
		base = temp / 10;
		p1->val = temp % 10;
		if (base)
			p1->next = new ListNode(base);
		return l1;
	}
	else if (p2)
	{
		p1 = l1;
		p2 = l2;
		base = 0, temp;
		while (p1&&p2)
		{
			temp = (p1->val + base);
			base = temp / 10;
			p2->val = temp % 10;
			p1 = p1->next;
			p2 = p2->next;
		}
		while (p2->next)
		{
			temp = (p2->val + base);
			base = temp / 10;
			p2->val = temp % 10;
			p2 = p2->next;
		}
		temp = (p2->val + base); // last one
		base = temp / 10;
		p2->val = temp % 10;
		if (base)
			p2->next = new ListNode(base);
		return l2;
	}
	else
	{
		p1 = l1;
		base = 0, temp;
		while (p1->next)
		{
			temp = (p1->val + base);
			base = temp / 10;
			p1->val = temp % 10;
			p1 = p1->next;
		}
		temp = (p1->val + base); // last one
		base = temp / 10;
		p1->val = temp % 10;
		if (base)
			p1->next = new ListNode(base);
		return l1;
	}
	/*
	ListNode head(0);
	ListNode* p=&head;
	int base=0;
	while(l1&&l2)
	{
	int temp=l1->val+l2->val+base;
	p->next=new ListNode(temp%10);
	base=temp/10;
	l1=l1->next;l2=l2->next;
	p=p->next;
	}
	while(l1)
	{
	int temp=l1->val+base;
	p->next=new ListNode(temp%10);
	base=temp/10;
	l1=l1->next;
	p=p->next;
	}
	while(l2)
	{
	int temp=l2->val+base;
	p->next=new ListNode(temp%10);
	base=temp/10;
	l2=l2->next;
	p=p->next;
	}
	if(base)
	p->next=new ListNode(base);
	return head.next;
	*/
}
ListNode *Solution::insertionSortList(ListNode *head)
{
	if (!head || !head->next) return head;
	ListNode hd(0), *p, *q, *temp;;
	hd.next = head;
	p = head->next; hd.next->next = NULL;
	while (p)
	{
		q = &hd;
		while (q->next&&q->next->val < p->val)q = q->next;
		temp = p; p = p->next;
		temp->next = q->next;
		q->next = temp;
	}
	return hd.next;
}
int Solution::ladderLength(string start, string end, unordered_set<string> &dict)
{
	if (dict.empty() || dict.find(start) == dict.end() || dict.find(end) == dict.end())
		return 0;

	queue<string> q;
	q.push(start);
	unordered_map<string, int> visited;  // visited track the distance
	visited[start] = 1;
	unordered_set<string> unvisited = dict;  // unvisited prevent searching through the whole dict
	unvisited.erase(start);

	while (!q.empty()) {
		string word = q.front(); q.pop();
		auto itr = unvisited.begin();
		while (itr != unvisited.end()) {
			string adjWord = *itr;
			if (oneCharDiff(word, adjWord)) {
				visited[adjWord] = visited[word] + 1;
				if (adjWord == end)
					return visited[adjWord];
				itr = unvisited.erase(itr);  // tricky here
				q.push(adjWord);
			}
			else
				++itr;
		}
	}
	return 0;
}
int Solution::lengthOfLongestSubstring(string s)
{
	/*if (s.length() < 2)return s.length();

	int lens;
	lens = 1; int max_len = 1;
	for (int j = 1; j <s.length(); j++)
	{
		string sub = s.substr(j - lens, lens);
		string::size_type pos = sub.find(s[j]);
		if (pos == string::npos)
		{
			lens = lens + 1;
			max_len = std::max(max_len, lens);
		}
		else
		{
			lens = lens - pos;
		}
	}
	return max_len;*/
	if (s.length() < 2)return s.length();

	int lens = 0, max_len = 0;
	int start = 0;
	vector<int> mask(256, -1);
	for (int j = 0; j < s.length(); j++)
	{
		if (mask[s[j]]>=start)
			start = mask[s[j]] + 1;
		mask[s[j]] = j; 
		max_len = max(max_len, j - start + 1);
	}
	return max_len;
}
string Solution::longestPalindrome(string s)
{
	//if (s.length() < 2) return s;
	//int len = s.length();

	//int left, right;
	//int index = 1, flag = 0, pos = 1;
	//int tlen = 0, max_p = 0,max_flag=0;

	//while (pos < len)
	//{
	//	left = pos - 1;
	//	right = pos;
	//	if (s[left] == s[right])// deal with odd and even
	//	{
	//		flag = 0;
	//		tlen = 0;
	//		while (left >= 0 && right < len&&s[left] == s[right])
	//		{
	//			tlen++;
	//			left--;
	//			right++;
	//		}
	//		if ((2 * tlen + flag) >(2 * max_p + max_flag))
	//		{
	//			max_p = tlen;
	//			max_flag = flag;
	//			index = pos;
	//		}
	//	}
	//	left = pos - 1;
	//	right = pos+1;
	//	if (s[left] == s[right])// deal with odd and even
	//	{
	//		tlen = 0;
	//		flag = 1;

	//		while (left >= 0 && right < len&&s[left] == s[right])
	//		{
	//			tlen++;
	//			left--;
	//			right++;
	//		}
	//		if ((2 * tlen + flag) >(2 * max_p + max_flag))
	//		{
	//			max_p = tlen;
	//			max_flag = flag;
	//			index = pos;
	//		}
	//	}
	//	pos++;
	//}
	//if (max_p == 0 && max_flag == 0)
	//	return s.substr(0,1);
	//return s.substr(index - max_p, 2 * max_p + max_flag);
	//-----------------Manacher Algorithm----------------------------------------------
	int len = 2 * (s.length() + 1);
	string temp(len, '#');
	temp[0] = '$';
	vector<int> lens(len, 0);
	for (int k = 1; k <= s.length(); k++)
	{
		temp[2 * k] = s[k - 1];
	}
	int id = 0, mx = 0, max_len = 0, max_ind = 0;
	for (int i = 1; i < len; i++)
	{
		if (mx>i)
		{
			lens[i] = min(mx - i, lens[2 * id - i]);
		}
		else
			lens[i] = 1;
		while (temp[i + lens[i]] == temp[i - lens[i]])
		{
			lens[i]++;
		}
		if (i + lens[i] > mx)
		{
			id = i;
			mx = i + lens[i];
		}
		if (lens[i] > max_len)
		{
			max_len = lens[i];
			max_ind = i;
		}
	}
	string ans(max_len - 1, '#');
	int j = 0;
	for (int k = max_ind - max_len + 1; k < max_ind + max_len - 1; k++)
	{
		if (temp[k] != '#')
		{
			ans[j++] = temp[k];
		}
	}
	return ans;
}
int Solution::strStr(char *haystack, char *needle)
{
	// inplement the sunday algorithm:
	int skip[256];
	memset(skip, -1, sizeof(skip));
	int len_h = strlen(haystack);
	int len_n = strlen(needle);
	for (int k = 0; k < len_n; k++)
	{
		skip[needle[k]] = k;
	}
	int pos = 0;
	while (pos < (len_h - len_n + 1))
	{
		int i = pos;
		int j;
		for (j = 0; j < len_n; j++)
		{
			if (haystack[i] != needle[j])
			{
				pos += len_n - skip[haystack[pos + len_n]];
				break;
			}
			i++;
		}
		if (j == len_n) return pos;
	}
	return -1;
}
//double Solution::pow_helper(long double x, long long n)
//{
//	if (n == 0) return 1.0;
//	if (n == 1) return x;
//	long long left = n / 2;
//	return pow_helper(x, left)*pow_helper(x, n - left);
//}
//double Solution::pow(double x, int n) {
//	if (n < 0) return pow(1.0 / x, -n);
//	return pow_helper(x, n);
//}
double Solution::pow_helper(long double x, long long n)
{
	if (n == 0) return 1.0;
	if (n == 1) return x;
	vector<bool> mp;
	long double temp = x;
	long double re = 1.0;
	while (n>0)
	{
		if (n % 2 == 1)
		{
			re *= temp;
		}
		temp = temp * temp;
		n =n>>1;
	}
	return re;
}
double Solution::pow(double x, int n) {
	if (n < 0) return pow_helper(1.0 / x, -n);
	return pow_helper(x, n);
}

void  Solution::tree_list(TreeNode* root, vector<int>& list)
{
	if (!root) return;
	tree_list(root->left, list);
	list.push_back(root->val);
	tree_list(root->right, list);
}
bool isValidBST_helper(TreeNode* root, long long min, long long max)
{
	if (!root)return true;
	if (root->val >= max || root->val <= min)return false;
	return isValidBST_helper(root->left, min, root->val) && isValidBST_helper(root->right, root->val, max);
}
bool Solution::isValidBST(TreeNode *root)
{
	return isValidBST_helper(root, LLONG_MIN, LLONG_MAX);
}
void Solution::gp_dfs(int left, int right, string cur, int num)
{
	printf("%d %d %s\n", left, right, cur.c_str());
	if (left == num && right == num){
		parenthesis.push_back(cur);
		return;
	}
	if (left != num)//no need to check left brace 
		gp_dfs(left + 1, right, cur + "(", num);
	if (right<left)
		gp_dfs(left, right + 1, cur + ")", num);
}
vector<string> Solution::generateParenthesis(int n)
{
	if (!n) return vector<string>();
	//this->num = n;
	gp_dfs(1, 0, "(", n);
	return parenthesis;
}
int Solution::minimumTotal(vector<vector<int> > &triangle)
{
	int layers = triangle.size();
	if (!layers) return 0;
	if (layers == 1)return triangle[0][0];

	for (int k = 0; k < layers - 1; k++)
	{
		triangle[k + 1][0] += triangle[k][0];
		triangle[k + 1][k + 1] += triangle[k][k];
		for (int s = 1; s < triangle[k + 1].size() - 1; s++)
		{
			triangle[k + 1][s] += min(triangle[k][s - 1], triangle[k][s]);
		}
	}
	int min_val = INT_MAX;
	for (int k = 0; k < triangle[layers - 1].size(); k++)
	{
		if (triangle[layers - 1][k] < min_val)
			min_val = triangle[layers - 1][k];
	}
	return min_val;
}
bool Solution::canJump(int A[], int n)
{
	if (n < 1) return true;
	int pos = n - 2, dest = n - 1;
	while (pos >= 0)
	{
		if (pos + A[pos] >= dest)
		{
			dest = pos;
		}
		pos--;
	}
	if (dest == 0) return true;
	return false;
}
void permute_helper(vector<int> nums,int pos,vector<vector<int>>& res)
{
	if (pos == nums.size()-1)
	{
		res.push_back(nums);
		return;
	}
	for (int i = pos; i < nums.size(); i++)
	{
		if (i == pos || nums[i] != nums[i - 1])
		{
			swap(nums[i], nums[pos]);
			permute_helper(nums, pos + 1, res);
		}
		//swap(nums[i], nums[pos]);
	}
}
vector<vector<int> > Solution::permute(vector<int> &num)
{
	/*int len = num.size();
	vector<vector<int> > re;
	if (len <= 1)
	{
		re.push_back(num);
		return re;
	}
	vector<int> temp(num.begin(), num.end() - 1);
	re = permute(temp);
	vector<int> cur;
	int len_pre = re.size();
	for (int k = 0; k < len_pre; k++)
	{
		for (int j = 0; j <re[k].size(); j++)
		{
			cur = re[k];
			cur.insert(cur.begin() + j, *(num.end() - 1));
			re.push_back(cur);
		}
		re[k].push_back(*(num.end() - 1));
	}
	return re;*/
	sort(num.begin(), num.end());
	vector<vector<int>> res;
	permute_helper(num, 0, res);
	return res;

}
int Solution::romanToInt(string s) {
	int len = s.length();
	if (len<1) return 0;
	int re = romanCharToInt(s[0]);
	for (int k = 1; k<len; k++)
	{
		re += romanCharToInt(s[k]);
		if (romanCharToInt(s[k])>romanCharToInt(s[k - 1]))
		{
			re -= 2 * romanCharToInt(s[k - 1]);
		}
	}
	return re;
}
inline int Solution::romanCharToInt(char c) {
	switch (c) {
	case 'I':   return 1;
	case 'V':   return 5;
	case 'X':   return 10;
	case 'L':   return 50;
	case 'C':   return 100;
	case 'D':   return 500;
	case 'M':   return 1000;
	default:    return 0;
	}
}
string Solution::intToRoman(int num)
{
	string romanPieces[] = { "", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", \
		"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC", \
		"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM", \
		"", "M", "MM", "MMM", "MMMM" };
	return romanPieces[num / 1000 + 30] + romanPieces[(num / 100) % 10 + 20]\
		+ romanPieces[(num / 10) % 10 + 10] + romanPieces[num % 10];
}
int Solution::searchInsert(int A[], int n, int target) {
	int lo = 0, hi = n - 1;
	int mid;
	while (lo<hi)
	{
		mid = (lo + hi) / 2;
		if (A[mid]>target)
		{
			hi = mid - 1;
		}
		else if (A[mid]<target)
		{
			lo = mid + 1;
		}
		else
			return mid;
	}
	if (A[lo]<target)
		return lo + 1;
	else
		return lo;
}
vector<int> searchRange(int A[], int n, int target)
{
	int lo = 0, hi = n - 1;
	int mid;
	while (A[lo] != A[hi])
	{
		mid = (lo + hi) / 2;
		if (A[mid]>target)
		{
			hi = mid - 1;
		}
		else if (A[mid]<target)
		{
			lo = mid + 1;
		}
		else
		{
			if (A[hi] != target)hi--;
			if (A[lo] != target)lo++;
		}
	}
	vector<int> range(2);
	if (A[lo] != target)
	{
		range[0] = -1;
		range[1] = -1;
		return range;
	}
	else
	{
		range[0] = lo;
		range[1] = hi;
		return range;
	}
}
bool Solution::isValid(string s)
{
	stack<char> st;
	for (int k = 0; k < s.length(); k++)
	{
		switch (s[k])
		{
		case '(': case '[': case '{':
			st.push(s[k]); break;
		case ')':
			if (st.empty() || st.top() != '(') return false;
			st.pop(); break;
		case ']':
			if (st.empty() || st.top() != '[') return false;
			st.pop(); break;
		case '}':
			if (st.empty() || st.top() != '{') return false;
			st.pop(); break;
		default:
			break;
		}
	}
	if (st.empty()) return true;
	return false;
}
void Solution::sortColors(int A[], int n)
{
	int lo = 0, hi = n - 1;
	int i = lo;
	while (i <= hi)
	{
		if (A[i] < 1)
			swap(A[lo++], A[i++]);
		else if (A[i]>1)
			swap(A[hi--], A[i]);// be careful
		else
			i++;
	}
}
int Solution::maxArea(vector<int> &height)
{
	int max_val = 0, lo = 0, hi = height.size() - 1;
	while (lo<hi)
	{
		int he;
		if (height[lo]<height[hi])he = height[lo++];
		else he = height[hi--];
		max_val = max(max_val, he*(hi - lo + 1));
	}
	return max_val;
}
int Solution::threeSumClosest(vector<int> &num, int target)
{
	if (num.size()<3)
		return accumulate(num.begin(), num.end(), 0);
	sort(num.begin(), num.end());
	int gap = INT_MAX, ans;
	for (int i = 0; i<num.size() - 2; i++)
	{
		int lo = i + 1, hi = num.size() - 1;
		while (lo<hi)
		{
			int sum = num[i] + num[lo] + num[hi];
			if (abs(target - sum)<gap)	{
				ans = sum;
				gap = abs(target - sum);
			}
			if (sum == target) return target;
			else if (sum>target) hi--;
			else lo++;
		}
	}
	return ans;
}
void Solution::lcs_dfs(int count, string cur, string digits, int num)
{
	if (count == num)
	{
		lcs.push_back(cur);
		return;
	}
	switch (digits[0])
	{
	case '2':
		lcs_dfs(count + 1, cur + "a", digits.substr(1, digits.length() - 1), num);
		lcs_dfs(count + 1, cur + "b", digits.substr(1, digits.length() - 1), num);
		lcs_dfs(count + 1, cur + "c", digits.substr(1, digits.length() - 1), num);
		break;
	case '3':
		lcs_dfs(count + 1, cur + "d", digits.substr(1, digits.length() - 1), num);
		lcs_dfs(count + 1, cur + "e", digits.substr(1, digits.length() - 1), num);
		lcs_dfs(count + 1, cur + "f", digits.substr(1, digits.length() - 1), num);
		break;
	case '4':
		lcs_dfs(count + 1, cur + "g", digits.substr(1, digits.length() - 1), num);
		lcs_dfs(count + 1, cur + "h", digits.substr(1, digits.length() - 1), num);
		lcs_dfs(count + 1, cur + "i", digits.substr(1, digits.length() - 1), num);
		break;
	case '5':
		lcs_dfs(count + 1, cur + "j", digits.substr(1, digits.length() - 1), num);
		lcs_dfs(count + 1, cur + "k", digits.substr(1, digits.length() - 1), num);
		lcs_dfs(count + 1, cur + "l", digits.substr(1, digits.length() - 1), num);
		break;
	case '6':
		lcs_dfs(count + 1, cur + "m", digits.substr(1, digits.length() - 1), num);
		lcs_dfs(count + 1, cur + "n", digits.substr(1, digits.length() - 1), num);
		lcs_dfs(count + 1, cur + "o", digits.substr(1, digits.length() - 1), num);
		break;
	case '7':
		lcs_dfs(count + 1, cur + "p", digits.substr(1, digits.length() - 1), num);
		lcs_dfs(count + 1, cur + "q", digits.substr(1, digits.length() - 1), num);
		lcs_dfs(count + 1, cur + "r", digits.substr(1, digits.length() - 1), num);
		lcs_dfs(count + 1, cur + "s", digits.substr(1, digits.length() - 1), num);
		break;
	case '8':
		lcs_dfs(count + 1, cur + "t", digits.substr(1, digits.length() - 1), num);
		lcs_dfs(count + 1, cur + "u", digits.substr(1, digits.length() - 1), num);
		lcs_dfs(count + 1, cur + "v", digits.substr(1, digits.length() - 1), num);
		break;
	case '9':
		lcs_dfs(count + 1, cur + "w", digits.substr(1, digits.length() - 1), num);
		lcs_dfs(count + 1, cur + "x", digits.substr(1, digits.length() - 1), num);
		lcs_dfs(count + 1, cur + "y", digits.substr(1, digits.length() - 1), num);
		lcs_dfs(count + 1, cur + "z", digits.substr(1, digits.length() - 1), num);
		break;
	default:
		break;
	}
}
vector<string> Solution::letterCombinations(string digits)
{
	if (digits.length() < 1) return lcs;
	lcs_dfs(0, "", digits, digits.length());
	return lcs;
}
ListNode *Solution::swapPairs(ListNode *head)
{
	/*if (!head || !head->next) return head;
	ListNode *temp = head->next->next;
	ListNode* p = head->next;
	p->next = head;
	head->next = swapPairs(temp);
	return p;*/
	if (!head||!head->next)return head;
	ListNode* p1 = head, *p2 = head->next;
	head = p2; ListNode* newHead = NULL;
	while (p1&&p2)
	{
		newHead = p2->next;
		p2->next = p1;
		if (!newHead||!newHead->next)
		{
			p1->next = newHead;
			break;
		}
		else
		{
			p1->next = newHead->next;
			p1 = newHead;
			p2 = newHead->next;
		}
	}
	return head;
}
vector<int> Solution::grayCode(int n) {
	gray_code.push_back(0);
	for (int k = 0; k < n; k++)
	{
		int s = gray_code.size() - 1;
		while (s >= 0)
		{
			int code = gray_code[s] | (1 << k);
			gray_code.push_back(code);
			s--;
		}
	}
	return gray_code;
}
vector< vector<int> > Solution::fourSum(vector<int>& num, int target)
{
	sort(num.begin(), num.end());
	int len = num.size();
	vector< vector<int> > res;
	if (len < 4) return res;
	//int ans = num[0] + num[1] + num[2] + num[3];
	for (int i = 0; i < len - 3; i++)
	{
		if (!(i == 0 || num[i] != num[i - 1])) continue;

		for (int j = i + 1; j < len - 2; j++)
		{
			if (!(j == i + 1 || num[j] != num[j - 1])) continue;

			int lo = j + 1, hi = len - 1, sum = num[i] + num[j] + num[lo] + num[hi];
			while (lo < hi)
			{
				if (sum == target)
				{
					vector<int> temp(4);
					temp[0] = num[i]; temp[1] = num[j]; temp[2] = num[lo]; temp[3] = num[hi];
					res.push_back(temp);
					while (lo<hi&&num[lo + 1] == num[lo])lo++;
					while (lo<hi&&num[hi - 1] == num[hi])hi--;

					lo++;// one step at least
					hi--;// one step at least
				}
				else if (sum < target)
				{
					while (lo<hi&&num[lo + 1] == num[lo])lo++;
					lo++;// one step at least
				}
				else
				{
					while (lo<hi&&num[hi - 1] == num[hi])hi--;
					hi--;// one step at least
				}
				sum = num[i] + num[j] + num[lo] + num[hi];
			}
		}
	}
	return res;
}
int Solution::divide(int dividend, int divisor)
{
	long long res = dividelong(dividend, divisor);
	if (res > INT_MAX) return INT_MAX;
	if (res < INT_MIN) return INT_MIN;
	return res;
}
long long Solution::dividelong(long long dividend, long long divisor)
{
	int symb = 1;
	if (dividend < 0)
	{
		dividend = -dividend;
		symb *= -1;
	}
	if (divisor < 0)
	{
		divisor = -divisor;
		symb *= -1;
	}
	if (dividend < divisor) return 0;

	long long res = 0;
	while (divisor <= dividend)
	{
		long long temp = 1;
		long long s = divisor;
		while ((s + s) <= dividend)
		{
			s = s + s;
			temp *= 2;
		}
		res += temp;
		dividend -= s;
	}
	return res*symb;
}
void Solution::nextPermutation(vector<int> &num)
{
	int pos = num.size() - 1;
	if (pos < 1)return;
	while (pos>0 && num[pos] <= num[pos - 1])pos--;
	sort(num.begin() + pos, num.end());
	if (pos)
	{
		int pt = pos;
		while (num[pt] <= num[pos - 1])pt++;
		swap(num[pos - 1], num[pt]);
	}
}
vector<vector<int> > Solution::combinationSum_helper(vector<int> &candidates, int start, int target)
{
	if (target == 0) return vector< vector<int> >();
	vector< vector<int> > cur;
	for (int k = start; k < candidates.size(); k++)
	{
		vector< vector<int> > temp;
		if (target == candidates[k])
		{
			vector<int> stemp;
			stemp.push_back(candidates[k]);
			cur.push_back(stemp);
		}
		else if (candidates[k] < target)
		{
			temp = combinationSum_helper(candidates, k, target - candidates[k]);
			for (int i = 0; i < temp.size(); i++)
			{
				temp[i].push_back(candidates[k]);
			}
			cur.insert(cur.end(), temp.begin(), temp.end());
		}
	}
	return cur;
}
vector<vector<int> > Solution::combinationSum(vector<int> &candidates, int target)
{
	sort(candidates.begin(), candidates.end(), greater<int>());
	vector<vector<int> > res = combinationSum_helper(candidates, 0, target);
	return res;
}
vector<vector<int> > Solution::combinationSum2_helper(vector<int> &candidates, int start, int target)
{
	if (target == 0) return vector< vector<int> >();
	vector< vector<int> > cur;
	for (int k = start; k < candidates.size(); k++)
	{
		if (k == start || candidates[k - 1] != candidates[k])
		{
			vector< vector<int> > temp;
			if (target == candidates[k])
			{
				vector<int> stemp;
				stemp.push_back(candidates[k]);
				cur.push_back(stemp);
			}
			else if (candidates[k] < target)
			{
				temp = combinationSum2_helper(candidates, k + 1, target - candidates[k]);
				for (int i = 0; i < temp.size(); i++)
				{
					temp[i].push_back(candidates[k]);
				}
				cur.insert(cur.end(), temp.begin(), temp.end());
			}
		}
	}
	return cur;
}
vector<vector<int> > Solution::combinationSum2(vector<int> &candidates, int target)
{
	sort(candidates.begin(), candidates.end(), greater<int>());
	vector<vector<int> > res = combinationSum2_helper(candidates, 0, target);
	return res;
}
string Solution::multiply(string num1, string num2)
{
	int len1 = num1.length();
	int len2 = num2.length();
	int len = len1 + len2;
	string res(len, '0');
	int base = 0;
	reverse(num1.begin(), num1.end());
	reverse(num2.begin(), num2.end());
	for (int i = 0; i < len1; i++)
	{
		for (int j = 0; j < len2; j++)
		{
			int t = (num1[i] - '0')*(num2[j] - '0') + res[i + j] - '0';
			res[i + j + 1] += t / 10;
			res[i + j] = t % 10 + '0';
		}
	}
	while (len>0 && res[len - 1] == '0')len--;
	if (!len)return "0";

	res = res.substr(0, len);

	reverse(res.begin(), res.end());
	return res;
}
void Solution::rotate(vector<vector<int> > &matrix) {
	int n = matrix.size();
	for (int i = 0; i < n - 1; i++)
	{
		for (int j = 0; j < n - i - 1; j++)
		{
			swap(matrix[i][j], matrix[n - j - 1][n - i - 1]);
		}
	}
	reverse(matrix.begin(), matrix.end());
}
vector<string> Solution::anagrams(vector<string> &strs)
{
	vector<string> res;
	unordered_map<string, int> dict;
	for (int k = 0; k < strs.size(); k++)
	{
		string temp = strs[k];
		sort(temp.begin(), temp.end());
		auto iter = dict.find(temp);
		if (iter != dict.end())
		{
			res.push_back(strs[k]);
			if (dict[temp] != -1)res.push_back(strs[dict[temp]]);
			dict[temp] = -1;
		}
		else
			dict[temp] = k;
	}
	return res;
}
int Solution::maxSubArray(vector<int>& nums)
{
	if (nums.size() < 1)return 0;
	int res = nums[0];
	int cur = nums[0];
	for (int k = 1; k < nums.size(); k++)
	{
		cur = max(cur + nums[k], nums[k]);
		if (cur>res)
			res = cur;
	}
	return res;
}
vector<int> Solution::spiralOrder(vector<vector<int> > &matrix)
{
	int m = matrix.size();
	if (m < 1) return vector<int>();
	int n = matrix[0].size();
	if (n < 1)return vector<int>();
	vector<int> res;

	int start = 0;
	while (2*start<m&&2*start<n)
	{
		for (int i = start; i < n - start; i++)
			res.push_back(matrix[start][i]);
		if (start + 1 >= m - start)break;
		for (int j = start + 1; j < m - start;j++)
			res.push_back(matrix[j][n-start-1]);
		if (n - start - 2 < start)break;
		for (int i = n - start - 2; i>=start; i--)
			res.push_back(matrix[m - start-1][i]);
		if (m - start - 2 <= start)break;
		for (int j = m - start - 2; j>start; j--)
			res.push_back(matrix[j][start]);
		start++;
	}
	return res;
}
vector<vector<int> > Solution::generateMatrix(int n)
{
	vector<vector<int>> matrix(n,vector<int>(n));
	int start = 0, count = 0;
	while (2 * start<n)
	{
		for (int i = start; i < n - start; i++)
			matrix[start][i]=(++count);
		if (start + 1 >= m - start)break;
		for (int j = start + 1; j < m - start; j++)
			matrix[j][n - start - 1]=(++count);
		if (n - start - 2 < start)break;
		for (int i = n - start - 2; i >= start; i--)
			matrix[m - start - 1][i]=(++count);
		if (m - start - 2 <= start)break;
		for (int j = m - start - 2; j>start; j--)
			matrix[j][start]=(++count);
		start++;
	}
	return matrix;
}

string Solution::getPermutation(int n, int k)
{
	string res(n, '0'),used(n,'1');
	int fact = 1; used[1] = '2';
	for (int i = 2; i < n; i++){
		fact *= i; used[i] = i+'1';
	}k--;
	for (int i = 0;i < n - 1; i++)
	{
		int pos = k / fact;
		res[i] = used[pos];
		used.erase(used.begin() + pos);

		k %= fact;
		fact /= (n - i - 1);
	}
	res[n - 1] = used[0];
	return res;
}
ListNode *Solution::rotateRight(ListNode *head, int k)
{
	if (!head || !k)return head;

	ListNode* p = head;
	int count = 1;
	while (p->next)
	{
		p = p->next;
		count++;
	}
	k %= count;
	p->next = head; // get a circle;
	ListNode* q = head;
	while (k > 0) // go forward k step
	{
		q = q->next;
		k--;
	}
	while (q != p)
	{
		head = head->next;
		q = q->next;
	}
	q = head->next;
	head->next = NULL;
	return q;
}
int Solution::uniquePaths(int m, int n)
{
	for (int k = 0; k < 101; k++)
	{
		Path[0][k] = 1;
		Path[k][0] = 1;
	}
	for (int i = 1; i < m; i++)
	{
		for (int j = 1; j < n; j++)
		{
			Path[i][j] = Path[i - 1][j] + Path[i][j - 1];
		}
	}
	return Path[m - 1][n - 1];
}

int Solution::uniquePathsWithObstacles(vector<vector<int> > &obstacleGrid)
{
	int m = obstacleGrid.size();
	if (m< 1)return 0;
	int n = obstacleGrid[0].size();
	if (n < 1)return 0;
	if (obstacleGrid[0][0] == 1)return 0;
	int val = 1;
	for (int i = 0; i < m; i++)
	{
		if (obstacleGrid[i][0] == 1)val = 0;
		obstacleGrid[i][0] = val;
	}
	val = 1;
	for (int i = 1; i < n; i++)
	{
		if (obstacleGrid[0][i] == 1)val = 0;
		obstacleGrid[0][i] = val;
	}
	for (int i = 1; i < m; i++)
	{
		for (int j = 1; j < n; j++)
		{
			if (obstacleGrid[i][j] == 1)
				obstacleGrid[i][j] = 0;
			else
				obstacleGrid[i][j] = obstacleGrid[i - 1][j] + obstacleGrid[i][j - 1];
		}
	}
	return obstacleGrid[m - 1][n - 1];
}
int Solution::minPathSum(vector<vector<int> > &grid)
{
	int m = grid.size();
	if (m< 1)return 0;
	int n = grid[0].size();
	if (n < 1)return 0;
	int val = 0;
	for (int i = 1; i < m; i++)
	{
		val += grid[i][0];
		grid[i][0] = val;
	}
	val = 0;
	for (int i = 1; i < n; i++)
	{
		val += grid[0][i];
		grid[0][i] = val;
	}
	for (int i = 1; i < m; i++)
	{
		for (int j = 1; j < n; j++)
		{
			grid[i][j] = min(grid[i - 1][j], grid[i][j - 1]) + grid[i][j];
		}
	}
	return grid[m - 1][n - 1];
}
string Solution::simplifyPath(string path)
{
	path.append("/"); // for example /home/foo ---> /home/foo/ (set the guard)
	vector<string> simp_path;
	int pre = 0, pos = 1;
	string temp;
	while (pos < path.length())
	{
		if (path[pos] == '/')
		{
			temp = path.substr(pre + 1, pos - pre - 1);
			if (temp.empty() || temp == ".")
				;//do nothinh
			else if (temp == "..")
			{
				if (!simp_path.empty())
					simp_path.pop_back();
			}
			else
				simp_path.push_back(temp);
			pre = pos;
		}
		pos++;
	}
	string res;
	if (simp_path.empty())return "/";
	for (int k = 0; k < simp_path.size(); k++)
	{
		res.append("/" + simp_path[k]);
	}
	return res;
}
void Solution::setZeroes(vector<vector<int> > &matrix) {
	int i = 0, j = 0;
	int m = matrix.size();
	if (m<1)return;
	int n = matrix[0].size();
	if (n<1)return;
	bool flag = true;
	for (i = 0; i<m; i++)
	{
		for (j = 0; j<n; j++)
		{
			if (matrix[i][j] == 0)
			{
				flag = false;
				break;
			}
		}
		if (!flag)break;
	}
	if (i == m&&j == n)return;
	for (int k = 0; k<n; k++)
	if (matrix[i][k])matrix[i][k] = -1;
	for (int k = 0; k<m; k++)
	if (matrix[k][j])matrix[k][j] = -1;
	for (int s = i + 1; s<m; s++)
	{
		for (int l = 0; l<n; l++)
		{
			if (l == j)continue;
			if (matrix[s][l] == 0)
			{
				matrix[s][j] = 0;
				matrix[i][l] = 0;
			}
		}
	}
	for (int s = i + 1; s<m; s++)
	{
		if (matrix[s][j] == 0)
		{
			for (int l = 0; l<n; l++)
				matrix[s][l] = 0;
		}
	}
	for (int s = 0; s<n; s++)
	{
		if (matrix[i][s] == 0)
		{
			for (int l = 0; l<m; l++)
				matrix[l][s] = 0;
		}
	}
	for (int k = 0; k<n; k++)
		matrix[i][k] = 0;
	for (int k = 0; k<m; k++)
		matrix[k][j] = 0;
}
void Solution::combine_helper(vector<int> source, vector<int> cur, int k, int ind)
{
	if (source.size() - ind + cur.size() < k) return;
	if (cur.size() == k)
	{
		combine_bin.push_back(cur);
		return;
	}
	if (ind >= source.size()) return;
	vector<int> tcur = cur;
	tcur.push_back(source[ind]);

	combine_helper(source, cur, k, ind + 1);
	combine_helper(source, tcur, k, ind + 1);
}
vector<vector<int> > Solution::combine(int n, int k)
{
	vector<int> val(n), cur;
	for (int k = 0; k < n; k++)
	{
		val[k] = k + 1;
	}
	combine_helper(val, cur, k, 0);
	return combine_bin;
}
void Solution::subset_helper(vector<int> source, vector<int> cur, int ind)
{
	if (ind == source.size())
	{
		subset_bin.push_back(cur);
		return;
	}
	if (ind >= source.size()) return;
	vector<int> tcur = cur;
	tcur.push_back(source[ind]);

	subset_helper(source, cur, ind + 1);
	subset_helper(source, tcur, ind + 1);
}
vector<vector<int> > Solution::subsets(vector<int> &S)
{
	sort(S.begin(), S.end());
	vector<int> cur;
	subset_helper(S, cur, 0);
	return subset_bin;
}
bool Solution::isFound(vector<vector<char> > &board, const char* w, int x, int y)
{
	if (x<0 || y<0 || x >= m || y >= n || board[x][y] == '\0' || *w != board[x][y])
		return false;
	if (*(w + 1) == '\0')
		return true;
	char t = board[x][y];
	board[x][y] = '\0';
	if (isFound(board, w + 1, x - 1, y) || isFound(board, w + 1, x + 1, y) || isFound(board, w + 1, x, y - 1) || isFound(board, w + 1, x, y + 1))
		return true;
	board[x][y] = t;
	return false;
}
bool Solution::exist(vector<vector<char> > &board, string word)
{
	m = board.size();
	n = board[0].size();
	for (int x = 0; x < m; x++)
	{
		for (int y = 0; y<n; y++)
		{
			if (isFound(board, word.c_str(), x, y))
				return true;
		}
	}
	return false;
}
int Solution::removeDuplicates(vector<int>&num)
{
	int n = num.size();
	if (n < 3)return n;
	int k = 2,pos =1,index = 2;
	for (; index < n; index++)
		if (num[index] != num[pos - k + 1])
			num[++pos] = num[index];
	num.resize(pos + 1);
}
bool Solution::search(int A[], int n, int target)
{
	if (n<1)return false;
	int hi = n - 1;
	int low = 0;
	int mid = low + hi / 2;
	if (A[mid] == target || A[low] == target || A[hi] == target)return true;
	return search(A + (low + 1), mid - low - 1, target) || search(A + (mid + 1), hi - mid - 1, target);
}

TreeNode *Solution::sortedArrayToBST(vector<int> &num)
{
	TreeNode* root = NULL;
	if (num.size() < 1)return root;
	int low = 0;
	int hi = num.size() - 1;
	int mid = (low + hi) / 2;
	root = new TreeNode(num[mid]);
	vector<int> left(num.begin(), num.begin() + mid);
	vector<int> right(num.begin() + mid, num.end());
	root->left = sortedArrayToBST(left);
	root->right = sortedArrayToBST(right);
	return root;
}
int Solution::maxProfit(vector<int>& prices)
{
	int release = 0;
	int hold = INT_MIN;
	for (int i = 0; i < prices.size(); i++)
	{
		release = max(release, hold + prices[i]);
		hold = max(hold, -prices[i]);
	}
	return release;
}
int Solution::maxProfit2(vector<int>& prices)
{
	if (prices.size()<2) return 0;
	int min = prices[0];
	int profit = 0;
	int i;
	for (i = 1; i < prices.size() - 1; i++)
	{
		if (prices[i]>min)
		{
			if (prices[i + 1] < prices[i])
			{
				profit += prices[i] - min;
				min = prices[i + 1];
			}
		}
		else
			min = prices[i];
	}
	if (prices[i] >= prices[i - 1] && prices[i] > min)profit += prices[i] - min;
	return profit;
}
int Solution::maxProfit3(vector<int>& prices)
{
	int sz = prices.size();
	if (sz < 2)return 0;
	vector<int> leftMin(sz, 0);
	int min = prices[0], max = prices[sz - 1];
	int profit = 0;
	for (int i = 1; i < prices.size(); i++)
	{
		if (prices[i]>min)	profit = std::max(profit, prices[i] - min);
		else min = prices[i];
		leftMin[i] = profit;
	}
	int right_max = 0;
	for (int j = sz - 2; j >= 0; j--)
	{
		if (prices[j] < max) right_max = std::max(right_max, max - prices[j]);
		else max = prices[j];
		if (j>0)
		{
			profit = std::max(profit, leftMin[j - 1] + right_max);
		}
	}
	return std::max(profit, right_max);
}
int Solution::maxProfit4(int k, vector<int>& prices)
{
	if (prices.size() < 2 || k < 1)return 0;
	int len = prices.size();
	int maxProfit = 0;
	if (k >= len / 2)
	{
		for (int i = 1; i < len; i++)
		{
			maxProfit += max(0, prices[i] - prices[i - 1]);
		}
		return maxProfit;
	}
	vector<int> release(k, 0);
	vector<int> holds(k, INT_MIN);
	for (int i = 0; i < prices.size(); i++)
	{
		for (int j = 0; j < k-1; k++)
		{
			release[j] = max(release[j], holds[j] + prices[i]);
			holds[j] = max(holds[j], release[j + 1] - prices[i]);
		}
		release[k - 1] = max(release[k - 1], holds[k - 1] + prices[i]);
		holds[k - 1] = max(holds[k - 1], -prices[i]);
	}
	return release[0];
}
vector<int> Solution::inorderTraversal(TreeNode *root)
{
	vector<int> res;
	if (root == NULL) return res;
	stack<TreeNode*> st;
	TreeNode *p = root;
	while (p || !st.empty())
	{
		while (p)
		{
			st.push(p);
			p = p->left;
		}
		if (!st.empty())
		{
			p = st.top();
			st.pop();
			res.push_back(p->val);
			p = p->right;
		}
	}
	return res;
}
vector<int> binTree::preOrder_non_recursive()
{
	vector<int> res;
	if (!root) return res;
	stack<TreeNode*> st;
	TreeNode *p = root;
	while (p || !st.empty())
	{
		while (p)
		{
			st.push(p);
			res.push_back(p->val);
			p = p->left;
		}
		if (!st.empty())
		{
			p = st.top();
			st.pop();
			p = p->right;
		}
	}
	return res;
	// another very concise solution:
	/*
	vector<int> rs;
	if (!root) return rs;
	stack<TreeNode *> stk;
	stk.push(root);
	while (!stk.empty())
	{
	TreeNode *t = stk.top();
	stk.pop();
	rs.push_back(t->val);

	if (t->right) stk.push(t->right);
	if (t->left) stk.push(t->left);
	}
	//reverse(rs.begin(), rs.end());
	return rs;
	*/
}
vector<int> binTree::inOrder_non_recursive()
{
	vector<int> res;
	if (root == NULL) return res;
	stack<TreeNode*> st;
	TreeNode *p = root;
	while (p || !st.empty())
	{
		while (p)
		{
			st.push(p);
			p = p->left;
		}
		if (!st.empty())
		{
			p = st.top();
			st.pop();
			res.push_back(p->val);
			p = p->right;
		}
	}
	return res;
}
vector<int> binTree::postOrder_non_recursive()
{
	vector<int> res;
	if (root == NULL) return res;
	stack<TreeNode*> st;
	TreeNode *p = root;
	TreeNode *pre = NULL;
	st.push(root);
	while (!st.empty())
	{
		p = st.top();
		if ((p->left == NULL&&p->right == NULL) ||
			(pre!=NULL&&(pre == p->left || pre == p->right)))
		{
			res.push_back(p->val);
			st.pop();
			pre = p;
		}
		else
		{
			if (p->right)
				st.push(p->right);
			if (p->left)
				st.push(p->left);
		}
	}
	return res;
}
vector<int> binTree::preOrder_MorrisTravel()
{
	vector<int> res;
	TreeNode* cur = root;
	while (cur)
	{
		if (!cur->left)
		{
			res.push_back(cur->val);
			cur = cur->right;
		}
		else
		{
			TreeNode* temp = cur->left;
			while (temp->right&&temp->right != cur)temp = temp->right;
			if (!temp->right)
			{
				temp->right = cur;
				res.push_back(cur->val);
				cur = cur->left;
			}
			else
			{
				cur = cur->right;
				temp->right = NULL;
			}
		}
	}
	return res;
}
vector<int> binTree::inOrder_MorrisTravel()
{
	vector<int> res;
	TreeNode* cur = root;
	while (cur)
	{
		if (!cur->left)
		{
			res.push_back(cur->val);
			cur = cur->right;
		}
		else
		{
			TreeNode* temp = cur->left;
			while (temp->right&&temp->right != cur)temp = temp->right;
			if (!temp->right)
			{
				temp->right = cur;
				cur = cur->left;
			}
			else
			{
				res.push_back(cur->val);
				cur = cur->right;
				temp->right = NULL;
			}
		}
	}
	return res;
}
void binTree::reverse(TreeNode* from, TreeNode* to)
{
	if (from == to)return;
	TreeNode* pre = from, *cur = from->right, *temp;
	while (1)
	{
		temp = cur->right;
		cur->right = pre;
		pre = cur;
		cur = temp;
		if (pre == to)
			break;
	}
}
void binTree::reverse_push(TreeNode* from, TreeNode* to, vector<int>&res)
{
	reverse(from, to);
	TreeNode* p = to;
	while (1)
	{
		res.push_back(p->val);
		if (p == from)break;
		p = p->right;
	}
	reverse(to, from);
}
vector<int> binTree::postOrder_MorrisTravel()
{
	vector<int> res;
	TreeNode dump(0); dump.left = root;
	TreeNode* cur = &dump;
	while (cur)
	{
		if (!cur->left)
		{
			cur = cur->right;
		}
		else
		{
			TreeNode* temp = cur->left;
			while (temp->right&&temp->right != cur)temp = temp->right;
			if (!temp->right)
			{
				temp->right = cur;
				cur = cur->left;
			}
			else
			{
				reverse_push(cur->left, temp, res);
				cur = cur->right;
				temp->right = NULL;
			}
		}
	}
	return res;
}
// a very concise solution:
//vector<int> postorderTraversal(TreeNode* root) {
//	vector<int> rs;
//	if (!root) return rs;
//	stack<TreeNode *> stk;
//	stk.push(root);
//	while (!stk.empty())
//	{
//		TreeNode *t = stk.top();
//		stk.pop();
//		rs.push_back(t->val);
//
//		if (t->left) stk.push(t->left);
//		if (t->right) stk.push(t->right);
//	}
//	reverse(rs.begin(), rs.end());
//	return rs;
//}
bool Solution::searchMatrix(vector<vector<int> > &matrix, int target)
{
	int rows = matrix.size();
	if (rows < 1)return false;
	int cols = matrix[0].size();
	if (cols < 1)return false;

	int col = cols - 1;
	int row = 0;
	while (row<rows&&col >= 0)
	{
		if (matrix[row][col] == target)
			return true;
		else if (matrix[row][col] > target)
			col--;
		else
			row++;
	}
	return false;
}
ListNode* Solution::deleteDuplicates(ListNode* head)
{
	if (!head || !head->next)return head;
	ListNode hd(0);
	hd.next = head;
	ListNode* p1, *p2;
	p1 = &hd; p2 = hd.next;
	while (p2)
	{
		if (p2->val == p1->next->val)
			p2 = p2->next;
		else
		{
			if (!p2 || p1->next->next != p2)
			{
				p1->next = p2;
			}
			else
				p1 = p1->next;
		}
	}
	if (p1->next&&p1->next->next != p2)
		p1->next = p2;
	return hd.next;
}
vector<vector<int>> subset_res;
void Solution::dfs_subset(vector<int> &S, int start, vector<int> cur)
{
	subset_res.push_back(cur);
	for (int k = start; k < S.size(); k++)
	{
		cur.push_back(S[k]);
		dfs_subset(S, k + 1, cur);
		cur.pop_back();
		while (k + 1<S.size() && S[k + 1] == S[k])k++;
	}
}
vector<vector<int> > Solution::subsetsWithDup(vector<int> &S)
{
	if (S.size() == 0)return subset_res;
	sort(S.begin(), S.end());
	vector<int> cur;
	dfs_subset(S, 0, cur);
	return subset_res;
}
int Solution::parti(vector<int>& array, int lo, int hi)
{
	if (hi <= lo)return lo;
	int pivot;
	pivot = array[lo];
	while (lo < hi)
	{
		while (lo<hi&&array[hi] >= pivot)hi--;
		swap(array[lo], array[hi]);
		while (lo<hi&&array[lo] <= pivot)lo++;
		swap(array[lo], array[hi]);
	}
	return lo;
}
void Solution::qsort(vector<int>& array, int low, int hi)
{
	int index = parti(array, low, hi);
	if (index>low)
		qsort(array, low, index - 1);
	if (index < hi)
		qsort(array, index + 1, hi);
}
void Solution::print_array(vector<int>& array)
{
	for (int k = 0; k < array.size(); k++)
		cout << array[k] << '\t';

}
void Solution::print_topk(vector<int>& array, int topK)
{
	build_heap(array, 0, array.size());
	for (int k = array.size() - 1; k >= 0 && topK>0; k--)
	{
		cout << array[0] << '\t'; topK--;
		swap(array[0], array[k]);
		build_heap(array, 0, k);
	}
	cout << endl;
}
ListNode* Solution::partition(ListNode* head, int x)
{
	if (!head || !head->next)return head;
	ListNode tr(0), tl(0);
	ListNode *pr = &tr;
	ListNode *pl = &tl;
	ListNode* p = head;
	while (p)
	{
		if (p->val < x)
		{
			pl->next = p;
			pl = pl->next;
			p = p->next;
		}
		else
		{
			pr->next = p;
			pr = pr->next;
			p = p->next;
		}
	}
	pr->next = NULL;
	pl->next = tr.next;
	return tl.next;
}
int Solution::numDecodings(string s)
{
	if (s.length() < 1)return 0;
	int len = s.length();
	int pre_2 = 1;
	int pre_1 = s[len - 1] == '0' ? 0 : 1;
	int cur = 1;
	for (int k = len - 2; k >= 0; k--)
	{
		if (s[k] == '0')
			cur = 0;
		else if ((s[k] - '0') * 10 + (s[k + 1] - '0')>26)
			cur = pre_1;
		else
			cur = pre_1 + pre_2;
		pre_2 = pre_1;
		pre_1 = cur;
	}
	return cur;
}
ListNode* Solution::reverseBetween(ListNode* head, int m, int n)
{
	ListNode *pre, *start, *then;
	ListNode tp(0); tp.next = head;
	pre = &tp;
	for (int k = 0; k < m - 1; k++)
	{
		pre = pre->next;
	}
	start = pre->next;
	then = start->next;
	for (int k = m; k < n; k++)
	{
		start->next = then->next;
		then->next = pre->next;
		pre->next = then;
		then = start->next;
	}
	return tp.next;
}
vector<string> IPs;
void dfs_IP_helper(string&s, string cur, int pos, int count)
{
	/** 
	scanf("%d.%d.%d.%d", &ips[i][0], &ips[i][1], &ips[i][2], &ips[i][3]);
	read a IP adress from the text file
	*/
	if (count == 0)
	{
		int vl = atoi(s.substr(pos, s.length() - pos).c_str());
		if (((vl == 0 && s.length() - pos == 1) ||
			(s.length() - pos<4 && vl>0 && vl < 256 && s[pos] != '0')))
		{
			IPs.push_back(cur);
			cout << cur << endl;
		}
		return;
	}
	int val = 0;
	for (int k = pos; k < s.length() - 1 && val < 256; k++)
	{
		string temp = cur;
		int offset = k + 4 - count;
		if (offset< temp.length())
			temp.insert(temp.begin() + offset, '.');
		val = val * 10 + s[k] - '0';
		if (s[pos] == '0')
		{
			dfs_IP_helper(s, temp, k + 1, count - 1);
			break;
		}
		if (val == 0 || (val<256 && s[pos] != '0'))
			dfs_IP_helper(s, temp, k + 1, count - 1);
	}
}
vector<string> Solution::restoreIpAddresses(string s)
{
	if (s.empty() || s.length()<4)return IPs;
	string cur = s;
	dfs_IP_helper(s, cur, 0, 3);
	return IPs;
}
vector<TreeNode*>  Solution::geneTrees(int start, int end)
{
	vector<TreeNode*> tp;
	if (start>end)
	{
		tp.push_back(NULL);
		return tp;
	}
	if (start == end)
	{
		TreeNode *root = new TreeNode(start);
		tp.push_back(root);
		return tp;
	}
	vector<TreeNode*> left;
	vector<TreeNode*> right;
	for (int k = start; k <= end; k++)
	{
		left = geneTrees(start, k - 1);
		right = geneTrees(k + 1, end);
		for (int i = 0; i < left.size(); i++)
		{
			for (int j = 0; j < right.size(); j++)
			{
				TreeNode *root = new TreeNode(k);
				root->left = left[i];
				root->right = right[j];
				tp.push_back(root);
			}
		}
	}
	return tp;
}
vector<TreeNode *> Solution::generateTrees(int n)
{
	vector<TreeNode *>res = geneTrees(1, n);
	return res;
}
vector<vector<int> > Solution::zigzagLevelOrder(TreeNode *root)
{
	stack<TreeNode*> pre;
	stack<TreeNode*> cur;
	vector<vector<int> > res;
	if (!root)return res;
	pre.push(root);
	while (!pre.empty() || !cur.empty())
	{
		vector<int> temp;
		while (!pre.empty())
		{
			TreeNode* p = pre.top();
			temp.push_back(p->val);
			pre.pop();
			if (p->left)cur.push(p->left);
			if (p->right)cur.push(p->right);
		}
		if (!temp.empty())res.push_back(temp);
		temp.clear();
		while (!cur.empty())
		{
			TreeNode* p = cur.top();
			temp.push_back(p->val);
			cur.pop();
			if (p->right)pre.push(p->right);
			if (p->left)pre.push(p->left);
		}
		if (!temp.empty())res.push_back(temp);
	}
	return res;
}
TreeNode *build_helper(vector<int> &preorder, int pres, int pred,
	vector<int> &inorder, int curs, int curd)
{
	if (pres > pred)return NULL;
	int index;
	for (index = curs; index <= curd; index++)
	{
		if (inorder[index] == preorder[pres])
			break;
	}
	if (inorder[index] != preorder[pres]) return NULL; // invalid
	TreeNode* root = new TreeNode(preorder[pres]);
	if (index>curs)
		root->left = build_helper(preorder, pres + 1, pres + index - curs, inorder, curs, index - 1);
	if (index < curd)
		root->right = build_helper(preorder, pres + index - curs + 1, pred, inorder, index + 1, curd);
	return root;
}
TreeNode *Solution::buildTree(vector<int> &preorder, vector<int> &inorder)
{
	if (preorder.empty() || inorder.empty())return NULL;
	return build_helper(preorder, 0, preorder.size() - 1,
		inorder, 0, inorder.size() - 1);
}
TreeNode *sortedListToBST(ListNode *head)
{
	if (!head)return NULL;
	if (!head->next)return new TreeNode(head->val);

	ListNode* slow = head;
	ListNode* fast = head->next->next;
	while (fast&&fast->next)
	{
		slow = slow->next;
		fast = fast->next->next;
	}
	TreeNode* root = new TreeNode(slow->next->val);
	fast = slow->next->next;
	slow->next = NULL;
	root->left = sortedListToBST(head);
	root->right = sortedListToBST(fast);
}
TreeNode *build_helper2(vector<int> &inorder, int ins, int ind,
	vector<int> &postorder, int posts, int postd)
{
	if (ins > ind) return NULL;
	int index = ins;
	for (index = ins; index <= ind; index++)
	{
		if (inorder[index] == postorder[postd])
			break;
	}
	if (inorder[index] != postorder[postd]) return NULL; // invalid
	TreeNode* root = new TreeNode(inorder[index]);
	if (index > ins)
		root->left = build_helper2(inorder, ins, index - 1, postorder, posts, posts + index - ins - 1);
	if (index < ind)
		root->right = build_helper2(inorder, index + 1, ind, postorder, posts + index - ins, postd - 1);
	return root;
}
TreeNode *Solution::buildTree2(vector<int> &inorder, vector<int> &postorder)
{
	return build_helper2(inorder, 0, inorder.size() - 1, postorder, 0, postorder.size() - 1);
}
vector<vector<int> >path_Sum;
void dfs_pathSum(TreeNode* node, vector<int> cur, int sum)
{
	if (node == NULL)return;
	//if (node->val > sum)return; // if all the number on the tree is bigger than -1 then this line will improve the speed
	cur.push_back(node->val);
	if (sum == node->val&&!node->left&&!node->right)
	{
		path_Sum.push_back(cur);
		return;
	}
	if (node->left)
		dfs_pathSum(node->left, cur, sum - node->val);
	if (node->right)
		dfs_pathSum(node->right, cur, sum - node->val);
}
vector<vector<int> > Solution::pathSum(TreeNode *root, int sum)
{
	vector<int> cur;
	dfs_pathSum(root, cur, sum);
	return path_Sum;
}
void Solution::flatten(TreeNode *root)
{
	if (!root)return;

	TreeNode* left = root->left;
	TreeNode* right = root->right;

	flatten(left);
	flatten(right);

	root->left = NULL;
	if (!left)return;
	root->right = left;
	TreeNode* p = root;
	while (p->right)p = p->right;
	p->right = right;
}
void Solution::connect(TreeLinkNode *root)
{
	while (root)
	{
		if (!root->left&&!root->right)return;
		TreeLinkNode *p = root;
		while (p)
		{
			p->left->next = p->right;
			if (p->next)p->right->next = p->next->left;
			p = p->next;
		}
		root = root->left;
	}
}
struct pos
{
	int x;
	int y;
};
void Solution::solve(vector<vector<char>> &board)
{
	int rows = board.size();
	if (rows <= 2)return;
	int cols = board[0].size();
	if (cols <= 2)return;
	int x_d[] = { 0, 1, 0, -1 };
	int y_d[] = { 1, 0, -1, 0 };
	for (int i = 1; i < rows - 1; i++)
	{
		for (int j = 1; j < cols - 1; j++)
		{
			if (board[i][j] == 'O')
			{
				pos pt = { j, i };
				board[i][j] = 'X';

				vector<pos> qt;
				int count = 0;
				qt.push_back(pt);

				while (count<qt.size())
				{
					pos ttp = qt[count];
					count++;

					for (int k = 0; k < 4; k++)
					{
						int x = x_d[k] + ttp.x;
						int y = y_d[k] + ttp.y;
						if (((x <= 0 || x == rows - 1 || y <= 0 || y == cols - 1) && board[y][x] == 'O') ||
							board[y][x] == '+')
						{
							for (int k = 0; k < qt.size(); k++)
							{
								pt = qt[k];
								board[pt.y][pt.x] = '+';
							}
							count = qt.size();
							break;
						}
						else if (board[y][x] == 'O')
						{
							pt = { x, y };
							qt.push_back(pt);
							board[y][x] = 'X';
						}
					}
				}
			}
		}
	}
	for (int i = 1; i < rows - 1; i++)
	{
		for (int j = 1; j < cols - 1; j++)
		{
			if (board[i][j] == '+')
				board[i][j] = 'O';
		}
	}
}
bool ispalindrome(string&s, int st, int end)
{
	int mid = (st + end) / 2;
	for (int k = st; k < mid + 1; k++)
	{
		if (s[k] != s[end + st - k])
			return false;
	}
	return true;
}
vector<vector<string>> partition_helper(string s, int st, int end)
{
	vector < vector<string> > res;
	if (st == end)
	{
		res.push_back(vector<string>(1, s.substr(st, 1)));
		return res;
	}
	if (ispalindrome(s, st, end))
		res.push_back(vector<string>(1, s.substr(st, end - st + 1)));
	for (int k = st; k <end; k++)
	{
		if (k == 2 && st == 1)
		{
			printf("here!\n");
		}
		if (ispalindrome(s, st, k))
		{
			vector < vector<string> > sub = partition_helper(s, k + 1, end);
			for (int j = 0; j < sub.size(); j++)
			{
				sub[j].insert(sub[j].begin(), s.substr(st, k - st + 1));
				res.push_back(sub[j]);
			}
		}
	}
	return res;
}
vector<vector<string>> Solution::partition(string s)
{
	int len = s.length() - 1;
	vector<vector<string>> res;
	if (len < 0)return res;
	res = partition_helper(s, 0, len);
	return res;
}
void dfs_grid(vector<vector<char>> &grid, int rows, int cols, int i, int j)
{
	if (grid[i][j] == '1')
	{
		grid[i][j] = '0';
		if (j + 1 < cols)dfs_grid(grid, rows, cols, i, j + 1);
		if (j - 1 >= 0)dfs_grid(grid, rows, cols, i, j - 1);
		if (i + 1 < rows)dfs_grid(grid, rows, cols, i + 1, j);
		if (i - 1 >= 0)dfs_grid(grid, rows, cols, i - 1, j);
	}
}
int Solution::numIslands(vector<vector<char>> &grid)
{
	int rows = grid.size();
	if (rows < 1)return 0;
	int cols = grid[0].size();
	if (cols < 1)return 0;
	int count = 0;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			dfs_grid(grid, rows, cols, i, j);
			count++;
		}
	}
	return count;
}
int Solution::rangeBitwiseAnd(int m, int n)
{
	int count = 0;
	while (n>m)
	{
		n = n&(n - 1);
		count++;
	}
	return n;
}
int Solution::countPrimes(int n)
{
	vector<int> res;
	if (n < 2) return 0;
	vector<bool> sif(n, false);
	int count = 1; sif[2] = true;
	int upper = sqrt(n);
	for (int k = 3; k < n; k += 2)
	{
		if (!sif[k])
		{
			count++;
			if (k>upper)continue;
			for (int j = k*k; j < n; j++)
				sif[j] = true;
		}
	}
	return res.size();
}
int find_kth_of2sortedArray(vector<int>& nums1, int st1, int end1,
	vector<int>& nums2, int st2, int end2, int k)
{
	int m = end1 - st1;
	int n = end2 - st2;
	if (m > n)
	{
		return find_kth_of2sortedArray(nums2, st2, end2,
			nums1, st1, end1, k);
	}
	if (m == 0)return nums2[k + st2 - 1];
	if (k == 1)return nums1[st1] < nums2[st2] ? nums1[st1] : nums2[st2];
	int pa = k / 2 < m ? k / 2 : m;
	int pb = k - pa;
	if (nums1[pa + st1 - 1] == nums2[pb + st2 - 1])return nums1[pa + st1 - 1];
	if (nums1[pa + st1 - 1] < nums2[pb + st2 - 1])
		return find_kth_of2sortedArray(nums1, st1 + pa, end1, nums2, st2, end2, k - pa);
	if (nums1[pa + st1 - 1] > nums2[pb + st2 - 1])
		return find_kth_of2sortedArray(nums1, st1, end1, nums2, st2 + pb, end2, k - pb);
}
double Solution::findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2)
{
	int m = nums1.size();
	int n = nums2.size();

	if (m == 0 && n == 0)return 0.0; // invalid

	int len = m + n;
	if (len % 2 == 1)
		return find_kth_of2sortedArray(nums1, 0, m, nums2, 0, n, len / 2 + 1);
	else
		return double(find_kth_of2sortedArray(nums1, 0, m, nums2, 0, n, len / 2) + find_kth_of2sortedArray(nums1, 0, m, nums2, 0, n, len / 2 + 1)) / 2;
}
bool isMatch_rec(const char* s, const char* p)
{
	if (!*p)return !*s;
	if (!*s)return (*(p + 1) == '*') && isMatch_rec(s, p + 2);
	if (*(p + 1) == '*') return isMatch_rec(s, p + 2) || (*s == *p || *p == '.') && isMatch_rec(s + 1, p);
	else
		return (*s == *p || *p == '.') && isMatch_rec(s + 1, p + 1);
}
bool Solution::isMatch(string s, string p)
{
	int m = s.length();
	int n = p.length();
	vector<vector<bool> > status(m + 1, vector<bool>(n + 1, false));
	status[0][0] = true;
	for (int j = 2; j <= n; j += 2)
	{
		if (status[0][j - 2] && p[j - 1] == '*')
			status[0][j] = true;
		else
			break;
	}
	for (int i = 1; i <= m; i++)
	{
		for (int j = 1; j <= n; j++)
		{
			if (p[j - 1] == '*')
				status[i][j] = (s[i - 1] == p[j - 2] || p[j - 2] == '.') && status[i - 1][j] || status[i][j - 2];
			else
				status[i][j] = (s[i - 1] == p[j - 1] || p[j - 1] == '.') && status[i - 1][j - 1];
		}
	}
	return status[m][n];
}
static bool comp_listnode(const ListNode* n1, const ListNode* n2)
{
	return n1->val >= n2->val;
}
ListNode* Solution::mergeKLists(vector<ListNode*>& lists)
{
	vector<ListNode*> hp_lists;
	for (int k = 0; k < lists.size(); k++)
	{
		if (lists[k])hp_lists.push_back(lists[k]);
	}
	if (hp_lists.empty())return NULL;
	make_heap(hp_lists.begin(), hp_lists.end(), comp_listnode);
	ListNode* head = hp_lists[0];
	ListNode* p = head;
	while (!hp_lists.empty())
	{
		pop_heap(hp_lists.begin(), hp_lists.end(), comp_listnode);
		hp_lists.pop_back();
		if (p->next)
		{
			hp_lists.push_back(p->next);
			push_heap(hp_lists.begin(), hp_lists.end(), comp_listnode);
		}
		if (hp_lists.empty()) return head;
		push_heap(hp_lists.begin(), hp_lists.end(), comp_listnode);
		p->next = hp_lists[0];
		p = p->next;
	}
	return head;
}
void Solution::build_heap(vector<int>& array, int start, int end)
{

	/*for (int k = (start+end) / 2-1; k >= start; k--)
	{
	int i = 2 * (k+1);
	if (i>=end||array[i - 1] > array[i])i--;
	if (array[i] > array[k])swap(array[i], array[k]);
	}*/
	for (int i = (start + end) / 2 - 1; i >= 0; i--)
		adjust_heap(array, i, end);
}
void Solution::adjust_heap(vector<int>& array, int start, int end)
{
	int k = start;
	int temp = array[start];
	while (k < end)
	{
		int j = 2 * k + 1;
		if (j< end - 1 && array[j + 1]>array[j])j++;
		if (j < end && array[j] > temp)
		{
			array[k] = array[j];
			k = j;
		}
		else
			break;
	}
	array[k] = temp;
}
void Solution::heap_sort(vector<int>& array)
{
	build_heap(array, 0, array.size());
	for (int k = array.size() - 1; k >= 0; k--)
	{
		swap(array[0], array[k]);
		build_heap(array, 0, k);
	}
}
ListNode* reverseKGroup(ListNode* head, int k)
{
	int i;
	if (k < 2 || !head || !head->next)return head;

	ListNode* p = head;
	for (i = 0; i < k - 1 && p; i++)p = p->next;
	if (i != k - 1 || !p)return head;
	ListNode* q = head, *pre = NULL, *temp;
	for (i = 0; i < k; i++)
	{
		temp = q;
		q = q->next;

		temp->next = pre;
		pre = temp;
	}
	head->next = reverseKGroup(q, k);
	return p;
}
bool isMapEqual(map<string, int>& map1, map<string, int>& map2)
{
	if (map1.size() != map2.size())return false;
	for (map<string, int>::iterator iter = map1.begin(); iter != map1.end(); iter++)
	{
		if (iter->second != map2[iter->first])return false;
	}
	return true;
}
vector<int> Solution::findSubstring(string s, vector<string>& words) {
	map<string, int> target;
	map<string, int> source;
	vector<int> res;
	if (words.empty())return res;

	for (int k = 0; k < words.size(); k++)
		target[words[k]]++;

	int step = words[0].size();
	int start = 0, end = words.size()*step;
	if (s.length()<end)return res;
	string temp;
	while (end <= s.length())
	{
		int posi;
		for (posi = start; posi <end; posi += step) //init
		{
			temp = s.substr(posi, step);
			if (target.find(temp) != target.end())
				source[temp]++;
			else
				break;
		}
		if (posi == end&&isMapEqual(target, source))res.push_back(start);
		source.clear();
		start++;
		end++;
	}
	return res;
}
int Solution::longestValidParentheses(string s)
{
	if (s.length() < 2)return 0;
	vector<int> lps(s.length(), 0);
	if (s[0] == '('&&s[1] == ')')lps[1] = 2;
	for (int i = 2; i < s.length(); i++)
	{
		if (s[i] == '(')lps[i] = 0;
		else
		{
			/*if (i == 5)
			printf("here\n");
			int temp = lps[i - 1];*/
			if (i - lps[i - 1]>0 && s[i - lps[i - 1] - 1] == '(')
			{
				lps[i] = lps[i - 1] + 2;
				if (i - lps[i - 1] - 1 > 0)lps[i] += lps[i - lps[i - 1] - 2];
			}

		}
	}
	return *max_element(lps.begin(), lps.end());
}
int Solution::search(vector<int>& nums, int target)
{
	int lo = 0;
	int hi = nums.size() - 1;
	if (lo >= hi)return lo;

	while (lo < hi)
	{
		int mid = (lo + hi) / 2;
		if (nums[mid] == target)return mid;
		if (nums[mid] >= nums[lo])
		{
			if (target < nums[lo] || target>nums[mid])
				lo = mid + 1;
			else
				hi = mid - 1;
		}
		else
		{
			if (target < nums[mid] || target>nums[hi])
				hi = mid - 1;
			else
				lo = mid + 1;
		}
	}
	return lo;
}
//---------------------------My Own sudokusolver and check (lees memory)---------------------------
bool valid(short& vld, int pos)
{
	if (pos < 0 || pos >= 9)return true;
	if (vld & 1 << pos)return false;
	vld |= 1 << pos;
	return true;
}
bool Solution::isValidSudoku(vector<vector<char> >& board)
{
	for (int i = 0; i < 9; i++)
	{
		short valid_r = 0;
		short valid_c = 0;
		short valid_b = 0;

		for (int j = 0; j < 9; j++)
		{
			if (!valid(valid_r, board[i][j] - '1') ||
				!valid(valid_c, board[j][i] - '1') ||
				!valid(valid_b, board[(i / 3) * 3 + j / 3][3 * (i % 3) + j % 3] - '1'))
				return false;
		}
	}
	return true;
}
bool checkSudoku(vector<vector<char>>& board, int i, int j)
{
	short valid_c = 0;
	short valid_r = 0;
	short valid_b = 0;
	for (int k = 0; k < 9; k++)
	{
		if (!valid(valid_r, board[i][k] - '1') || // i rows
			!valid(valid_c, board[k][j] - '1') || // j cols
			!valid(valid_b, board[3 * (i / 3) + k / 3][3 * (j / 3) + k % 3] - '1')) // j cols
			return false;
	}
	return true;
}
bool solveHelper(vector<vector<char> >& board, int i, int j)
{
	if (i == 9)
		return true;
	if (board[i][j] != '.'){
		if (j<8)
			return solveHelper(board, i, j + 1);
		else
			return solveHelper(board, i + 1, 0);
	}
	else for (int k = 0; k<9; k++){
		board[i][j] = (char)(k + '1');
		if (checkSudoku(board, i, j)){
			if (j<8 && solveHelper(board, i, j + 1))
				return true;
			else if (j == 8 && solveHelper(board, i + 1, 0))
				return true;
			else{
				board[i][j] = '.';
			}
		}
		else
			board[i][j] = '.';
	}
	return false;
}
void Solution::solveSudoku(vector<vector<char> >& board)
{
	solveHelper(board, 0, 0);
}
string Solution::countAndSay(int n)
{
	string buf = "1";
	string res = "1";
	for (int k = 2; k <= n; k++)
	{
		res = "";
		int len = buf.length();
		char  pre = buf[0];
		int count = 1;
		for (int i = 1; i <= len; i++)
		{
			if (pre == buf[i])count++;
			else
			{
				char str[] = { count + '0', pre, '\0' };
				res.append(string(str));
				pre = buf[i]; count = 1;
			}
		}
		buf = res;
	}
	return res;
}
int Solution::firstMissingPositive(vector<int>& nums)
{
	for (int k = 0; k < nums.size(); k++)
	while (nums[k]>0 && nums[k] <= nums.size() && nums[k] != nums[nums[k] - 1])
		swap(nums[k], nums[nums[k] - 1]);
	for (int k = 0; k < nums.size(); k++)
	if (nums[k] != k + 1)return k + 1;
	return nums.size() + 1;
}
int Solution::sunday(const char* s, int ls, const char* p, int lp)
{
	int next[256];
	memset(next, -1, sizeof(next));
	for (int k = 0; k < lp; k++)
		next[p[k]] = k;
	int pos = 0;
	while (pos < ls - lp + 1) // last one is '\0'
	{
		int  i = pos;
		int k = 0;
		for (; k < lp; k++)
		{
			if (s[i] == p[k] || p[k] == '?')i++;
			else
			{
				pos += lp - next[s[pos + lp]];
				break;
			}
		}
		if (k == lp)return pos;
	}
	return -1;
}
bool Solution::MatchHelper2(const char* s, const char* p)
{
	/*
	int m = strlen(s);
	int n = strlen(p);
	vector<vector<bool> > match(m+1, vector<bool>(n+1, 0));
	match[0][0] = true;
	for (int k = 1; k <= n; k++)
	{
	if (p[k - 1] == '*')match[0][k] = true;
	else break;
	}
	for (int i = 1; i <= m; i++)
	{
	for (int j = 1; j <= n; j++)
	{
	if (p[j] == '*')
	match[i][j] = match[i][j - 1] || match[i - 1][j];
	else
	match[i][j] = match[i][j - 1] &&
	(s[i-1] == p[j-1] || p[j - 1] == '?');
	}
	}
	return match[m][n];
	*/
	// Start typing your C/C++ solution below  
	// DO NOT write int main() function  
	bool star = false;
	const char *str, *ptr;
	for (str = s, ptr = p; *str != '\0'; str++, ptr++)
	{
		switch (*ptr)
		{
		case '?':
			break;
		case '*':
			star = true;
			s = str, p = ptr;
			while (*p == '*')
				p++;
			if (*p == '\0') // 如果'*'之后，pat是空的，直接返回true  
				return true;
			str = s - 1;
			ptr = p - 1;
			break;
		default:
			if (*str != *ptr)
			{
				// 如果前面没有'*'，则匹配不成功  
				if (!star)
					return false;
				s++;
				str = s - 1;
				ptr = p - 1;
			}
		}
	}
	while (*ptr == '*')
		ptr++;
	return (*ptr == '\0');
}
bool Solution::isMatch2(string s, string p)
{
	/*int m = s.length();
	int n = p.length();
	vector<bool>match(n + 1, 0);
	match[0] = true;
	for (int k = 1; k <= n; k++)
	{
	if (p[k - 1] == '*')match[k] = true;
	else break;
	}
	for (int i = 1; i <= m; i++)
	{
	vector<bool>cur(n + 1, 0);
	for (int j = 1; j <= n; j++)
	{
	if (p[j] == '*')
	cur[j] = cur[j - 1] || match[j];
	else
	cur[j] = match[j - 1] &&
	(s[i - 1] == p[j - 1] || p[j - 1] == '?');
	}
	match = cur;
	}
	return match[n];*/
	return MatchHelper2(s.c_str(), p.c_str());
}
/** my own trap solution time limit excced--------------------------
int lo = 0, hi = height.size() - 1;
int hei = 1, res = 0;
while (lo<hi)
{
while (height[lo] < hei)lo++;
while (height[hi] < hei)hi--;
for(int k = lo + 1; k < hi; k++)
if (height[k] < hei)res++;
hei++;
}
return res;
--------------------------------------------------------------------*/
int Solution::trap(vector<int>& height)
{
	/*if (height.size() < 3)return 0;
	vector<int> temp(height.size(), 0);
	int max = height[0];
	for (int k = 1; k < height.size(); k++)
	{
	if (height[k]>max)
	{
	temp[k] = height[k];
	max = height[k];
	}
	else
	temp[k] = max;
	}
	int res = 0;
	max = height[height.size() - 1];
	for (int k = height.size() - 2; k >= 0; k--)
	{
	if (height[k] > max)
	max = height[k];
	else
	{
	int t = (std::min(max, temp[k]) - height[k]);
	res += std::max(0, t);
	}
	}
	return res;*/
	// more faster vesion---------------
	if (height.size() < 3)return 0;
	int lo = 0, hi = height.size() - 1;
	int res = 0, plank = 0;
	while (lo < hi)
	{
		int t = std::min(height[lo], height[hi]);
		plank = std::max(t, plank);
		res += height[lo] >= height[hi] ? (plank - height[hi--]) : (plank - height[lo++]);
	}
	return res;
}
//--------------------------------Java Sudoku solver(much faster more memory needed)-----------------------------------------------
//bool row[9][9];
//bool col[9][9];
//bool blo[9][9];
//bool solveHelper(vector<vector<char> >& board, int i, int j)
//{
//	if (i == 9)
//		return true;
//	if (board[i][j] != '.'){
//		if (j<8)
//			return solveHelper(board, i, j + 1);
//		else
//			return solveHelper(board, i + 1, 0);
//	}
//	else for (int k = 0; k<9; k++){
//		if (row[i][k] == false && col[j][k] == false && blo[3 * (i / 3) + j / 3][k] == false){
//			board[i][j] = (char)(k + '0' + 1);
//			row[i][k] = true;
//			col[j][k] = true;
//			blo[3 * (i / 3) + j / 3][k] = true;
//			if (j<8 && solveHelper(board, i, j + 1))
//				return true;
//			else if (j == 8 && solveHelper(board, i + 1, 0))
//				return true;
//			else{
//				board[i][j] = '.';
//				row[i][k] = false;
//				col[j][k] = false;
//				blo[3 * (i / 3) + j / 3][k] = false;
//			}
//		}
//	}
//	return false;
//}
//void Solution::solveSudoku(vector<vector<char> >& board)
//{
//	memset(row, false, sizeof(bool)* 81);
//	memset(row, false, sizeof(bool)* 81);
//	memset(row, false, sizeof(bool)* 81);
//
//	for (int i = 0; i<9; i++)
//	for (int j = 0; j<9; j++){
//		int temp = board[i][j] - '1';
//		if (board[i][j] != '.'){
//			row[i][temp] = true;
//			col[j][temp] = true;
//			blo[3 * (i / 3) + j / 3][temp] = true;
//		}
//	}
//	solveHelper(board,0,0);
//}
int Solution::jump(vector<int>&nums)
{
	int n = nums.size();
	int i = 0, j = 1, cnt = 0, mx = 0;
	if (n <= 1) return 0;
	while (i < n - 1 && i + nums[i] < n - 1) {
		for (; j <= i + nums[i] && mx<n - 1; j++) { mx = (mx + nums[mx] < j + nums[j]) ? j : mx; }
		i = mx; cnt++;
	}
	return ++cnt; /* One more step to last index. */
}

void dfs_permute(vector<int> nums, vector<vector<int> >& permutes, int pos) // do not use referrence at here
{
	if (pos == nums.size() - 1){
		permutes.push_back(nums);
		//cout << permutes.size() << endl;
		return;
	}
	for (int k = pos; k < nums.size(); k++)
	{
		if (pos == k || nums[pos] != nums[k])
		{
			swap(nums[pos], nums[k]);
			dfs_permute(nums, permutes, pos + 1);
		}
	}
}
vector<vector<int> > Solution::permuteUnique(vector<int>& nums)
{
	sort(nums.begin(), nums.end());
	vector<vector<int> > permutes;
	dfs_permute(nums, permutes, 0);
	return permutes;
}
//bool check_quees(vector<int>& quees,int start,int end)
//{
//	for (int i = start; i < end; i++)
//	{
//		for (int j = i+1; j < end; j++)
//		{
//			if (abs(i - j) == abs(quees[i] - quees[j]))
//				return false;
//		}
//	}
//	return true;
//}
//void dfs_nqueens(vector<int> qs, int pos, int& total)
//{
//	if (pos == qs.size() - 1)
//	{
//		if (check_quees(qs,0,pos+1))
//			total++;
//		return;
//	}
//	for (int k = pos; k < qs.size(); k++)
//	{
//		swap(qs[pos], qs[k]);
//		if(check_quees(qs,0, pos));
//			dfs_nqueens(qs, pos + 1, total);
//	}
//}
//int Solution::totalNQueens(int n)
//{
//	vector<int> qs(n, 0);
//	for (int k = 0; k < n; k++)
//		qs[k] = k;
//	int total = 0;
//	dfs_nqueens(qs, 0, total);
//	return total;
//}
bool valid(vector<string>& board, int n, int row, int col)
{
	for (int i = 0; i < row; i++)
	{
		if (board[i][col] == 'Q')
			return false;
	}
	for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--)
	{
		if (board[i][j] == 'Q')
			return false;
	}
	for (int i = row - 1, j = col + 1; i >= 0 && j <n; i--, j++)
	{
		if (board[i][j] == 'Q')
			return false;
	}
	return true;
}
void dfs_nqueens(vector<string>& qs, int n, int rows, vector<vector<string> >& solus)
{
	if (rows == n){
		solus.push_back(qs);
		return;
	}
	for (int k = 0; k < n; k++)	{
		if (valid(qs, n, rows, k)){
			qs[rows][k] = 'Q';
			dfs_nqueens(qs, n, rows + 1, solus);
			qs[rows][k] = '.';
		}
	}
}
vector<vector<string> > Solution::solveNQueens(int n)
{
	vector<string> board(n, string(n, '.'));
	vector <vector<string> > solus;
	dfs_nqueens(board, n, 0, solus);
	return solus;
}

void count_nqueens(vector<string>& qs, int n, int rows, int& total)
{
	if (rows == n)
	{
		total++;
		return;
	}
	for (int k = 0; k < n; k++)
	{
		if (valid(qs, n, rows, k))
		{
			qs[rows][k] = 'Q';
			count_nqueens(qs, n, rows + 1, total);
			qs[rows][k] = '.';
		}
	}
}
int Solution::totalNQueens(int n)
{
	vector<string> board(n, string(n, '.'));
	int total = 0;
	count_nqueens(board, n, 0, total);
	return total;
}
static bool comp_intv(const Interval& in1, const Interval& in2)
{
	return in1.start < in2.start;
}
vector<Interval> Solution::merge(vector<Interval>& intervals)
{
	sort(intervals.begin(), intervals.end(), comp_intv);
	vector<Interval> invs;
	if (intervals.size() < 1)return invs;
	Interval itv = intervals[0];
	for (int k = 1; k < intervals.size(); k++)
	{
		if (intervals[k].start < itv.end)
			itv.end = max(itv.end, intervals[k].end);
		else
		{
			invs.push_back(itv);
			itv = intervals[k];
		}
	}
	return invs;
}
#define INTERSECT(i1,i2) max(i1.start, i2.start) <= min(i1.end, i2.end)
vector<Interval> Solution::insert(vector<Interval>& intervals, Interval newInterval)
{
	/*if (intervals.size() < 1 || (intervals.end() - 1)->end<newInterval.start)
	{
		intervals.push_back(newInterval);
		return intervals;
	}
	if (intervals.begin()->start>newInterval.end)
	{
		intervals.insert(intervals.begin(), newInterval);
		return intervals;
	}
	if (intervals.size() == 1)
	{
		intervals[0].end = max(intervals[0].end, newInterval.end);
		intervals[0].start = min(intervals[0].start, newInterval.start);
		return intervals;
	}
	int low = 0, hi = intervals.size() - 1;
	while (low < hi)
	{
		int mid = (low + hi) / 2;
		if (intervals[mid].end >= newInterval.start)
			hi = mid - 1;
		else
			low = mid + 1;
	}
	int l2 = 0, h2 = intervals.size() - 1;
	while (l2 < h2)
	{
		int md = (l2 + h2) / 2;
		if (intervals[md].start <= newInterval.end)
			l2 = md + 1;
		else
			h2 = md - 1;
	}
	while (intervals[low].end < newInterval.start)low++;
	while (intervals[l2].start>newInterval.end)l2--;
	if (low <= l2)
	{
		newInterval.start = min(newInterval.start, intervals[low].start);
		newInterval.end = max(newInterval.end, intervals[l2].end);
		intervals.erase(intervals.begin() + low, intervals.begin() + l2 + 1);
	}
	intervals.insert(intervals.begin() + low, newInterval);
	return intervals;*/
	vector<Interval> res;
	for (int i = 0; i < intervals.size(); i++){
		if (INTERSECT(newInterval, intervals[i])){
			newInterval.start = min(newInterval.start, intervals[i].start);
			newInterval.end = max(newInterval.end, intervals[i].end);
		}
		else
			res.push_back(intervals[i]);
	}
	auto it = res.begin();
	for (; it != res.end(); it++)
	if (it->start > newInterval.start)
		break;
	res.insert(it, newInterval);
	return res;
}
bool Solution::isNumber(string s) {
	if (s.empty())return false;
	int st = 0, end = s.length() - 1;
	while (s[st] && isspace(s[st]))st++;
	while (s[end] && isspace(s[end]))end--;
	if (end<st)return false;
	s = s.substr(st, end - st + 1);
	bool has_num = false;
	int cur = 0;
	if (s[0] == '+' || s[0] == '-')cur++;
	if (!s[cur])return false;
	while (s[cur] && s[cur] >= '0'&&s[cur] <= '9')
	{
		has_num = true;
		cur++;
	}
	if (s[cur] == '\0')return true;
	if (s[cur] == '.')
	{
		cur++;// skip the '.'
		if (!has_num&&!s[cur])return false;
		string tp2;
		while (s[cur] && s[cur] >= '0'&&s[cur] <= '9')
		{
			has_num = true;
			cur++;
		}
		if (s[cur] == '\0')return true;
		else if (s[cur] != 'e' && s[cur] != 'E') return false;
		else
		{
			if (!has_num)return false;
			cur++;
			if (s[cur] == '+' || s[cur] == '-')cur++;
			if (!s[cur])return false;
			while (s[cur] && s[cur] >= '0'&&s[cur] <= '9')cur++;
			if (s[cur] == '\0')return true;
			else
				return false;
		}
	}
	else if (s[cur] == 'e' || s[cur] == 'E')
	{
		if (!has_num)return false;
		cur++;
		if (s[cur] == '+' || s[cur] == '-')cur++;
		if (s[cur] == '\0')return false;
		while (s[cur] && s[cur] >= '0'&&s[cur] <= '9')cur++;
		if (s[cur] == '\0')return true;
		else
			return false;
	}
	else
		return false;
}
vector<string> Solution::fullJustify(vector<string>& words, int maxWidth)
{
	int len = 0, count = 0, index = 0;
	vector<string> wres;
	for (int k = 0; k<words.size(); k++)
	{
		if (len + words[k].length() <= maxWidth)
		{
			count++;//count words
			len += words[k].length() + 1;// one space
		}
		else
		{
			string temp = "";
			if (count == 0)break;
			else if (count == 1)
			{
				temp.append(words[index]).append(string(maxWidth - words[index].length(), ' '));
				index++;
				wres.push_back(temp);
			}
			else
			{
				int res = (maxWidth - len + count) % (count - 1);
				int step = (maxWidth - len + count) / (count - 1);
				for (int j = 0; j<res; j++)
				{
					temp.append(words[index++]).append(string(step + 1, ' '));
				}
				for (int j = res; j<count - 1; j++)
				{
					temp.append(words[index++]).append(string(step, ' '));
				}
				temp.append(words[index++]);
				wres.push_back(temp);
			}
			len = 0;
			count = 0;
			k--;
		}
	}
	if (count > 0)
	{
		string temp = "";
		for (int i = 0; i < count - 1; i++)
			temp.append(words[index++]).append(" ");
		temp.append(words[index++]);
		temp.append(string(maxWidth - temp.length(), ' '));
		wres.push_back(temp);
	}
	return wres;
}
void dfs_findWords(vector<vector<char> >&board, int m, int n, int i, int j,
	string str, preTree& pTree, vector<string>&wres)
{
	if (board[i][j] == '#')return;
	str = str + board[i][j];
	//if (str == "oath")
	//cout << str << endl;
	if (pTree.search(str))
		wres.push_back(str);
	if (!pTree.startwith(str))return;

	board[i][j] = '#';
	if (j < n - 1)
		dfs_findWords(board, m, n, i, j + 1, str, pTree, wres);
	if (j > 0)
		dfs_findWords(board, m, n, i, j - 1, str, pTree, wres);
	if (i < m - 1)
		dfs_findWords(board, m, n, i + 1, j, str, pTree, wres);
	if (i > 0)
		dfs_findWords(board, m, n, i - 1, j, str, pTree, wres);
	board[i][j] = *(str.end() - 1);
}
vector<string> Solution::findWords(vector<vector<char>>& board, vector<string>& words)
{
	vector<string> wres;
	preTree pTree;
	for (int k = 0; k < words.size(); k++)
		pTree.insert(words[k]);
	int m = board.size();
	if (m < 1)return wres;
	int n = board[0].size();
	if (n < 1)return wres;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			dfs_findWords(board, m, n, i, j, "", pTree, wres);
		}
	}
	return wres;
}
void Solution::pair2graph(vector<pair<int, int>>& prerequisites, Graph& graph)
{
	for (int k = 0; k < prerequisites.size(); k++)
	{
		Graph::Gnode* p = new Graph::Gnode(prerequisites[k].first);
		p->next = graph.G[prerequisites[k].second].next;
		graph.G[prerequisites[k].second].next = p;
		graph.degree[prerequisites[k].first]++;
	}
}
vector<int> Solution::findOrder(int numCourses, vector<pair<int, int>>& prerequisites)
{
	Graph graph(numCourses);
	pair2graph(prerequisites, graph);
	stack<int> st;
	for (int k = 0; k < numCourses; k++)
	{
		if (!graph.degree[k])
			st.push(k);
	}
	vector<int> res;
	while (!st.empty())
	{
		int tp = st.top();
		res.push_back(tp);
		st.pop();
		Graph::Gnode *q = graph.G[tp].next;
		while (q)
		{
			graph.degree[q->val]--;
			if (!graph.degree[q->val])
				st.push(q->val);
			q = q->next;
		}
	}
	if (res.size() != numCourses)
		res.clear();
	return res;
}
bool Solution::canFinish(int numCourses, vector<pair<int, int>>& prerequisites)
{
	Graph graph(numCourses);
	int ncount = 0;
	pair2graph(prerequisites, graph);
	stack<int> st;
	for (int k = 0; k < numCourses; k++)
		if (!graph.degree[k])
			st.push(k);
	while (!st.empty())
	{
		int tp = st.top();
		st.pop(); ncount++;
		Graph::Gnode *q = graph.G[tp].next;
		while (q)
		{
			graph.degree[q->val]--;
			if (!graph.degree[q->val])
				st.push(q->val);
			q = q->next;
		}
	}
	return ncount == numCourses;
}
int minL_helper(int s, vector<int>& nums, int lo, int hi)
{
	if (lo > hi)return 0;
	if (lo == hi)
	{
		if (nums[lo] < s)return 0;
		else return 1;
	}
	int mid = (lo + hi) / 2;
	int l1 = minL_helper(s, nums, lo, mid);
	int l2 = minL_helper(s, nums, mid + 1, hi);
	int len = 0;
	int i1 = mid, i2 = mid + 1;
	int sum = nums[mid] + nums[mid + 1];
	while (i1 >= lo || i2 <= hi)
	{
		if (sum >= s)
		{
			len = i2 - i1 + 1;
			break;
		}
		else
		{
			if (i1 > lo && i2 == hi)
				sum += nums[--i1];
			else if (i1 == lo&&i2 < hi)
				sum += nums[++i2];
			else if (i1>lo&&i2 < hi)
			{
				if (nums[i1 - 1] < nums[i2 + 1])
					sum += nums[++i2];
				else
					sum += nums[--i1];
			}
			else
				break;
		}
	}
	if (!len)
	{
		if (!l1)return l2;
		if (!l2)return l1;
		return min(l1, l2);
	}
	else if (!l1)
	{
		if (!l2)return len;
		return min(len, l2);
	}
	else if (!l2)
		return min(l1, len);
	else
		return min(len, min(l1, l2));
}
int Solution::minLength(int s, vector<int>& nums)
{
	/*int i = 0, j = 0;
	int sum = 0;
	int len = nums.size() + 1;
	if (len <= 1)return 0;
	while (i < nums.size())
	{
	if (sum < s)
	{
	if (j == nums.size())
	break;
	sum += nums[j++];

	}
	else
	{
	if (i == j)return 1;
	if (j - i< len)len = j - i;
	sum -= nums[i++];
	}
	}
	if (len == nums.size() + 1)return 0;
	return len;*/
	return minL_helper(s, nums, 0, nums.size() - 1);
}
int Solution::subcompare(string& num1, string& num2)
{
	int len1 = num1.length();
	int len2 = num2.length();
	if (len2<len1)return -subcompare(num2, num1);
	int k = 0, j = 0;
	for (; k<len2 - len1; k++)
		if (num2[k]>'0') return -1;
	while (j<len1&&k<len2)
	{
		if (num1[j]<num2[k]) return -1;
		else if (num1[j]>num2[k]) return 1;
	}
	return 0;
}
int Solution::compareVersion(string version1, string version2)
{
	int v1, v2;
	int l1 = version1.length();
	int l2 = version2.length();
	int i1 = 0, i2 = 0;
	while (i1<l1 || i2<l2)
	{
		v1 = v2 = 0;
		while (i1<l1&&version1[i1] != '.')
		{
			v1 = v1 * 10 + version1[i1] - '0';
			i1++;
		}
		while (i2<l2&&version2[i2] != '.')
		{
			v2 = v2 * 10 + version2[i2] - '0';
			i2++;
		}
		if (v1<v2)return -1;
		if (v2<v1)return 1;
		if (i1<l1)i1++;
		if (i2<l2)i2++;
	}
	return 0;
}
int Solution::minDistance(string word1, string word2)
{
	int m = word1.size();
	int n = word2.size();
	vector<vector<int> > distMat(m + 1, vector<int>(n + 1, 0));
	for (int i = 0; i <= m; i++)
		distMat[i][0] = i;
	for (int j = 0; j <= n; j++)
		distMat[0][j] = j;
	for (int i = 1; i <= m; i++)
	{
		for (int j = 0; j <= n; j++)
		{
			distMat[i][j] = min(distMat[i - 1][j] + 1, min(distMat[i][j - 1] + 1, distMat[i - 1][j - 1] + (word1[i - 1] == word2[j - 1] ? 0 : 1)));
		}
	}
	return distMat[m][n];
}
bool check_map(int mark[256], int chs[256])
{
	for (int i = 0; i < 256; i++)
	{
		if (mark[i]&&mark[i]>chs[i])
			return false;
	}
	return true;
}
string Solution::minWindow(string s, string t)
{
	int chs[256];
	int mark[256];
	memset(mark, 0, sizeof(mark));
	memset(chs, 0, sizeof(chs));
	for (int i = 0; i < t.length(); i++)
		mark[t[i]]++;
	int lo, hi, count = 0;
	count=lo = hi = 0;
	int lind = -2 - s.length(), rind = -1;
	while (s[lo])
	{
		if (count < t.length() || !check_map(mark, chs))
		{
			if (s[hi])
			{
				if (mark[s[hi]])
				{
					count++;
					chs[s[hi]]++;
				}
				hi++;
			}
			else
				break;
		}
		else
		{
			if (mark[s[lo]])
			{
				chs[s[lo]]--;
				count--;
			}
			if (hi - lo < rind - lind)
			{
				lind = lo;
				rind = hi;
			}
			lo++;
		}
	}
	if (rind - lind > s.length())
		return "";
	return s.substr(lind, rind - lind);
}
ListNode* Solution::reverseList(ListNode* head) // recursive
{
	if (!head || !head->next)return head;
	ListNode* q = head; head = head->next;
	q->next = NULL;
	ListNode* p = reverseList(head);
	head->next = q;
	return p;
}
bool Solution::isIsomorphic(string s, string t)
{
	if (s.length() != t.length())
		return false;
	if (s.length() < 1)return true;
	map<char, char> s2t;
	for (int k = 0; k < s.size(); k++)
	{
		s2t[s[k]] = t[k];
	}
	set<char>ttp(t.begin(), t.end());
	if (s2t.size() != ttp.size())
		return false;
	string temp = s;
	for (int k = 0; k < s.size(); k++)
	{
		temp[k] = s2t[temp[k]];
	}
	return temp == t;
}
bool Solution::hasDuplicate(vector<int>& nums)
{
	return set<int>(nums.begin(), nums.end()).size() < nums.size();
}
int Solution::largestRectangleArea(vector<int>& height) 
{
	if (height.size() < 1)return 0;
	if (height.size() < 2)return height[0];
	int  min = height[0], max = 0;;
	for (int k = 1; k < height.size(); k++)
	{
		
	}
}
ListNode* Solution::removeElements(ListNode* head, int val)
{
	ListNode tp(0);
	tp.next = head;
	ListNode* q = &tp;
	while (q&&q->next)
	{
		if (q->next->val == val)
		{
			ListNode* p = q->next;
			q->next = p->next;
			delete p;
		}
		else
			q = q->next;

	}
	return tp.next;
}
bool Solution::containsNearbyDuplicate(vector<int>& nums, int k)
{
	map<int, int> index;
	for (int s = 0; s<nums.size(); s++)
	{
		if (index.find(nums[s]) != index.end())
		{
			if (s - index[nums[s]] <= k)
				return true;
		}
		index[nums[s]] = s;
	}
	return false;
}
bool Solution::isHappy(int n) 
{
	vector<bool> have_see(10, 0);
	if (n <= 0)return false;
	while (n!=1)
	{
		if (n < 10)
		{
			if (have_see[n])
				return false;
			have_see[n] = true;
		}
		int t = 0;
		while (n)
		{
			int tp = n % 10;
			if (tp)t += tp*tp;
			n /= 10;
		}
		n = t;
	}
	return n == 1;
}
int largestRectangleArea(vector<int>& height)
{
	int max_are = 0;
	stack<int> st;
	for (int i = 0; i <= height.size(); i++)
	{
		while(!st.empty() && (i == height.size() || height[i] < height[st.top()]))
		{
			int ht = height[st.top()];
			max_are = max(max_are, ht*(st.empty() ? i : i - st.top()));
			st.pop();
		}
		st.push(i);
	}
	return max_are;
}
int Solution::minSubArrayLen(int s, vector<int>& nums) 
{
	int i = 0, j = 0;
	if (nums.size()<1)return 0;
	int len = nums.size() + 1;
	int sum = 0;
	while (i<nums.size())
	{
		if (sum<s)
		{
			if (j >= nums.size())
			{
				if (len>nums.size())return 0;
				return len;
			}
			sum += nums[j++];
		}
		else
		{
			if (j == i + 1)return 1;
			if (j - i<len)len = j - i;
			sum -= nums[i++];
		}
	}
	if (len>nums.size())return 0;
	return len;
}
int histMaxArea(vector<int>& heights)
{
	int max_area = 0;
	stack<int> st;
	for (int i = 0; i <= heights.size(); i++)
	{
		while (!st.empty() && (i == heights.size() || heights[i] < heights[st.top()]))
		{
			int ht = heights[st.top()];
			st.pop();
			max_area = max(max_area, ht*(st.empty()?i:i - st.top() - 1));
		}
		st.emplace(i);
	}
	return max_area;
}
int Solution::maximalRectangle(vector<vector<char>>& matrix)
{
	int m = matrix.size();
	if (m == 0)return 0;
	int n = matrix[0].size();
	if (n == 0)return 0;
	vector<int> hist(n, 0);
	int max_area = 0;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			hist[j]++;
			if (matrix[i][j] == '0')hist[j] = 0;
		}
		max_area = max(max_area, histMaxArea(hist));
	}
	return max_area;
}
bool Solution::isScramble(string s1, string s2)
{
	if (s1.length() != s2.length())
		return false;
	int len = s1.length();
	if (len == 1)
		return s1 == s2;
	string str11, str12, str21, str22;
	int res = false;
	string temp1 = s1;
	string temp2 = s2;
	sort(temp1.begin(), temp1.end()); // this is very important for speed
	sort(temp2.begin(), temp2.end());
	if (temp1 != temp2)return false;
	for (int i = 1; i < len&&!res; i++)
	{
		str11 = s1.substr(0, i);
		str12 = s1.substr(i, len - i);
		str21 = s2.substr(0, i);
		str22 = s2.substr(i, len - i);
		res = isScramble(str11, str21) && isScramble(str12, str22);
		if (!res)
		{
			str21 = s2.substr(0, len-i);
			str22 = s2.substr(len-i,i);
			res = isScramble(str11, str22) && isScramble(str12, str21);
		}
	}
	return res;
}
bool Solution::isScramble_DP(string s1, string s2)
{
	if (s1.length() != s2.length())
		return false;
	int len = s1.length();
	if (len == 1)
		return s1 == s2;
	vector<vector<vector<bool> > > DP(len, vector<vector<bool> >(len, vector<bool>(len, false)));
	//[k][i][j]:k is the length of the substr start form i and j
	for (int i = 0; i < len; i++)
	{
		for (int j = 0; j < len; j++)
		{
			if (s1[i] == s2[j])DP[1][i][j] = true;
		}
	}
	bool res = false;
	for (int k = 2; k <len; k++)
	{
		for (int i = 0; i <= len-k; i++)
		{
			for (int j = 0; j <= len-k; j++)
			{
				res = false;
				for (int st = 1; st < k&&!res; st++)
				{
					res = DP[st][i][j] && DP[k - st][i + st][j + st];
					if (!res)
					{
						res = DP[st][i][j + k - st] && DP[k - st][i + st][j];
					}
				}
				DP[k][i][j] = res;
			}
		}
	}
	res = false;
	for (int st = 1; st < len&&!res; st++)
	{
		res = DP[st][0][0] && DP[len - st][st][st];
		if (!res)
		{
			res = DP[st][0][len - st] && DP[len - st][st][0];
		}
	}
	return res;
}
void Solution::recoverTree(TreeNode* root)
{
	// using morris travelsal
	if (!root)return;
	TreeNode* cur = root;
	TreeNode* pre = NULL, *first = NULL, *second = NULL, *temp = NULL;
	while (cur)
	{
		if (!cur->left)
		{
			if (pre&&pre->val>cur->val)
			{
				if (!first)first = pre;
				second = cur;
			}
			pre = cur;
			cur = cur->right;
		}
		else
		{
			temp = cur->left;
			while (temp->right&&temp->right != cur)temp = temp->right;
			if (!temp->right)
			{
				temp->right = cur;
				cur = cur->left;
			}
			else
			{
				if (pre&&pre->val>cur->val)
				{
					if (!first)first = pre;
					second = cur;
				}
				pre = cur;
				cur = cur->right;
				temp->right = NULL;
			}
		}
	}
	if (first&&second)
		swap(first->val, second->val);
}
int Solution::numDistinct(string s, string t)
{
	if (s.length()<t.size())return 0;
	if (s.length() == t.length())return s == t;
	if (s.length() == 0)return 0;
	int m = s.length();
	int n = t.length();
	vector<int> vec(n + 1, 0);
	int val = 0;
	vec[0] = 1;
	for (int i = 1; i <= m; i++)
	{
		int len = min(n, i);
		for (int j = len; j >0; j--)
		{
			vec[j] +=(s[i - 1] == t[j - 1] ? vec[j-1] : 0);
		}
	}
	return vec[n];
}
void Solution::connect2(TreeLinkNode* root)
{
	TreeLinkNode* p = root;
	TreeLinkNode* head = root;
	while (head)
	{
		p = head;
		while (p)
		{
			if (p->right)
			{
				if (p->left)p->left->next = p->right;
				TreeLinkNode*q = p->next;
				while (q)
				{
					if (q->left)
					{
						p->right->next = q->left;
						break;
					}
					else if (q->right)
					{
						p->right->next = q->right;
						break;
					}
					else
						q = q->next;
				}
			}
			else if (p->left)
			{
				TreeLinkNode*q = p->next;
				while (q)
				{
					if (q->left)
					{
						p->left->next = q->left;
						break;
					}
					else if (q->right)
					{
						p->left->next = q->right;
						break;
					}
					else
						q = q->next;
				}
			}
			p = p->next;
		}
		while (head)
		{
			if (head->left)
			{
				head = head->left;
				break;
			}
			else if (head->right)
			{
				head = head->right;
				break;
			}
			else
				head = head->next;
		}
	}
}
int Solution::maxPath_helper(TreeNode* root)
{
	if (!root)return 0;
	int left = max(maxPath_helper(root->left), 0);
	int right = max(maxPath_helper(root->right), 0);
	max_val = max(max_val, left + right + root->val);
	return max(left, right) + root->val;
}
int Solution::maxPathSum(TreeNode* root)
{
	max_val = INT_MIN;
	maxPath_helper(root);
	return max_val;
}
int Solution::longestConsecutive(vector<int>& nums) 
{
	if (nums.size() < 1)return nums.size();
	unordered_set<int> nbs(nums.begin(), nums.end());
	int len = 0;
	while (!nbs.empty())
	{
		auto iter = nbs.cbegin();
		int start = *iter;
		int end = *iter;
		int le = 0, re = 0,ce=0;
		while (!nbs.empty()&&nbs.find(--start)!=nbs.end())
		{
			le+=nbs.erase(start);
		}
		while (!nbs.empty() && nbs.find(++end) != nbs.end())
		{
			re+=nbs.erase(end);
		}
		ce = nbs.erase(*iter);
		if (re + le + ce > len)len = re + le + ce;
	}
	return len;
}
vector<vector<string>> Solution::findLadders(string start, string end, unordered_set<string> &dict)
{
	unordered_map<string, vector<vector<string>>> set1, set2, *set_cur, *set_target;
	vector<vector<string>> vec_start(1, vector<string>(1, start)), vec_end(1, vector<string>(1, end));
	//two-end BFS to effectively prune, BFS strategy will get the smaller set to traverse in each iteration
	set1[start] = vec_start;
	set2[end] = vec_end;
	int K = start.size();
	bool isUpdated = true, isFinished = false;
	vector<vector<string>> res;
	for (int depth = 2; isUpdated && !isFinished; ++depth) {
		if (set1.size() > set2.size()) {
			set_cur = &set2;
			set_target = &set1;
		}
		else {
			set_cur = &set1;
			set_target = &set2;
		}
		unordered_map<string, vector<vector<string>>> inter_set;
		isUpdated = false;
		unordered_set<string> deleting;
		for (auto iteri = set_cur->cbegin(), end = set_cur->cend(); iteri != end; ++iteri) {
			for (int i = 0; i<K; ++i) {
				string temp = (*iteri).first;
				char ch = 'a';
				for (int c = 0; c<26; ++c) {
					temp[i] = ch + c;
					if (set_target->find(temp) != set_target->end()) {
						const vector<vector<string>> *new_paths_first = &(*iteri).second;
						const vector<vector<string>> *new_paths_second = &(*set_target)[temp];
						auto iter_in = set_cur->cbegin();
						if (((*iter_in).second)[0][0] != start) {
							new_paths_first = &(*set_target)[temp];
							new_paths_second = &(*iteri).second;
						}
						for_each(new_paths_first->cbegin(), new_paths_first->end(), [&](const vector<string> &first_path) {
							for_each(new_paths_second->cbegin(), new_paths_second->cend(), [&](const vector<string> &second_path) {
								vector<string> temp_path = first_path;
								for_each(second_path.crbegin(), second_path.crend(), [&](const string &s) {
									temp_path.push_back(s);
								});
								res.push_back(temp_path);
							});
						});
						isFinished = true;
					}
					else if (dict.find(temp) != dict.end()) {
						vector<vector<string>> new_paths = (*iteri).second;
						for_each(new_paths.begin(), new_paths.end(), [&](vector<string> &path) {
							path.push_back(temp);
						});
						if (inter_set.find(temp) == inter_set.end())
							inter_set[temp] = new_paths;
						else {
							for_each(new_paths.begin(), new_paths.end(), [&](vector<string> &path) {
								inter_set[temp].push_back(path);
							});
						}
						deleting.insert(temp);
						isUpdated = true;
					}
				}
			}
		}
		for_each(deleting.begin(), deleting.end(), [&](const string &s) {
			dict.erase(s);
		});
		*set_cur = inter_set;
	}
	return res;
}
bool isTure(string& s, int start, int end)
{
	if (start > end)return false;
	if (start == end)return true;
	while (start<end)
	{
		if (s[start] == s[end])
		{
			start--;
			end--;
		}
		else
			return false;
	}
}
int Solution::minCut(string s)
{
	vector<int> cuts(s.length(), 0);
	vector<vector<bool> > isParli(s.length(), vector<bool>(s.length(), true));
	// first build the parlidome look up table
	for (int k = 1; k < s.length(); k++)
	{
		int i = 0, j = k;
		while (j < s.length())
		{
			isParli[i][j] = (s[i] == s[j] && isParli[i + 1][j - 1]);
			i++;
			j++;
			// not when j=i+1.then (i+1,j-1)=(i+1,i) which is under the dialog and value of it is true so it is OK
		}
	}
	for (int i = 1; i < s.length(); i++)
	{
		if (!isParli[0][i])
		{
			int st = INT_MAX;
			for (int j = i; j>0; j--)
			{
				if (isParli[j][i])
					st = min(cuts[j - 1] + 1, st);
			}
			cuts[i] = st;
		}
	}
	return cuts[s.length()-1];
}
string Solution::longestCommonPrefix(vector<string>& strs) 
{
	int i = 0;
	if (strs.empty())return string("");
	for (; i<strs[0].length(); i++)
	{
		for (int j = 1; j<strs.size(); j++)
		if (strs[j][i] != strs[j - 1][i])
			return strs[0].substr(0, i);
	}
	return strs[0];
}
int Solution::candy(vector<int>& ratings) 
{
	//int n = ratings.size();
	//if (n < 2)return n;
	//// by topoligy sort
	//vector<int> degree(n,0);
	//vector<list<int> > Graph(n);
	//queue<pair<int,int> > qt;
	//for (int i = 1; i < n; i++)
	//{
	//	if (ratings[i]<ratings[i - 1])
	//	{
	//		Graph[i].push_back(i - 1);
	//		degree[i - 1]++;
	//	}
	//	else if (ratings[i] > ratings[i - 1])
	//	{
	//		Graph[i - 1].push_back(i);
	//		degree[i]++;
	//	}
	//	if (!degree[i - 1])
	//		qt.push(make_pair(i - 1,1));
	//}
	//if (!degree[n - 1])qt.push(make_pair(n - 1, 1));
	//int total = 0;
	//while (!qt.empty())
	//{
	//	pair<int, int> pa = qt.front();
	//	total += pa.second;
	//	auto iter = Graph[pa.first].begin();
	//	while (iter!=Graph[pa.first].end())
	//	{
	//		degree[*iter]--;
	//		if (!degree[*iter])
	//		{
	//			qt.push(make_pair(*iter, pa.second + 1));
	//		}
	//		iter++;
	//	}
	//	qt.pop();
	//}
	//return total;
	int total = 1, last = 1, i = 1;
	while (i < ratings.size())
	{
		if (ratings[i - 1] < ratings[i])
		{
			total += ++last;
			i++;
		}
		else if (ratings[i - 1] == ratings[i])
		{
			last = 1;
			total += last;
			i++;
		}
		else
		{
			int c = 1;
			while (i<ratings.size())
			{
				if (ratings[i] >= ratings[i - 1])
					break;
				c++; i++;
			}
			if (last >= c) total += c*(c - 1) / 2;
			else
				total = total - last + c*(c + 1) / 2;
			last = 1;
		}
	}
	return total;
}
RandomListNode *copyRandomList(RandomListNode *head) {

	//first, transform from 1->2->3->4 to 1->1->2->2->3->3->4->4
	if (!head)return head;
	RandomListNode* p = head;
	RandomListNode* q = NULL;
	while (p)
	{
		q = p->next;
		p->next = new RandomListNode(p->label);
		p->next->next = q;
		p = q;
	}
	// second, copy the random point
	p = head;
	while (p)
	{
		if (p->random)p->next->random = p->random->next;
		p = p->next->next;
	}
	// third, split the list
	p = head;
	q = head->next;
	p->next = NULL;
	RandomListNode *cur = q;
	while (cur)
	{
		p->next = cur->next;
		if (p->next)
		{
			cur->next = p->next->next;
			cur = cur->next;
			p = p->next;
		}
		else
			break;
	}
	return q;
}
unordered_map<string, vector<string> > dpm;
vector<string> wordBreak(string s, unordered_set<string>& wordDict) {
	if (dpm.count(s))return dpm[s];
	vector<string> res;
	if (s.empty())return res;
	if (wordDict.count(s))
		res.push_back(s);
	for (int i = 1; i < s.size(); i++)
	{
		string str = s.substr(0, i);
		if (wordDict.count(str))
		{
			string word = s.substr(i);
			vector<string> temp = wordBreak(word, wordDict);
			for (int si = 0; si < temp.size(); si++)
			{
				res.push_back(str + " " + temp[si]);
			}
		}

	}
	dpm[s] = res;
	return res;
}
int Solution::maxPoints(vector<Point>& points)
{
	int res = 0;
	for (int i = 0; i < points.size(); i++)
	{
		unordered_map<double, int> stat;
		int duplicate = 1, vertics = 0;
		for (int j = i + 1; j < points.size(); j++)
		{
			if (points[i].x == points[j].x)
			{
				if (points[i].y == points[j].y)
					duplicate++;
				else
					vertics++;
			}
			else
			{
				double slope = (double)(points[i].y - points[j].y) / (double)(points[i].x - points[j].x);
				stat[slope]++;
			}
		}
		int local = 0;
		for (auto it = stat.begin(); it != stat.end(); it++)
			local = max(it->second, local);
		local = max(local, vertics) + duplicate;
		res = max(local, res);
	}
	return res;
}
int Solution::maximumGap(vector<int>& nums)
{
	/**
	* using bucket sort to solve the problem:
	* fisrt, get the min and max of the nums;
	* then the most important thing is we observe that:
	* max_gap >= (max-min+nums.size()-1)/(nums.size())
	* so when we set the width of bucket as (max-min+nums.size()-1)/(nums.size())
	* the only thing that need to record is the min and max in a bucket
	*/
	if (nums.size() < 2)return 0;
	int max = *max_element(nums.begin(), nums.end());
	int min = *min_element(nums.begin(), nums.end());
	int bz = (max - min + nums.size() - 1) / nums.size();
	vector<pair<int, int>> bucket(nums.size(), make_pair(INT_MAX, INT_MIN));
	for (int i = 0; i < nums.size(); i++)
	{
		int index = (nums[i]-min) / bz;
		if (bucket[index].first>nums[i])bucket[index].first = nums[i];
		if (bucket[index].second<nums[i])bucket[index].second = nums[i];
	}
	int max_gap = 0, prev = bucket[0].second; // note,since min exsit so bucket[0] can not be (INT_MAX,INT_MIN);
	for (int i = 1; i < nums.size();i++)
	{
		if (bucket[i].first == INT_MAX)
			continue;
		max_gap = std::max(max_gap, bucket[i].first - prev);
		prev = bucket[i].second;
	}
	return max_gap;
}
int Solution::calculateMinimumHP(vector<vector<int>>& dungeon)
{
	/*int m = dungeon.size();
	if (!m)return 0;
	int n = dungeon[0].size();
	if (!n)return 0;

	dungeon[m - 1][n - 1] = min(0, dungeon[m-1][n-1]);
	for (int i = m - 2; i >= 0; i--)
		dungeon[i][n - 1] = min(0, dungeon[i + 1][n - 1] + dungeon[i][n - 1]);
	for (int j = n - 2; j >= 0; j--)
		dungeon[m - 1][j] = min(0, dungeon[m - 1][j + 1] + dungeon[m - 1][j]);
	for (int i = m - 2; i >= 0; i--)
	{
		for (int j = n - 2; j >= 0; j--)
		{
			dungeon[i][j] += max(dungeon[i + 1][j], dungeon[i][j + 1]);
			dungeon[i][j] = min(0, dungeon[i][j]);
		}
	}
	return 1 - dungeon[0][0];*/
	int N = dungeon.size();
	int M = dungeon[0].size();

	// just pick a simple path through the dungeon to obtain an upperbound
	int lowerbound = 0;
	int upperbound = INT_MAX;

	// A number so small impossible to come back alive from
	const int dead = INT_MIN / 3;

	// Binary search looking for the smallest starting health which we
	// survive from. Invariant we maintain is lowerbound dies and
	// upperbound survives
	while (lowerbound < upperbound - 1) {
		int mid = (upperbound - lowerbound) / 2 + lowerbound;

		// create a buffer N + 1 and M + 1 size so we have sentinal values
		// padding the first row and column.
		auto cur_health = vector<vector<int> >(N + 1);
		for (int n = 0; n <= N; n++) {
			cur_health[n].resize(M + 1, dead);
		}

		// Seed in our starting health
		cur_health[0][1] = cur_health[1][0] = mid;
		for (int n = 1; n <= N; n++) {
			for (int m = 1; m <= M; m++) {
				cur_health[n][m] = max(cur_health[n - 1][m], cur_health[n][m - 1]) + dungeon[n - 1][m - 1];
				if (cur_health[n][m] < 1) {
					// Once we are dead, ensure we stay dead
					cur_health[n][m] = dead;
				}
			}
		}

		// If we have positive health at the end we survived!
		if (cur_health[N][M] > 0) {
			upperbound = mid;
		}
		else
		{
			if (cur_health[N][M]<0){
				lowerbound = mid;
			}
			else
				return mid + 1;
		}
	}
	return upperbound;
}
vector<pair<int, int>> Solution::getSkyline(vector<vector<int>>& buildings) {
	int len = buildings.size();
	int cur = 0, cur_X, cur_H;
	priority_queue<pair<int, int> >lvbd;
	vector<pair<int, int>> res;
	while (cur < len || !lvbd.empty())
	{
		cur_X = lvbd.empty() ? buildings[cur][0] : lvbd.top().second;
		if (cur>=len||buildings[cur][0]>cur_X)
		{
			while (!lvbd.empty()&&(lvbd.top().second <= cur_X))
				lvbd.pop();
		}
		else
		{
			cur_X = buildings[cur][0];
			while (cur<len&&cur_X==buildings[cur][0])
			{
				lvbd.push(make_pair(buildings[cur][2], buildings[cur][1]));
				cur++;
			}
		}
		cur_H = lvbd.empty() ? 0 : lvbd.top().first;
		if (res.empty() || res.back().second != cur_H)
			res.push_back(make_pair(cur_X, cur_H));
	}
	return res;
}
int Solution::rob(vector<int>& nums)
{
	/** for the easy one
	if (nums.size() < 1)return 0;
	vector<int> money(nums.size() + 1, 0);
	money[1] = nums[0];
	for (int i = 1; i < nums.size(); i++)
		money[i + 1] = max(money[i - 1]+nums[i], money[i]);
	return money[nums.size()];
	*/
	
	// for the extend problem
	int len = nums.size();
	if (len < 1)return 0;
	if (len == 1)return nums[0];
	if (len == 2)return nums[1];

	vector<int> money(len, 0);
	money[1] = nums[0]; 
	for (int i = 1; i < len-1; i++)
		money[i + 1] = max(money[i - 1] + nums[i], money[i]);
	int max1=money[len-1];
	money[1] = nums[1];
	for (int i = 2; i < len; i++)
		money[i] = max(money[i - 2] + nums[i], money[i-1]);
	return max(max1, money[len - 1]);
}
string Solution::shortestPalindrome(string s)
{
	/*the recursive solution:
	string temp = s;
	reverse(temp.begin(),temp.end());
	if (temp == s)return s;
	temp = s.substr(0, s.length() - 1);
	temp = shortestPalindrome(temp);
	temp.insert(temp.begin(), s.back());
	temp.append(s.substr(s.length()-1,1));
	return temp;
	*/

	int len = s.length();
	string T(2 * (len + 1), '#');
	T[0] = '$';
	int n = T.length();
	int mx = 0, id = 0;
	vector<int> lens(n,0);
	for (int i = 0; i < len; i++)
		T[2 * (i + 1)] = s[i];
	for (int i = 1; i < n; i++)
	{
		if (mx>i)
			lens[i] = min(lens[2 * id - i], mx - i);
		else
			lens[i] = 1;
		while (T[i - lens[i]] == T[i + lens[i]])
			lens[i]++;
		if (i + lens[i]>mx)
		{
			id = i;
			mx = i + lens[i];
		}
	}
	int maxLen = 0;
	for (int i = n-2; i >0; i--)
	{
		if (i == lens[i])
		{
			maxLen = lens[i]-1;
			break;
		}
	}
	string temp = s.substr(maxLen);
	reverse(temp.begin(), temp.end());
	return temp + s;
}
int Solution::findKthLargest(vector<int>& nums, int k) 
{
	if (nums.size() < k)return INT_MIN;
	vector<int> pool(k, 0);
	for (int i = 0; i < k; i++)
		pool[i] = nums[i];
	make_heap(pool.begin(), pool.end(),greater<int>());
	for (int i = k; i < nums.size(); i++)
	{
		if (pool[0] < nums[i])
		{
			pop_heap(pool.begin(), pool.end(), greater<int>());
			pool.pop_back(); pool.push_back(nums[i]);
			push_heap(pool.begin(), pool.end(), greater<int>());
		}
	}
	return *min_element(pool.begin(), pool.end());
}
vector<vector<int>> cmb_res;
void dfs_combineSum3(int k, int n, vector<int> cur,int sum)
{
	if (cur.size() == k)
	{
		if (sum==n)
			cmb_res.push_back(cur);
		return;
	}
	int pos = cur.empty() ? 1 : cur.back()+1;
	for (; pos <= 9; pos++)
	{
		if (cur.empty()||pos<=n-sum)
		{
			cur.push_back(pos);
			dfs_combineSum3(k,n,cur,sum + pos);
			cur.pop_back();
		}
	}
}
vector<vector<int>> Solution::combinationSum3(int k, int n)
{
	vector<int> cur;
	dfs_combineSum3(k, n, cur, 0);
	return cmb_res;
}
bool Solution::containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t)
{
	if (nums.size() < 2)return false;
	set<int> prevoius;
	for (int i = 0; i < nums.size()&&i<k; i++)
	{
		auto it = prevoius.lower_bound(nums[i] - t);
		if (it != prevoius.end()&&abs(nums[i] - *it) <= t)
			return true;
		prevoius.insert(nums[i]);
	}
	for (int i = k; i < nums.size(); i++)
	{
		auto it = prevoius.lower_bound(nums[i] - t);
		if (it != prevoius.end()&&abs(nums[i] - *it) <= t)
			return true;
		prevoius.insert(nums[i]);
		prevoius.erase(nums[i - k]);
	}
	return false;
}
int Solution::maximalSquare(vector<vector<char>>& matrix) 
{
	int m = matrix.size();
	if (!m)return 0;
	int n = matrix[0].size();
	if (!n)return 0;

	vector<int> dp(n+1,0);
	int max_size = 0;
	int flag = 1, temp, lastTopLeft = 0;
	for (int i = 1; i <= m; i++)
	{	
		for (int j = 1; j <= n; j++)
		{
			temp = dp[j];
			if (matrix[i-1][j-1] == '1')
			{
				dp[j] = min(dp[j], min(dp[j-1], lastTopLeft)) + 1;
				max_size = max(max_size, dp[j]);
			}
			else
				dp[j] = 0;
			lastTopLeft = temp;
		}
		flag = (flag + 1) % 2;
	}
	return max_size*max_size;
}
int Solution::countNodes(TreeNode* root)
{
	if (!root)return 0;
	TreeNode* p = root->left;
	TreeNode* q = root->right;
	int left = 0, right = 0;
	while (p)
	{
		p = p->left;
		left++;
	}
	while (q)
	{
		q = q->left;
		right++;
	}
	if (left == right)
		return countNodes(root->right) + (1 << left);
	else
		return countNodes(root->left) + (1 << right);
}
int overlap_interval(int l1, int r1, int l2, int r2)
{
	return min(r1, r2) - max(l1, l2);
}
int Solution::computeArea(int A, int B, int C, int D, int E, int F, int G, int H)
{
	int v1 = max(0, overlap_interval(A, C, E, G));
	int v2 = max(0, overlap_interval(B, D, F, H));
	return (C - A)*(D - B) + (G - E)*(H - F) - v1*v2;
}
vector<int> twoSum(vector<int>& nums, int target) 
{
	vector<int> temp = nums;
	sort(temp.begin(), temp.end());
	int lo = 0, hi = nums.size() - 1;
	while (lo<hi)
	{
		int val = temp[lo] + temp[hi];
		if (val>target)
			hi--;
		else if (val<target)
			lo++;
		else
			break;
	}
	vector<int>indexs(2);
	for (int i = 0, j = 0; i<nums.size() && j<2; i++)
	{
		if (temp[lo] == nums[i])
		{
			indexs[j++] = i + 1;
			temp[lo] = temp[hi];
		}
		else if (temp[hi] == nums[i])
		{
			indexs[j++] = i + 1;
			temp[hi] = temp[lo];
		}
	}
	if (indexs[0]>indexs[1])
		swap(indexs[0], indexs[1]);
	return indexs;
}
int Solution::calculate(string s)
{
	// first split the express into number and operater
	string suffix(s.length()*2, 0);
	int top = 0;
	stack<char> ops;
	for (int i = 0; i < s.length(); i++)
	{
		if (s[i] >= '0' && s[i] <= '9')
			suffix[top++] = s[i];
		else if (s[i] == ')')
		{
			suffix[top++] = ' ';
			while (!ops.empty() && ops.top() != '(')
			{
				suffix[top++] = ' ';
				suffix[top++] = ops.top();
				ops.pop();
			}
			ops.pop();
		}
		else if (s[i] == ' ')
		{
			suffix[top++] = ' ';
			while (s[i] == ' ')i++;
			i--;
		}
		else
		{
			suffix[top++] = ' ';
			if (s[i] != '(')
			{
				while (!ops.empty() && ops.top() != '(')
				{
					suffix[top++] = ' ';
					suffix[top++] = ops.top();
					ops.pop();
				}
			}
			ops.push(s[i]);
		}
	}
	while (!ops.empty())
	{
		suffix[top++] = ' ';
		suffix[top++] = ops.top();
		ops.pop();
	}
	int cur = 0;
	stack<int> nums;
	int v1, v2;
	for (int i = 0; i <=top; i++)
	{
		if (suffix[i] >= '0' && suffix[i] <= '9')
			cur = cur * 10 + suffix[i] - '0';
		else
		{
			if (suffix[i] == '+')
			{
				v1 = nums.top(); nums.pop();
				v2 = nums.top(); nums.pop();
				nums.push(v2 + v1);
				while (i<top&&suffix[i+1] == ' ')i++;
			}
			else if (suffix[i] == '-')
			{
				v1 = nums.top(); nums.pop();
				v2 = nums.top(); nums.pop();
				nums.push(v2 - v1);
				while (i<top&&suffix[i + 1] == ' ')i++;
			}
			else
			{
				nums.push(cur);
				while (i<top&&suffix[i + 1] == ' ')i++;
				cur = 0;
			}
		}
	}
	while (nums.size()>1 && nums.top() == 0)
		nums.pop();
	return nums.top();
	/*
	// the given expression is always valid!!!
        // only + and - !!!
        // every + and - can be flipped base on it's depth in ().
        stack<int> signs;
        int sign = 1;
        int num = 0;
        int ans = 0;

        // always transform s into ( s )
        signs.push(1);

        for (auto c : s) {
            if (c >= '0' && c <= '9') {
                num = 10 * num + c - '0';
            } else if (c == '+' || c == '-') {
                ans = ans + signs.top() * sign * num;
                num = 0;
                sign = (c == '+' ? 1 : -1);
            } else if (c == '(') {
                signs.push(sign * signs.top());
                sign = 1;
            } else if (c == ')') {
                ans = ans + signs.top() * sign * num;
                num = 0;
                signs.pop();
                sign = 1;
            }
        }

        if (num) {
            ans = ans + signs.top() * sign * num;
        }

        return ans;
    }*/
}
string Solution::convert(string s,int numRows)
{
	if (numRows<2)return s;
	vector<string> strs(numRows);
	int index = 0;
	while (index<s.length())
	{
		for (int i = 0; i<numRows&&index<s.length(); i++)
			strs[i].append({ s[index++] });
		for (int i = numRows - 2; i>0 && index<s.length(); i--)
			strs[i].append({ s[index++] });
	}
	string res;
	for (int i = 0; i<numRows; i++)
		res.append(strs[i]);
	return res;
}
int Solution::myAtoi(string str)
{
	if (str.empty())return 0;
	int lo = 0; while (isspace(str[lo]))lo++;
	int hi = str.length() - 1; while (isspace(str[hi]))hi--;
	if (lo>hi)return 0;
	int flag = 1;
	if (str[lo] == '+')lo++;
	else if (str[lo] == '-')
	{
		lo++;
		flag = -1;
	}
	long long val = 0;
	for (int i = lo; i <= hi; i++)
	{
		if (str[i] >= '0'&&str[i] <= '9')
			val = val * 10 + str[i] - '0';
		else
			break;
		if (val*flag>INT_MAX) return INT_MAX;
		if (val*flag<INT_MIN) return INT_MIN;
	}
	return val*flag;
}
ListNode* Solution::removeNthFromEnd(ListNode* head, int n) {
	ListNode hd(0); hd.next = head;
	ListNode* p = &hd, *q = &hd;
	for (int i = 0; i<n + 1 && q; i++)q = q->next;
	while (q){
		q = q->next;
		p = p->next;
	}
	if (p->next){
		q = p->next;
		p->next = q->next;
		delete q;
	}
	return hd.next;
}
ListNode* Solution::mergeTwoLists(ListNode* l1, ListNode* l2) 
{
	ListNode dump(0);
	ListNode* p = &dump;
	while (l1&&l2){
		if (l1->val < l2->val){
			p->next = l1;
			l1 = l1->next;
			p = p->next;
		}
		else{
			p->next = l2;
			l2 = l2->next;
			p = p->next;
		}
	}
	while (l1){
		p->next = l1;
		l1 = l1->next;
		p = p->next;
	}
	while (l2){
		p->next = l2;
		l2 = l2->next;
		p = p->next;
	}
	return dump.next;
}
int Solution::getGCD(int m, int n)
{
	/*if (m > n)return getGCD(n,m);
	if (m == 0)return n;
	if (!(n%m))return m;
	else return getGCD(n%m, m);*/
	if (m == 0)return n;
	if (n == 0)return m;
	int a = m, b = n;
	int r = a%b;
	while (r){
		a = b;
		b = r;
		r = a%b;
	}
	return b;
}
int Solution::getLCM(int m, int n)
{
	if (m == 0 || n == 0)return 0;
	int gcd = getGCD(m, n);
	return (m / gcd)*n;
}