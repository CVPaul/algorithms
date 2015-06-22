//#include <iostream>
//#include <vector>
//#include <string>
//#include <map>
//#include <algorithm>
//#include <numeric>
//#include <fstream>
//#include "expvalue.h"
//#include "leetcode.h"
//using namespace std;
//template< class T>
//int check(string res, string sol,T type)
//{
//	ifstream ro(res.c_str());
//	ifstream so(sol.c_str());
//	int count = 0;
//	while (1)
//	{
//		if (ro.good()&&so.good())
//		{
//			T t1, t2;
//			ro >> t1;
//			so >> t2;
//			if (t1 != t2)
//			{
//				ro.close();
//				so.close();
//				printf("failed at %d cases(%d,%d)!\n", count,t1,t2);
//				return count;
//			}
//			count++;
//		}
//		else if (ro.eof() && so.eof())
//		{
//			ro.close();
//			so.close();
//			printf("Accepted!\n");
//			return count;
//		}
//		else
//		{
//			ro.close();
//			so.close();
//			printf("failed at %d cases!\n", count);
//			return count;
//		}
//	}
//	printf("failed at %d cases!\n", count);
//	ro.close();
//	so.close();
//	return count;
//}
//void p2003()
//{
//	long long n;
//	while (cin>>n)
//	{
//		if (n < 3)cout << n << endl;
//		else if (n & 1)cout<<n*(n - 1)*(n - 2)<<endl;
//		else if (n % 3 == 0) cout << (n - 1)*(n - 2)*(n - 3) << endl;
//		else cout << n*(n - 1)*(n - 3)<<endl;
//	}
//}
//int IP2Int(char* ip)
//{
//	int i = 0;
//	int value = 0;
//	unsigned int res = 0;
//	while (ip[i] != '\0')
//	{
//		if (ip[i]!='.')
//		{
//			value = value * 10 + ip[i] - '0';
//		}
//		else
//		{
//			res <<= 8;
//			res |= value;
//			value = 0;
//		}
//		i++;
//	}
//	res <<= 8;// last one
//	res |= value;
//	return res;
//}
//void P003()
//{
//	int T;
//	scanf("%d\n", &T);
//	int ips[1003];
//	int M[51];
//	char ip_str[20];
//	for(int c = 0; c < T; c++)
//	{
//		int n, m;
//		scanf("%d%d\n", &n, &m);
//		printf("Case #%d:\n",c+1);
//		for (int k = 0; k < n; k++)
//		{
//			gets(ip_str);
//			ips[k] = IP2Int(ip_str);
//		}
//		for (int i = 0; i < m; i++)
//		{
//			set<int> cset;
//			gets(ip_str);
//			int mask = IP2Int(ip_str);
//			for (int j = 0; j < n; j++)
//			{
//				cset.insert(ips[j] & mask);
//			}
//			printf("%d\n",cset.size());
//		}
//	}
//}
//int main() {
//	/*long long type(1);
//	check("2003.sol", "2003.out",type);
//	return 1;*/
//	P003();
//	return 0;
//}
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <climits>
#include <algorithm>
#include "leetcode.h"
using namespace std;


#define N  300
#define M  1000

int Big_rand(int i)
{
	int hi = rand() % N;
	int lo = rand() % M;
	double rnd = hi*M + lo;
	return (int)(rnd / (double)(N*M)*i);
}
vector<int> find_lucky_n(int n=100000)
{
	int sz = N*M;
	vector<int> elems(sz);
	for (int i = 0; i < sz; i++)
		elems[i] = i + 1;
	while (--sz)
	{
		int j = Big_rand(sz+1);
		swap(elems[sz], elems[j]);
		cout << sz<<" "<<j << endl;
	}
	return vector<int>(elems.begin(), elems.begin() + n);
}
char firstChar(string str)
{
	/**
	如果是UTF-8的话可用vector<int> hash_map(65536,0)代替,对于比较复杂的类型(ComplexType)可以用map<ComplexTye，int>代替，只是此时的
	速度有所影响，因为C++ map的底层是红黑树，不是hash_map
	*/
	vector<int> times(256,0);
	for (int i = 0; i < str.length(); i++)
		times[str[i]]++;
	for (int i = 0; i < str.length(); i++)
	{
		if (times[str[i]] == 1)
			return str[i];
	}
}
int Sum3(vector<int>& num,int target)
{
	if (num.size()<3)
		return accumulate(num.begin(),num.end(),0);
	sort(num.begin(), num.end());
	int gap = INT_MAX,ans;
	for (int i = 0; i<num.size() - 2; i++)
	{
		int lo = i + 1, hi = num.size() - 1;
		while (lo<hi)
		{
			int sum = num[i] + num[lo] + num[hi];
			if (abs(target-sum)<gap)	{
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
string maxLCM(string str1, string str2)
{
	int m = str1.length();
	int n = str2.length();
	vector<vector<int> > lens(m + 1, vector<int>(n + 1, 0));
	vector<vector<pair<int, int> > > path(m + 1, vector<pair<int, int> >(n + 1, make_pair(-1, -1)));
	for (int i = 1; i <= m; i++)
	{
		for (int j = 1; j <= n; j++)
		{
			if (str1[i - 1] == str2[j - 1])
			{
				lens[i][j] = lens[i - 1][j - 1] + 1;
				path[i][j] = make_pair(i - 1, j - 1);
			}
			else
			{
				if (lens[i - 1][j]>lens[i][j - 1])
				{
					lens[i][j] = lens[i - 1][j];
					path[i][j] = make_pair(i - 1, j);
				}
				else
				{
					lens[i][j] = lens[i][j - 1];
					path[i][j] = make_pair(i, j - 1);
				}
			}
		}
	}
	pair<int, int> pr = make_pair(m, n);
	string res = "";
	while (pr.first > 0 && pr.second > 0)
	{
		if (str1[pr.first - 1] == str2[pr.second - 1])
		{
			res.insert(res.begin(), str1[pr.first - 1]);
		}
		pr = path[pr.first][pr.second];
	}
	return res;
}
int main()
{
	string str1 = "ABCBDAB", str2 = "BDCABA";
	string res = maxLCM(str1, str2);
	cout << res << endl;
	return 0;
}

//int main() 
//{
//	//vector<int>res = find_lucky_n();
//	Solution slu;//"ADOBECODEBANC"
//	//cout << slu.minWindow("ADOBECODEBANC", "ABCA") << endl;
//	//cout << slu.isIsomorphic("aba", "aaa") << endl;
//	//ListNode* head = new ListNode(1);
//	//slu.removeElements(head, 1);
//	//cout << slu.isHappy(20);
//	//vector<vector<char>> matrix(1, vector<char>(1, '1'));
//	//cout << slu.maximalRectangle(matrix) << endl;
//	//cout << slu.isScramble("great", "rtage")<<endl;
//	//cout << slu.isScramble_DP("great", "rtage") << endl;
//	/*TreeLinkNode* root = new TreeLinkNode(1);
//	root->left = new TreeLinkNode(2);
//	root->right = new TreeLinkNode(3);
//	root->left->left = new TreeLinkNode(4);
//	root->left->right = new TreeLinkNode(5);
//	root->right->right = new TreeLinkNode(6);
//	slu.connect2(root);*/
//	//unordered_set<string> dict;
//	//dict.insert("hit");
//	//dict.insert("hot");
//	//dict.insert("dot");
//	//dict.insert("dog");
//	//dict.insert("lot");
//	//dict.insert("log");
//	//dict.insert("cog");
//	//string start = "hit";
//	//string end = "cog";
//	///*dict.insert("a");
//	//dict.insert("b");
//	//dict.insert("c");
//	//string start = "a";
//	//string end = "c";*/
//	//vector<vector<string> > res = slu.findLadders(start, end, dict);
//	////vector<vector<string> > res=slu.findLadders(start, end, dict);
//	/*vector<int> nums;
//	nums.push_back(0);
//	nums.push_back(-1);
//	slu.longestConsecutive(nums);*/
//	//cout << slu.minCut("cabababcbc");
//	/*vector<int> ratings(5,1);
//	ratings[1] = 5;
//	ratings[2] = 7;
//	cout << slu.candy(ratings)<<endl;*/
//	//cout << slu.maxPoints(vector<Point>());
//	/*vector<vector<int> > dmap;
//	int A1[] = { -2, -3, 3 }; dmap.push_back(vector<int>(A1, A1 + 3));
//	int A2[] = { -5, -10, 1 }; dmap.push_back(vector<int>(A2, A2 + 3));
//	int A3[] = { 10, 30, -5 }; dmap.push_back(vector<int>(A3, A3 + 3));
//	cout << slu.calculateMinimumHP(dmap);*/
//	/*int prs[] = { 2, 1, 4, 5, 2, 9, 7 };
//	vector<int> prices(prs, prs + 7);
//	slu.maxProfit4(2, prices);*/
//	//vector<int> bd(3, 0);
//	//vector<vector<int>> buildings;
//	//bd[0] = 2, bd[1] = 9, bd[2] = 10;//[2 9 10], [3 7 15], [5 12 12], [15 20 10], [19 24 8] 
//	//buildings.push_back(bd);
//	//bd[0] = 3, bd[1] = 7, bd[2] = 15;
//	//buildings.push_back(bd);
//	//bd[0] = 5, bd[1] = 12, bd[2] = 12;
//	//buildings.push_back(bd);
//	//bd[0] = 15, bd[1] = 20, bd[2] = 10;
//	//buildings.push_back(bd);
//	//bd[0] = 19, bd[1] = 24, bd[2] = 8;
//	//buildings.push_back(bd);
//	//slu.getSkyline(buildings);
//	//cout<<slu.shortestPalindrome("adcba")<<endl;
//	/*vector<int> nums;
//	nums.push_back(2);
//	nums.push_back(1);
//	cout << slu.findKthLargest(nums,1) << endl;*/
//	//vector<vector<int>> res = slu.combinationSum3(2, 7);
//	//vector<int> nums(2, -1);
//	//slu.containsNearbyAlmostDuplicate(nums, 1, 0);
//	//cout << global_test << endl;
//	//cout << static_global_test << endl;
//	//cout << firstChar("abaccdef") << endl;
//	//cout << slu.calculate("  1  -  (2 + 3 - 45) +(6 -1+8)   ") << endl;
//	//cout << slu.convert("abcdefghijk", 3) << endl;
//	//cout << slu.calculate("1 + 1+(2 +3)+4 -7") << endl;
//	/*int arr[] = { 1, -2, 3, 3, 4, 5, -4, -2, -10, 3, 5, 7, 7, 9, 2 };
//	vector<int> num(arr,arr+15);
//	vector<vector<int>> res=Sum3(num);*/
//	/*ListNode * pairs = new ListNode(1);
//	pairs->next = new ListNode(2);
//	pairs->next->next = new ListNode(3);
//	pairs->next->next->next = new ListNode(4);
//	ListNode * res=slu.swapPairs(pairs);*/
//	/*string vec;
//	vec.insert(vec.end(), 1);
//	vec.insert(vec.end(), 2);
//	vec.insert(vec.end(), 3);
//	cout << vec.size() << endl;*/
//	/*cout << slu.getPermutation(2, 1) << endl;*/
//	//TreeNode* tr = new TreeNode(1);
//	////tr->left = new TreeNode(2);
//	//binTree btr; btr.root = tr;
//	//btr.postOrder_MorrisTravel();
//	/*string s = "1222";
//	vector<string> res=slu.permuteWithDuplicate(s);
//	for (int i = 0; i < res.size(); i++)
//	{
//		cout << res[i] << endl;
//	}
//	cout << res.size() << endl;*/
//	int arr[] = { 1, 2, 3};
//	vector<int> nums(arr, arr + 3);
//	//vector<vector<int>> res=slu.permuteUnique(nums);
//	vector<vector<int>> res = slu.permute(nums);
//	for (int i = 0; i < res.size(); i++)
//	{
//		for (int j = 0; j < res[i].size();j++)
//			cout << res[i][j] << ' ';
//		cout << endl;
//	}
//	cout << res.size()<<endl;
//	/*freopen("input.txt", "r", stdin);
//	int a, b, c = 1;
//	while (scanf("%d%d", &a, &b) != EOF)
//	{
//		int gcd = slu.getGCD(a, b);
//		int lcm = slu.getLCM(a, b);
//		printf("Case #%d(%d,%d):%d\t%d\n",c++,a,b,gcd,lcm);
//	}*/
//	return 0;
//}
//class hash_map
//{
//	struct elem
//	{
//		int key;
//		int val;
//		elem* next;
//		elem() :key(0), val(0), next(NULL){}
//		elem(int k, int v) :key(k), val(v), next(NULL){};
//	};
//public:
//	hash_map(int mp_sz)
//	{
//		if (mp_sz < 0) 
//		{
//			map_size = 1;// at last one,in this situation the map is equalize with list
//			maps = new elem;
//		}
//		else
//		{
//			map_size = mp_sz;
//			maps = new elem[map_size];
//		}
//	}
//	~hash_map()
//	{
//		for (int i = 0; i < map_size; i++)
//		{
//			if (maps[i].next)
//			{
//				elem* p = maps[i].next;
//				while (p)
//				{
//					elem* q = p;
//					p = p->next;
//					delete q;
//				}
//			}
//		}
//		delete[] maps;
//		map_size = 0;
//	}
//	bool map_get(int key,int& val)
//	{
//		int index = key%map_size;
//		elem* p = &maps[index];
//		while (p&&p->next&&p->next->key != key)
//		{
//			p = p->next;
//		}
//		if (!p || !p->next || p->next->key != key)
//		{
//			return false;
//		}
//		else
//		{
//			val = p->next->val;
//			return true;
//		}
//	}
//	void map_put(int key, int val)
//	{
//		int index = key%map_size;
//		elem* cur = new elem(key, val);
//		cur->next = maps[index].next;
//		maps[index].next = cur;
//	}
//	void map_delete(int key)
//	{
//		elem* fd = find(key);
//		if (!fd) return;
//		elem* q = fd->next;
//		fd->next = fd->next->next;
//		delete q;
//	}
//	elem* find(int key)
//	{
//		int index = key%map_size;
//		elem* p = &maps[index];
//		while (p&&p->next&&p->next->key != key)
//		{
//			p = p->next;
//		}
//		if (!p || !p->next || p->next->key != key)
//			return NULL;
//		else
//			return p;
//	}
//private:
//	int map_size;
//	elem* maps;
//};
