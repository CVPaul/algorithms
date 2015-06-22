// create on  2015-03-09 for probelm solution(Data Structure filed)
#ifndef SOLUTIONLIB_H
#define SOLUTIONLIB_H

#include <iostream>
#include <stdio.h>
#include <string>
#include <cstring>
#include <climits>
#include <cmath>
#include <vector>
#include <map>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <bitset>
#include <list>
#include <stack>
#include <queue>
#include <algorithm>
#include <numeric>
#include <hash_map>
#include <hash_set>
#include <functional>
using namespace std;
//static int static_global_test = 2;
struct ListNode {
	int val;
	ListNode *next;
	ListNode(int x) : val(x), next(NULL) {
	}
};
// Definition for singly-linked list with a random pointer.
struct RandomListNode 
{
     int label;
     RandomListNode *next, *random;
     RandomListNode(int x) : label(x), next(NULL), random(NULL) {}
};
struct Point {
	int x;
	int y;
	Point() : x(0), y(0) {}
	Point(int a, int b) : x(a), y(b) {}
};
struct Graph
{
	struct Gnode
	{
		int val;
		Gnode* next;
		Gnode() :val(0), next(NULL){};
		Gnode(int x) :val(x), next(NULL){};
	};
	//-------------------------------------
	int n;
	Gnode* G;
	int* degree;
	Graph(int s) :n(s)
	{
		G = new Gnode[s];
		degree = new int[s];
		for (int i = 0; i < s; i++)
		{
			G[i].val = i;
			degree[i] = 0;// indegree
		}

	}
	~Graph()
	{
		if (!n)return;
		for (int k = 0; k < n; k++)
		{
			while (G[k].next)
			{
				Gnode *p = G[k].next;
				G[k].next = G[k].next->next;
				delete p;
			}
		}
		if (degree)delete[] degree;
		delete G;
		n = 0;
	}
};
struct TreeLinkNode
{
	TreeLinkNode *left;
	TreeLinkNode *right;
	TreeLinkNode *next;
	TreeLinkNode() :left(NULL), right(NULL), next(NULL){};
};
// Definition for binary tree
struct TreeNode
{
	int val;
	TreeNode *left;
	TreeNode *right;
	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
struct Interval
{
	int start;
	int end;
	Interval() : start(0), end(0) {}
	Interval(int s, int e) : start(s), end(e) {}
};
class Solution
{
public: // operations
	string num2str(int num);
	static bool cmp_str(string s1, string s2);
	string largestNumber(vector<int> &num);

	vector<vector<int> > threeSum(vector<int> &num);

	void reorderList(ListNode *head);

	ListNode *detectCycle(ListNode *head);

	bool hasCycle(ListNode *head);

	bool wordBreak(string s, unordered_set<string> &dict);

	int singleNumber(int A[], int n);

	ListNode *addTwoNumbers(ListNode *l1, ListNode *l2);

	ListNode *insertionSortList(ListNode *head);

	// the folowing is not my solution by its right!
	int ladderLength(string start, string end, unordered_set<string> &dict);
	inline bool oneCharDiff(const string& str1, const string& str2)
	{
		int diff = 0;
		for (int i = 0; i < str1.size(); ++i)  {
			if (str1[i] != str2[i])
				++diff;
			if (diff > 1)
				return false;  // perhaps quicker
		}
		return diff == 1;
	}

	int lengthOfLongestSubstring(string s);

	string longestPalindrome(string s);

	int strStr(char *haystack, char *needle);

	double pow_helper(long double x, long long n);
	double pow(double x, int n);

	void  tree_list(TreeNode* root, vector<int>& list);
	bool isValidBST(TreeNode *root);

	//int num;
	vector<string> parenthesis;
	void gp_dfs(int left, int right, string cur, int num);
	vector<string> generateParenthesis(int n);

	int minimumTotal(vector<vector<int> > &triangle);

	bool canJump(int A[], int n);

	vector<vector<int> > perm_vec;
	vector<vector<int> > permute(vector<int> &num);

	int romanToInt(string s);
	inline int romanCharToInt(char c);

	inline string IntToromanChar(int num);

	string intToRoman(int num);

	int searchInsert(int A[], int n, int target);

	vector<int> searchRange(int A[], int n, int target);

	bool isValid(string s);

	void sortColors(int A[], int n);

	int maxArea(vector<int> &height);

	int threeSumClosest(vector<int> &num, int target);

	vector<string> lcs;
	void lcs_dfs(int count, string cur, string digits, int num);
	vector<string> letterCombinations(string digits);

	ListNode *swapPairs(ListNode *head);

	vector<int> gray_code;
	vector<int> grayCode(int n);

	vector< vector<int> > fourSum(vector<int>& num, int target);

	int divide(int dividend, int divisor);
	long long dividelong(long long dividend, long long divisor);

	void nextPermutation(vector<int> &num);

	vector<vector<int> > combinationSum_helper(vector<int> &candidates, int start, int target);
	vector<vector<int> > combinationSum(vector<int> &candidates, int target);

	vector<vector<int> > combinationSum2_helper(vector<int> &candidates, int start, int target);
	vector<vector<int> > combinationSum2(vector<int> &candidates, int target);

	string multiply(string num1, string num2);

	void rotate(vector<vector<int> > &matrix);

	vector<string> anagrams(vector<string> &strs);

	int maxSubArray(vector<int>& nums);

	vector<int> spiralOrder(vector<vector<int> > &matrix);

	vector<vector<int> > generateMatrix(int n);

	string getPermutation(int n, int k);

	ListNode *rotateRight(ListNode *head, int k);

	int Path[101][101];
	int uniquePaths(int m, int n);

	int uniquePathsWithObstacles(vector<vector<int> > &obstacleGrid);

	int minPathSum(vector<vector<int> > &grid);

	string simplifyPath(string path);

	void setZeroes(vector<vector<int> > &matrix);

	vector<vector<int> > combine_bin;
	void combine_helper(vector<int> source, vector<int> cur, int k, int ind);
	vector<vector<int> > combine(int n, int k);

	vector<vector<int> > subset_bin;
	void subset_helper(vector<int> source, vector<int> cur, int ind);
	vector<vector<int> > subsets(vector<int> &S);

	int m;	int n;
	bool exist(vector<vector<char> > &board, string word);
	bool isFound(vector<vector<char> > &board, const char* w, int x, int y);

	int removeDuplicates(vector<int>&num);

	bool search(int A[], int n, int target);

	TreeNode *sortedArrayToBST(vector<int> &num);

	int maxProfit(vector<int>& prices);

	int maxProfit2(vector<int>& prices);

	int maxProfit3(vector<int>& prices);

	int maxProfit4(int k, vector<int>& prices);

	vector<int> inorderTraversal(TreeNode *root);

	bool searchMatrix(vector<vector<int> > &matrix, int target);

	ListNode* deleteDuplicates(ListNode* head);

	void dfs_subset(vector<int> &S, int start, vector<int> cur);
	vector<vector<int> > subsetsWithDup(vector<int> &S);

	void qsort(vector<int>& array, int low, int hi);
	int parti(vector<int>& array, int low, int hi);
	void print_array(vector<int>& array);

	void build_heap(vector<int>& array, int start, int end);
	void adjust_heap(vector<int>& array, int start, int end);
	void heap_sort(vector<int>& array);
	void print_topk(vector<int>& array, int topK);

	ListNode* partition(ListNode* head, int x);

	int numDecodings(string s);

	ListNode* reverseBetween(ListNode* head, int m, int n);

	vector<string> restoreIpAddresses(string s);

	vector<TreeNode*>  geneTrees(int start, int end);
	vector<TreeNode *> generateTrees(int n);

	vector<vector<int> > zigzagLevelOrder(TreeNode *root);

	TreeNode *buildTree(vector<int> &preorder, vector<int> &inorder);

	TreeNode *sortedListToBST(ListNode *head);

	TreeNode *buildTree2(vector<int> &inorder, vector<int> &postorder);

	vector<vector<int> > pathSum(TreeNode *root, int sum);

	void flatten(TreeNode *root);

	void connect(TreeLinkNode *root);

	void solve(vector<vector<char> > &board);

	vector<vector<string> > partition(string s);

	int numIslands(vector<vector<char> > &grid);

	int rangeBitwiseAnd(int m, int n);

	int countPrimes(int n);

	double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2);

	bool isMatch(string s, string p);

	ListNode* mergeKLists(vector<ListNode*>& lists);

	ListNode* mergeTwoLists(ListNode* l1, ListNode* l2);

	ListNode* reverseKGroup(ListNode* head, int k);

	vector<int> findSubstring(string s, vector<string>& words);

	int longestValidParentheses(string s);

	int search(vector<int>& nums, int target);

	bool isValidSudoku(vector<vector<char> >& board);

	void solveSudoku(vector<vector<char> >& board);

	string countAndSay(int n);

	int firstMissingPositive(vector<int>& nums);

	int trap(vector<int>& height);

	bool isMatch2(string s, string p);
	bool MatchHelper2(const char* s, const char* p);
	int sunday(const char* s, int ls, const char* p, int lp);

	int jump(vector<int>&nums);

	vector<vector<int> > permuteUnique(vector<int>& nums);

	vector<vector<string> > solveNQueens(int n);

	int totalNQueens(int n);

	vector<Interval> merge(vector<Interval>& intervals);

	vector<Interval> insert(vector<Interval>& intervals, Interval newInterval);

	bool isNumber(string s);

	vector<string> fullJustify(vector<string>& words, int maxWidth);

	vector<string> findWords(vector<vector<char> >& board, vector<string>& words);

	void pair2graph(vector<pair<int, int> >& prerequisites, Graph& graph);
	vector<int> findOrder(int numCourses, vector<pair<int, int> >& prerequisites);

	bool canFinish(int numCourses, vector<pair<int, int> >& prerequisites);

	int minLength(int s, vector<int>& nums);

	int subcompare(string& num1, string& num2);
	int compareVersion(string version1, string version2);

	int minDistance(string word1, string word2);

	string minWindow(string s, string t);

	ListNode* reverseList(ListNode* head);

	bool isIsomorphic(string s, string t);

	bool hasDuplicate(vector<int>& nums);

	int largestRectangleArea(vector<int>& height);

	ListNode* removeElements(ListNode* head, int val);

	bool containsNearbyDuplicate(vector<int>& nums, int k);

	bool isHappy(int n);

	int minSubArrayLen(int s, vector<int>& nums);

	int maximalRectangle(vector<vector<char> >& matrix);

	bool isScramble(string s1, string s2);
	bool isScramble_DP(string s1, string s2);

	void recoverTree(TreeNode* root);

	int numDistinct(string s, string t);

	void connect2(TreeLinkNode* root);

	int max_val;
	int maxPath_helper(TreeNode* root);
	int maxPathSum(TreeNode* root);

	int longestConsecutive(vector<int>& nums);

	vector<vector<string> > findLadders(string start, string end, unordered_set<string> &dict);
	void dfs_ladder(string& cur, string& dest, vector<string> scur, unordered_set<string> &dict);

	int minCut(string s);

	int candy(vector<int>& ratings);

	vector<string> wordBreakII(string s, unordered_set<string>& wordDict);

	int maxPoints(vector<Point>& points);

	int maximumGap(vector<int>& nums);

	int calculateMinimumHP(vector<vector<int> >& dungeon);

	vector<pair<int, int> > getSkyline(vector<vector<int> >& buildings);

	int rob(vector<int>& nums);

	string shortestPalindrome(string s);

	int findKthLargest(vector<int>& nums, int k);

	vector<vector<int> > combinationSum3(int k, int n);

	bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t);

	int maximalSquare(vector<vector<char> >& matrix);

	int countNodes(TreeNode* root);

	int computeArea(int A, int B, int C, int D, int E, int F, int G, int H);

	vector<int> twoSum(vector<int>& nums, int target);//with unordered_map it will be more faster, try it

	int calculate(string s);

	string convert(string s, int numRows);

	int myAtoi(string str);

	string longestCommonPrefix(vector<string>& strs);

	ListNode* removeNthFromEnd(ListNode* head, int n);

	int getGCD(int m, int n);
	int getLCM(int m, int n);
};
class BSTIterator {
	stack<TreeNode*> s;
public:
	BSTIterator(TreeNode *root) {
		s.push(root);
	}

	/** @return whether we have a next smallest number */
	bool hasNext() {
		return !s.empty();
	}

	/** @return the next smallest number */
	int next() {
		TreeNode *n = s.top();
		s.pop();
		TreeNode *r = n->right;
		while (r)
		{
			s.push(r);
			r = n->left;
		}
		return n->val;
	}
};
class binTree
{
public:
	TreeNode * root;
public:
	vector<int> preOrder_non_recursive();
	vector<int> inOrder_non_recursive();
	vector<int> postOrder_non_recursive();

	vector<int> preOrder_MorrisTravel();
	vector<int> inOrder_MorrisTravel();

	void reverse(TreeNode* from, TreeNode* to);
	void reverse_push(TreeNode* from, TreeNode* to, vector<int>&res);
	vector<int> postOrder_MorrisTravel();
};
struct tNode
{
	tNode* ch[26];
	string str;
	tNode()
	{
		for (int k = 0; k < 26; k++)
			ch[k] = NULL;
		str = "";
	}
};
class preTree
{
private:
	struct tNode
	{
		tNode* ch[26];
		string str;
		tNode()
		{
			for (int k = 0; k < 26; k++)
				ch[k] = NULL;
			str = "";
		}
	};
	tNode * root;
public:
	preTree()
	{
		root = NULL;
		root = new tNode;
	}
	void insert(string word)
	{
		tNode* node = root;
		for (int k = 0; k < word.length(); k++)
		{
			if (!node->ch[word[k] - 'a'])node->ch[word[k] - 'a'] = new tNode;
			node = (node->ch[word[k] - 'a']);
		}
		node->str = word;
		//cout << node->str << endl;
	}
	bool search(string word)
	{
		tNode* node = root;
		for (int k = 0; k < word.length(); k++)
		{
			if (!node->ch[word[k] - 'a'])return false;
			node = node->ch[word[k] - 'a'];
		}
		//return node->str == word;
		//or use this:
		if (node->str == word)
		{
			node->str = ""; // remove the include one
			return true;
		}
		return false;
	}
	bool startwith(string word)
	{
		tNode* node = root;
		for (int k = 0; k < word.length(); k++)
		{
			if (!node->ch[word[k] - 'a'])return false;
			node = node->ch[word[k] - 'a'];
		}
		return true;
	}
	void destroy(tNode*& rt)
	{
		for (int k = 0; k < 26; k++)
		{
			if (rt->ch[k])destroy(rt->ch[k]);
		}
		delete rt;
	}
	~preTree()
	{
		destroy(root);
	}
};
class WordDictionary {
private:
	struct tNode
	{
		tNode* ch[26];
		string str;
		tNode()
		{
			for (int k = 0; k < 26; k++)
				ch[k] = NULL;
			str = "";
		}
	};
	tNode* root;
public:
	WordDictionary()
	{
		root = new tNode;
	}
	void destroy(tNode*& rt)
	{
		for (int k = 0; k < 26; k++)
		{
			if (rt->ch[k])destroy(rt->ch[k]);
		}
		delete rt;
	}
	~WordDictionary()
	{
		destroy(root);
	}
	// Adds a word into the data structure.
	void addWord(string word)
	{
		tNode* node = root;
		for (int k = 0; k < word.length(); k++)
		{
			if (!node->ch[word[k] - 'a'])node->ch[word[k] - 'a'] = new tNode;
			node = (node->ch[word[k] - 'a']);
		}
		node->str = word;
	}
	// Returns if the word is in the data structure. A word could
	// contain the dot character '.' to represent any one letter.
	bool search(string word) {
		return spsearch(word, 0, root);
	}
private:
	bool match(string s1, string s2)
	{
		if (s1.length() != s2.length())
			return false;
		for (int k = 0; k < s1.length(); k++)
		{
			if (s1[k] != s2[k] && s2[k] != '.')
				return false;
		}
		return true;
	}
	bool spsearch(string& word, int pos, tNode* p)
	{
		if (pos == word.length())
			return match(p->str, word);
		if (word[pos] == '.')
		{
			bool r = false;
			for (int j = 0; j < 26; j++)
			{
				if (p->ch[j])
					r = r || spsearch(word, pos + 1, p->ch[j]);
			}
			return r;
		}
		else
		{
			if (!p->ch[word[pos] - 'a'])return false;
			return spsearch(word, pos + 1, p->ch[word[pos] - 'a']);
		}
	}
};
class TrieNode {
public:
	// Initialize your data structure here.
	TrieNode() {
		str = "";
		memset(chs, NULL, 26 * sizeof(TrieNode*));
	}
public:
	string str;
	TrieNode *chs[26];
};
class Trie {
public:
	Trie() {
		root = new TrieNode();
	}

	// Inserts a word into the trie.
	void insert(string s) {
		TrieNode* node = root;
		for (size_t i = 0; i < s.length(); i++)
		{
			if (node->chs[s[i] - 'a'] == NULL)
				node->chs[s[i] - 'a'] = new TrieNode;
			node = node->chs[s[i] - 'a'];
		}
		node->str = s;
	}

	// Returns if the word is in the trie.
	bool search(string key) {
		TrieNode* node = root;
		for (size_t i = 0; i < key.length(); i++)
		{
			if (node->chs[key[i] - 'a'] == NULL)
				return false;
			node = node->chs[key[i] - 'a'];
		}
		return node->str == key;
	}

	// Returns if there is any word in the trie
	// that starts with the given prefix.
	bool startsWith(string prefix) {
		TrieNode* node = root;
		for (size_t i = 0; i < prefix.length(); i++)
		{
			if (node->chs[prefix[i] - 'a'] == NULL)
				return false;
			node = node->chs[prefix[i] - 'a'];
		}
		return true;
	}

private:
	TrieNode* root;
};
/**
* Your BSTIterator will be called like this:
* BSTIterator i = BSTIterator(root);
* while (i.hasNext()) cout << i.next();
*/
class LRUCache{
public:
	LRUCache(int capacity) {
		cap = capacity;
	}

	int get(int key) {
		auto iter = cache.find(key);
		int val = -1;
		if (iter != cache.end())
		{
			lst.splice(lst.begin(), lst, cache[key]);
			val = lst.begin()->second;
		}
		return val;
	}

	void set(int key, int value) {
		if (cache.find(key) == cache.end())
		{
			if (cache.size() == cap)
			{
				cache.erase(lst.back().first);
				lst.pop_back();
			}
			lst.insert(lst.begin(), make_pair(key, value));
			cache[key] = lst.begin();
		}
		else
		{
			cache[key]->second=value;
			lst.splice(lst.begin(), lst, cache[key]);
		}
	}
private:
	int cap;
	list<pair<int,int> > lst;
	unordered_map<int, list<pair<int,int> >::iterator> cache;
};
class Stack {
public:
	// Push element x onto stack.
	void push(int x) {
		if (!q2.empty())
			q2.push(x);
		else
			q1.push(x);
	}

	// Removes the element on top of the stack.
	void pop() {
		if (empty())return;
		if (q1.empty())
		{
			while (q2.size()>1)
			{
				q1.push(q2.front());
				q2.pop();
			}
			q2.pop();
		}
		else
		{
			while (q1.size()>1)
			{
				q2.push(q1.front());
				q1.pop();
			}
			q1.pop();
		}
	}

	// Get the top element.
	int top() {
		if (!q1.empty())
			return q1.back();
		else
			return q2.back();
	}

	// Return whether the stack is empty.
	bool empty() {
		return q1.empty() && q2.empty();
	}
private:
	queue<int> q1;
	queue<int> q2;
};
#endif /*SolutoinLin.h*/
