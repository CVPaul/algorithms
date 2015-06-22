////#include <cstdio>   
////#include <iostream>
////#include <queue>
////#include <string.h>   
////#include <cmath>
////#include <vector>
////#include <algorithm>   
////using namespace std;
////typedef long long ll;
////#define sfint(x) scanf("%d",&x)
////#define sfint2(x,y) scanf("%d%d",&x,&y)
////#define sfint3(x,y,z) scanf("%d%d%d",&x,&y,&z)
////#define sfstr(c) scanf("%s",c)
////#define sfdl(x) scanf("%lf",&x)
////#define sfll(x) scanf("%I64d",&x)
////#define sfch(c) scanf("%c",&c)
////#define fr(i,s,n) for(int i=s;i<n;++i)
////#define cl(a) memset(a,0,sizeof(a))
////int n, K;
////const ll inf = 1LL << 60;
////ll dp[1010][1010];
////
////struct STA{
////	struct Sta{
////		ll x, y;
////		Sta(ll a, ll b) :x(a), y(b){}
////		Sta(){}
////	}s[1010];
////	int top, now;
////	void init(){
////		top = -1;
////		now = 0;
////	}
////	void P(ll a, ll b){
////		if (top <1) s[++top] = Sta(a, b);
////		else{
////			while (1){
////				if (top == 0){
////					s[++top] = Sta(a, b);
////					break;
////				}
////				ll y1 = b - s[top].y;
////				ll x1 = a - s[top].x;
////				ll y2 = s[top].y - s[top - 1].y;
////				ll x2 = s[top].x - s[top - 1].x;
////				if (y1 * x2 > y2 * x1){
////					s[++top] = Sta(a, b);
////					break;
////				}
////				else{
////					top--;
////				}
////			}
////		}
////	}
////	Sta Top(){
////		return s[top];
////	}
////}S;
////
////struct P{
////	ll h, w;
////	void read(){
////		scanf("%lld%lld", &h, &w);
////	}
////}p[1010];
////
////ll sw[1010], sm[1010];
////void init(){
////	sw[0] = 0; sm[0] = 0;
////	fr(i, 1, n + 1){
////		sw[i] = sw[i - 1] + p[i].w;
////		sm[i] = sm[i - 1] + p[i].w * p[i].h;
////	}
////}
////
////double cal(int x){
////	return double(S.s[x + 1].y - S.s[x].y) / double(S.s[x + 1].x - S.s[x].x);
////}
////int main(){
////	freopen("1001.in", "r", stdin);
////	freopen("1001.out", "w", stdout);
////	/*
////	还有一个很重要的问题：如果你不想输入或输出到文件了，要恢复句柄，可以重新打开标准控制台设备文件，这个设备文件的名字是与操作系统相关：
////	DOS/Win:  freopen("CON", "r", stdin);
////	freopen("CON", "w", stdout);
////	Linux:    freopen("/dev/console", "r", stdin);
////	*/
////	while (sfint2(n, K) != EOF){
////		fr(i, 1, n + 1){
////			p[i].read();
////		}
////		init();
////		fr(i, 0, n + 1){
////			fr(j, 0, K + 1){
////				dp[i][j] = inf;
////			}
////		}
////		dp[1][1] = 0;
////		S.init();
////		fr(i, 2, n + 1){
////			dp[i][1] = dp[i - 1][1] + sw[i - 1] * (p[i].h - p[i - 1].h);
////		}
////		for (int j = 2; j <= K; ++j){
////			for (int i = j; i<n + 1; ++i){
////				S.P(sw[i - 1], dp[i - 1][j - 1] + sm[i - 1]);
////				while (S.now<S.top &&  cal(S.now) < double(p[i].h)){
////					S.now++;
////				}
////				dp[i][j] = S.s[S.now].y - S.s[S.now].x * p[i].h + sw[i] * p[i].h - sm[i];
////
////			}
////			S.init();
////		}
////		printf("%lld\n", dp[n][K]);
////	}
////	return 0;
////}
//int IP2Int(char* ip)
//{
//	int i = 0;
//	int value = 0;
//	unsigned int res = 0;
//	while (ip[i] != '\0')
//	{
//		if (ip[i] != '.')
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
//	for (int c = 0; c < T; c++)
//	{
//		int n, m;
//		scanf("%d%d\n", &n, &m);
//		printf("Case #%d:\n", c + 1);
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
//			printf("%d\n", cset.size());
//		}
//	}
//}