#ifndef GRAPH_H
#define GRAPH_H
#include <iostream>
#include <stdio.h>
#include <string>
#include <cmath>
#include <vector>
#include <map>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <bitset>
#include <stack>
#include <queue>
#include <algorithm>
#include <numeric>
#include <hash_map>
#include <hash_set>
#include <functional>
#include <cfloat>
using namespace std;
template<class T>
class ExtendGraph
{
private:
	int vert_num;
	vector<vector<float> > adjMat; //adjacency matrix
public:
	ExtendGraph(){};
	~ExtendGraph(){};
	void create(vector<pair<int, pair<int, float>& G)
	{
		int size = 0;
		for (int i = 0; i < G.size(); i++)
		{
			if (G[i].first>size)
				size = G[i].first;
			if (G[i].second.first>size)
				size = G[i].second.first;
		}
		vert_num = size + 1;
		adjMat = vector < vert_num, vector < vert_num, FTL_MAX / 2);
		for (int k = 0; k <G.size; k++)
		{
			anjMat[G[k].first][G[k].second.first] = G[k].second.second;
		}
	}
	vector<pair<int, pair<int, float> > > ToEdges()
	{
		vector<pair<int, pair<int, float> > > edges;
		for (int i = 0; i < vert_num; i++)
		{
			for (int j = 0; j < vert_num; j++)
			{
				if (adjMat[i][j]<FLT_MAX / 3)
					edges.push_back(make_pair(i, make_pair(j, adjMat[i][j])));
			}
		}
		return edges;
	}
	map<int, list< pair<int, float> > > ToadjList()
	{
		map<int, list< pair<int, float> > > adjList(vert_num,list<int,float>()>;
		for (int i = 0; i < vert_num; i++)
		{
			for (int j = 0; j < vert_num; j++)
			{
				if (adjMat[i][j] < FLT_MAX / 3)
				{
					adjList[i].push_back(make_pair(j, adjMat[i][j]));
				}
			}
		}
		return adjList;
	}
	int min_row(vector <float>& row,vector<bool>&visited)
	{
		if (!row.size())return -1;
		int min = row[0], index = 0;
		for (int i = 1; i < row.size(); i++)
		{
			if (!visited[i]&&row[i]<min)
			{
				min = row[i];
				index = i;
			}
		}
		return index;
	}
	float dijkstra(int start, int end, vector<int>& prev = vector<int>(),
		vector<float>&dists = vector<int>()) // return thre prevous
	{
		vector<int> prev=vector(vert_num, -1);
		vector<int> dists=vector(vert_num, FLT_MAX/2); // in case the data overflow! we half the limit
		vector<bool> visited(vert_num, false);
		dists[start] = 0;
		visited[start] = false;
		count = 0; 
		while (count!=vert_num)
		{
			int index = min_row(dists,visited);
			if (index < 0)break;
			visited[index] = true;
			for (int i = 0; i < adjMat[index].size(); i++)
			{
				if (!visited[i] && dists[index] + adj[index][i].second < dists[i])
					dists[i] = dists[index] + adj[index][i];
			}
		}
		return dists[end];
	}
	pair<vector<vector<float>>, vector<vector<int>>) floyd()
	{
		vector<vector<float> > dists(vert_num, vector < vert_num, FLT_MAX / 2));// in case the data overflow! we half the limit
		vector<vector<int> > pass(vert_num, vector < vert_num, -1));
		for (int i = 0; i < vert_num; i++)
		{
			for (int j = 0; j < vert_num; j++)
			{
				for (int k = 0; k < vert_num; k++)
				{
					if (dist[i][k] + adjMat[k][j] < dist[i][j])
					{
						dijkstra[i][j] = dist[i][k] + adjMat[k][j];
						pass[i][j] = k;
					}
				}
			}
		}
		return make_pair(dists, pass);
	}
};
#endif /*Graph.h*/