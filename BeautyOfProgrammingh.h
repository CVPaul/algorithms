#ifndef BEATUTYOFPROGRAMMING
#define BEATUTYOFPROGRAMMING
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>
#define LL long long
#define PI 3.1415926535
void CPU_play()
{
	const int SAMPLE_COUNT = 200;
	const int TOTAL_AMPLITUDE = 300;
	LL busySpan[SAMPLE_COUNT];
	double radianImcreament = 2.0 / (double)(SAMPLE_COUNT);
	double amplitude = TOTAL_AMPLITUDE / 2;
	double radian = 0.0;
	for (int i = 0; i < SAMPLE_COUNT; i++)
	{
		busySpan[i] = (amplitude + sin(PI*radian)*amplitude);
		radian += radianImcreament;
	}
	LL startTime = 0;
	for (int j = 0;; j = (j + 1) % SAMPLE_COUNT)
	{
		startTime = GetTickCount();
		while ((GetTickCount() - startTime) < busySpan[j]);
		Sleep(TOTAL_AMPLITUDE - busySpan[j]);
	}
}
void ChineseChess()
{
	struct
	{
		unsigned char a : 4;
		unsigned char b : 4;
	} bits;
	for (bits.a = 1; bits.a <= 9; bits.a++)
	{
		for (bits.b = 1; bits.b <= 9; bits.b++)
		{
			if (bits.a % 3 == bits.b % 3)
				continue;
			printf("<%d,%d>\n", bits.a, bits.b);
		}
	}
}
#endif/*BeautyOfProgramming.h*/