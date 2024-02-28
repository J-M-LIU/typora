## 1104 Sum of Number Segments (20 分)

Given a sequence of positive numbers, a segment is defined to be a consecutive subsequence. For example, given the sequence { 0.1, 0.2, 0.3, 0.4 }, we have 10 segments: (0.1) (0.1, 0.2) (0.1, 0.2, 0.3) (0.1, 0.2, 0.3, 0.4) (0.2) (0.2, 0.3) (0.2, 0.3, 0.4) (0.3) (0.3, 0.4) and (0.4).

Now given a sequence, you are supposed to find the sum of all the numbers in all the segments. For the previous example, the sum of all the 10 segments is 0.1 + 0.3 + 0.6 + 1.0 + 0.2 + 0.5 + 0.9 + 0.3 + 0.7 + 0.4 = 5.0.

### Input Specification:

Each input file contains one test case. For each case, the first line gives a positive integer *N*, the size of the sequence which is no more than 105. The next line contains *N* positive numbers in the sequence, each no more than 1.0, separated by a space.

### Output Specification:

For each test case, print in one line the sum of all the numbers in all the segments, accurate up to 2 decimal places.

### Sample Input:

```in
4
0.1 0.2 0.3 0.4
```

### Sample Output:

```out
5.00
```

**注意：数学问题，观察可发现，每个数字会存在 i * (n - i + 1) 次**

**1个片段包含当前数字temp，设这个片段的首和尾分别是p，q；p有i种可能，q有n-i+1种可能，所以1个数字存在 i * (n - i + 1)种**

