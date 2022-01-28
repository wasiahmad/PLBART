### [[MBPP] Mostly Basic Python Problems](https://github.com/google-research/google-research/tree/master/mbpp)

Every example is a JSON object that has the following keys.

``` 
"text": "Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].", 
"code": "R = 3\r\nC = 3\r\ndef min_cost(cost, m, n): \r\n\ttc = [[0 for x in range(C)] for x in range(R)] \r\n\ttc[0][0] = cost[0][0] \r\n\tfor i in range(1, m+1): \r\n\t\ttc[i][0] = tc[i-1][0] + cost[i][0] \r\n\tfor j in range(1, n+1): \r\n\t\ttc[0][j] = tc[0][j-1] + cost[0][j] \r\n\tfor i in range(1, m+1): \r\n\t\tfor j in range(1, n+1): \r\n\t\t\ttc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j] \r\n\treturn tc[m][n]", 
"task_id": 1, 
"test_setup_code": "", 
"test_list": ["assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8", "assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12", "assert min_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16"], 
"challenge_test_list": []
```

Unlike Human-Eval, in the MBPP dataset, the input prompt is a problem description. For example,

```
Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].
```

We adopt the following modification of Human-Eval to evaluate a model generated code for MBPP examples as follows.

``` 
completion + "\n\n" + test_setup_code + ("\n\n" if test_setup_code else "") + "\n".join(test_list)
```

The dataset provides a ground-truth solutions for each problem and a list of tests.

```
R = 3
C = 3
def min_cost(cost, m, n): 
    tc = [[0 for x in range(C)] for x in range(R)] 
    tc[0][0] = cost[0][0] 
    for i in range(1, m+1): 
        tc[i][0] = tc[i-1][0] + cost[i][0] 
    for j in range(1, n+1): 
        tc[0][j] = tc[0][j-1] + cost[0][j] 
    for i in range(1, m+1): 
        for j in range(1, n+1): 
            tc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j] 
    return tc[m][n]

assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8
assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12
assert min_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16
```

We can perform evaluation on MBPP by running:

``` 
# evaluate plbart_base on human-eval using GPU-0
bash evaluate.sh 0 human-eval base
```

**Note that**, PLBART is trained to auto-complete function, therefore PLBART is not expected to perform well on MBPP
since the dataset consists of complete python programs.
