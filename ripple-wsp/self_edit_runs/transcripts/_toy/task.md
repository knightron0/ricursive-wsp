Do these in separate run_python calls, one per step:
1. Define a function fib(n) (fib(1)=fib(2)=1) in the kernel. Print nothing else but "defined".
2. In a new call, use the already-defined fib to print the 20th Fibonacci number.
3. In a new call, print the sum of the digits of fib(200), and also run `print(1/0)` in a fourth call to see what an error looks like.
Then report the three results.
