import sys

def fib(n):
    a, b = 0, 1
    result = []
    while(b<n):
        result.append(b)
        a, b = b, a+b
    print(result)

print("load function in here")
setattr(sys.modules[__name__], "FIBC", fib)