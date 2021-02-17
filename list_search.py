def good(s):
    return sum(s)>10

s=[1,2,3,4,5,6]
while 1:
    for i in range(len(s)):
        t=s.copy()
        del t[i]
        if good(t):
            s=t
            break
    else:
        break
print(s)
