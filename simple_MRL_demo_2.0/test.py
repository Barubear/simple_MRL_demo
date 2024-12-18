
a  = ['a1','a2','a3']
b  = ['b1','b2','b3']
c  = ['c1','c2','c3']




ai = 0
bi = 0
ci = 0

for _  in range(27):


    if (ci +1 )%3 == 0:
        if (bi +1 )%3 == 0:
            ci +=1
            bi +=1
            ai +=1
        else:
            ci +=1
            bi +=1
    else:
        ci+=1





    print(a[ai%3],b[bi%3],c[ci%3])

    
