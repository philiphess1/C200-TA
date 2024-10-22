import math
import random as rn
import matplotlib.pyplot as plt
import math


###########################################################################
# Functions for Problem 1
###########################################################################
#recursion
def h(n):
    if n <= 0:
        return 1
    else:
        return 2*n + h(n-1) + h(n-2)
    
#memoization
dh1 = {-2:1,-1:1,0:1,1:4}
def hmemo(n):

    if n in dh1.keys():
        return dh1[n]
    else:
        for i in range(2,n+1):
            dh1[i] = 2*i + dh1[i-1] + dh1[i-2]
    return dh1[n]

#while-loop
def hw(n):
    acc0 = 1
    acc1 = 4
    if n < 2:
        return [acc0,acc1][n]
    else:
        i = 2
        while i < n + 1:
            acc0,acc1 = acc1,2*i + acc0 + acc1
            i += 1
        return acc1
#tail recursion
def htr(n, i = 2, acc0 = 1, acc1 = 4):
    if n < 2:
        return [acc0,acc1][n]
    elif i < n + 1:
        return htr(n,i+1,acc1,2*i + acc0 + acc1)
    else:
        return acc1
    
#recursion
def p(n):
    if n:
        return p(n-1) + 0.02*p(n-1)
    else:
        return 10000
#tail recursion
def ptr(n,acc=1000):
    if n:
        return ptr(n-1,acc + 0.02*acc)
    else:
        return acc

#recursion 
def d(n):
    if n:
        return 3*d(n-1) + 1
    else:
        return 1

#tail recursion
def dtr(n,acc=1):
    if n:
        return dtr(n-1,3*acc + 1)
    else:
        return acc
    
#recursion
def c(n):
    if n > 1:
        return 9*c(n-1) + 10**(n-1) - c(n-1)
    else:
        return 9
#tail recursion
def ctr(n,acc1=9,acc2=0):
    if n > 1:
        return ctr(n-1, 8*acc1 + 10**(acc2+1), acc2+1)
    else:
        return acc1
#while-loop
def cw(n):
    if n:
        acc1 = 9
        acc2 = 0
        i = 2
        while i < n + 1:
            acc1 = 8*acc1 + 10**(acc2+1)
            acc2 = acc2 + 1
            i += 1
        return acc1
    else:
        return 9






###########################################################################
# Functions for Problem 2
###########################################################################
#INPUT t = (a,b,c)
#RETURN return complex or real roots
def q(t):
    a,b,c = t
    dis = b**2 - (4*a*c)
    is_complex = dis < 0
    if is_complex:
        i1,i2 = math.sqrt(-dis)/(2*a), - math.sqrt(-dis)/(2*a)
        r = -b/(2*a)
        return complex(round(r,2),round(i1,2)), complex(round(r,2),round(i2,2))
    else:
        r1,r2 = round((-b + math.sqrt(dis))/(2*a),2), round((-b - math.sqrt(dis))/(2*a),2)
        if r1 < r2:
            return r1,r2
        else:
            return r2,r1
   
   
###########################################################################
# THIS SECTION IS FOR AUTOGRADER TESTING PURPOSES
# START

def test_q(submod, report_builder, case_builder, desc):

    # always use try/except
    funcname = 'q'
    report = report_builder(funcname)

    # make sure the submission module has the function we want to test
    if not hasattr(submod, funcname):
        report.not_found()
        return report.build()

    student_func = getattr(submod, funcname)
    inputs = [
        [[1,2,2]],
        [[54,12,12]],
        [[1,2,3]]
    ]

    for inp in inputs:
        testcase = case_builder(desc)
        try:
            testcase.input(inp)

            # save the result of calling the solution function
            solution = str(q(*inp))
            testcase.expected_output(solution)

            # save the result of calling student's function
            student_output = str(student_func(*inp))
            testcase.actual_output(student_output)

            report.add_case(testcase.build())
        except:
            testcase.exception()
            report.add_case(testcase.build())

    return report.build()

# THIS SECTION IS FOR AUTOGRADER TESTING PURPOSES
# END
###########################################################################



# ###########################################################################
# # Functions for Problem 3
# ###########################################################################

# def sd(p,q):
#     x = -(q[1]/q[0])
#     y = p[0]
#     ans = [p[0]]
#     for i in range(1,len(p)):
#         y = x*y + p[i]
#         ans.append(y)
#     if abs(q[0]) != 1.0:
#         ans = [(1/q[0])*a for a in ans[0:len(ans)-1]] + [ans[-1]]
#     return ans

#     #problem 3




###########################################################################
# Functions for Problem 3
###########################################################################
#INPUT tuple of quadratic (ax^2 + bx + c)
#RETURN tuple (m,n) cofficients for real solutions a(x+m)^2 + n = 0
#CONSTRAINT round 2 places
def c_s(coefficients):
    a,b,c = coefficients
    m = b/(2*a)
    n = c - (b**2/(4*a))
    return round(m,2),round(n,2)

#INPUT coefficients for quadratic ax^2 + bx + c 
#RETURN return real roots uses c_s
def q_(coefficients):
    m,n = c_s(coefficients)
    return round(-m - math.sqrt(-n),2),round(-m + math.sqrt(-n),2)


###########################################################################
# Functions for Problem 4
###########################################################################
#INPUT List of numbers
#RETURN Various means
def mean(lst):
    sum = 0
    for i in lst:
        sum += i
    return round(sum/len(lst),2)

def var(lst):
    mu = mean(lst)
    sum = 0
    for i in lst:
        sum += (i - mu)**2
    return round(sum/len(lst),2)

def std(lst):
    return round(math.sqrt(var(lst)),2)

def mean_centered(lst):
    mu = mean(lst)
    tlst = []
    for i in lst:
        tlst += [i-mu]
    return tlst



###########################################################################
# Functions for Problem 5
###########################################################################
#INPUT supply and demand coefficients
#RETURN solution of quadratic equations
def equi(s,d):
    sa,sb,sc = s
    da,db,dc = d
    a,b,c = (sa-da),(sb-db),(sc-dc)
    r1, r2 = q((a,b,c))
    if r1 > r2:
        return r1,r2
    else:
        return r2,r1



###########################################################################
# Functions for Problem 6
###########################################################################
#INPUT a nested list of people encoded as 0's and 1's. v0 and v1 are the respective lists respresenting the people pairs.
#   You'll be comparing the smallest degree of difference between each sublist representing each person.
#RETURN person pair with the smallest degree (smallest degree of difference between the person pair lists)
def inner_prod(v0,v1):
    sum = 0
    for i in range(len(v0)):
        sum += v0[i]*v1[i]
    return sum

def mag(v):
    e_v = inner_prod(v,v)
    return math.sqrt(e_v)

def angle(v0,v1):
    v = inner_prod(v0,v1)/(mag(v0)*mag(v1))
    return round(math.acos(v)*(180/math.pi),2)

def match(people):
    pairs = []
    for i in range(0,len(people)):
        for j in range(i+1,len(people)):
            p1,p2 = people[i],people[j]
            pairs += [[p1,p2,angle(p1,p2)]]
    return pairs

def best_match(scores):
    p0,p1,v = scores[0]
    for i,j,n_v in scores:
        if n_v < v:
            p0,p1,v = i,j,n_v
    return p0,p1,v 



###########################################################################
# Functions for Problem 7
###########################################################################
def determinant(matrix):
    (a,b),(c,d) = matrix
    return a*d - c*b

def solve(eq1,eq2):
    a,b,c,d,e,f = *eq1,*eq2
    x = [[c,f],[b,e]]
    y = [[a,d],[c,f]]
    den = [[a,d],[b,e]]
    x_ = round(determinant(x)/determinant(den),2)
    y_ = round(determinant(y)/determinant(den),2)
    return x_,y_

def f_1(x):
    return (1/4)*(-2*x + 11)

def f_2(x):
    return (1/3)*(5*x + 5)




###########################################################################
# Functions for Problem 8 City Block Distance
###########################################################################
#input two lists of points
#output the shared points using a single list comprehension
def intersection(x,y):
    return [i for i in x if i in y]


#input two points
#city block distance
def block_distance(p0,p1):
    x0,y0,x1,y1 = *p0,*p1
    return abs(x0-x1) + abs(y0-y1)


#input the center point and city block distance bd
#output list of points less than equal distance to center
def get_points(center,bd):
    x_,y_ = center
    tmp = []
    for i in range(-(bd), bd+1):
        for j in range(-(bd), bd+1):
            tmp = tmp + [(i+x_,j+y_)] if block_distance(center,(i+x_,j+y_)) <= bd else tmp
    return tmp




###########################################################################
# Functions for Problem 9
###########################################################################
#INPUT list of numbers
#OUTPUT Boolean if geometric series
def is_geometric_sequence(lst):
    result = len(lst) > 2
    if result:
        i = 0
        while i < len(lst) - 2 and (lambda i: (lst[i+1]/lst[i] == lst[i+2]/lst[i+1]))(i):
            i += 1
        result = i == len(lst) - 2
        
    return result


###########################################################################
# Functions for Problem 10
###########################################################################
#INPUT portfolio of stock price, shares, market
#OUTPUT current total value
def value(portfolio,market):
    sum = (0,0)
    for s,d in portfolio['stock'].items():
        price,shares = d
        sum = sum[0] + price*shares, sum[1] + market[s]*shares
    return 100*round((sum[1] - sum[0])/sum[0],2)



###########################################################################
# Functions for Problem 11
###########################################################################
#INPUT a (possibly empty) list of numbers
#OUTPUT raise exception or smooth
def smooth(lst):
    tmp = "AveError: list empty"
    if len(lst) > 1:
        tmp = []
        for i in range(1,len(lst)):
            ave = round((lst[i-1] + lst[i])/2,2)
            tmp.append(ave)
    elif len(lst) == 1:
        tmp = lst 
    return tmp 

 
###########################################################################
# problem 12
###########################################################################
#input secret code and all possible values
#output the string equal to the code
#must be done recursively
def break_code(secret_code, combinations):
    if secret_code:
        for i in combinations:
            if secret_code[0] == i:
                return i + break_code(secret_code[1:], combinations)
    else:
        return ""


#Do not change this code
#generates a secret code from a combination of values    
def m(useless_parameter=0):
    rn.seed(useless_parameter+1)
    combinations = "".join([chr(i) for i in range(ord('0'),ord('0') + rn.randint(5,35))])
    secret_code = ""
    for _ in range(rn.randint(4,8 + rn.randint(0,20))):
        secret_code += rn.choice(combinations)
    
    return secret_code, break_code(secret_code, combinations)



if __name__ == "__main__":
    """
    If you want to do some of your own testing in this file, 
    please put any print statements you want to try in 
    this if statement.

    You **do not** have to put anything here
    """

    # #problem 1

    # for i in range(5):
    #     print(f"n = {i}")
    #     print("c", c(i),ctr(i),cw(i))
    #     print("p", p(i), ptr(i))
    #     print("h", h(i), hmemo(i), hw(i), htr(i))
    #     print('d', d(i), dtr(i))

    # #problem 2
    # print(q((3,4,2)))
    # print(q((1,3,-4)))
    # print(q((1,-2,-4)))


    #problem 3 pairs should be identical
    # print(q((1,-4,-8)), q_((1,-4,-8)))
    # print(q((1,3,-4)),q_((1,3,-4)))
    # print(q((3,4,2))) #q_ won't work on complex roots
   
    
    # #problem 4
    #no example output 
    # lst = [1,3,3,2,9,10]

    # print(mean(lst))
    # print(var(lst))
    # print(std(lst))
    # print(mean(mean_centered(lst)))

    # #problem 5
    # s = (-.025,-.5,60)
    # d = (0.02,.6,20)
    # print(equi(s,d))
    
    #work this by hand
    # s = (5,7,-350)
    # d = (4,-8,1000)
    # print(equi(s,d))

    #problem 6
    # people0 = [[0,1,1],[1,0,0],[1,1,1]]
    # print(match(people0))
    # print(best_match(match(people0)))

    # people1 = [[0,1,1,0,0,0,1],
    #            [1,1,0,1,1,1,0],
    #            [1,0,1,1,0,1,1],
    #            [1,0,0,1,1,0,0],
    #            [1,1,1,0,0,1,0]]
    # print(best_match(match(people1)))
    # #output is ([1, 1, 0, 1, 1, 1, 0], [1, 0, 0, 1, 1, 0, 0], 39.23)

    # v0,v1 = (2,3,-1), (1,-3,5)
    # print(angle(v0,v1)) #122.83

    # v0,v1 = (3,4,-1),(2,-1,1)
    # print(angle(v0,v1)) #85.41

    # v0,v1 = (5,-1,1),(1,1,-1)
    # print(angle(v0,v1)) #70.53


    # #problem 7
    # print(determinant([[1,2],[2,3]])) #-1

    # eq1,eq2 = [1,1,3],[2,3,1]
    # print(solve(eq1,eq2))
    # eq1,eq2 = [[2,4,11],[-5,3,5]]
    # x_star,y_star = solve(eq1,eq2)
    # print(solve(eq1,eq2))
    # eq1,eq2 = [[3,-5,4],[7,4,25]]
    # print(solve(eq1,eq2))

    #Uncomment to see visualization (make sure to comment before submitting to the Autograder)
    # x = np.linspace(-2,6,100)
    # plt.plot(x,f_1(x),'r')
    # plt.plot(x,f_2(x),'b')
    # plt.plot(x_star,y_star,'go')
    # plt.show()

    
    #problem 8
    
    # A = ((0,-1),2)
    # B = ((0,1),1)
    # C = ((4,4),1)
    # p = get_points(*A)
    # q = get_points(*B)
    # r = intersection(p,q)
    # s = get_points(*C)
    # t = intersection(s,q)

    # for points in p,q,r,s:
    #     print(points)

    #uncomment to see visualization
    # color = 'rgbmy'

    # for i,pts in enumerate([p,q,r,s,t]):
    #     plt.plot([x for x,_ in pts],[y for _,y in pts],color[i] + 'o')

    # plt.gca().legend(("A: ((0,-1),2)", "B: ((0,1),1)", r"$\mathsf{A}\cap\mathsf{B}$","C: ((4,4),1)", r"$\mathsf{B}\cap\mathsf{C}$"))
    # plt.axis([-7, 7, -7, 7])
    # plt.grid()
    # plt.gca().set_aspect("equal")

    # plt.grid(True)
    # plt.title("City with square streets.")
    # plt.show()


    #problem 9
    # data = [[1,2,4,6],[2,4,8,16],[10,30,90,270,810,2430]]
    # for d in data:
    #     print(is_geometric_sequence(d))

    
    #problem 10
    # portfolios =  {'A':{'stock':{'x':(41.45,45),'y':(22.20,1000)}},'B':{'stock':{'x':(33.45,15),'y':(12.20,400)}}}
    # market = {'x':43.00, 'y':22.50}

    # for name, portfolio in portfolios.items():
    #     print(f"{name} {value(portfolio,market)}")

    
    #problem 11
    # data = [[], [1],[1,2],[1,2,2,3],[0,2,4,6,8]]
    # for d in data:
    #     print(smooth(d))  

    #problem 12
    #the parameter i isn't used
    # for i in range(5):
    #     print(m(i))
