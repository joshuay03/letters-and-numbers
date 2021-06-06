'''

In the Letters and Numbers (L&N) game,
One contestant chooses how many "small" and "large" numbers they would like 
to make up six randomly chosen numbers. Small numbers are between 
1 and 10 inclusive, and large numbers are 25, 50, 75, or 100. 
All large numbers will be different, 
so at most four large numbers may be chosen. 


How to represent a computation?

Let Q = [q0, q1, q2, q3, q4, q5] be the list of drawn numbers

The building blocks of the expression trees are
 the arithmetic operators  +,-,*
 the numbers  q0, q1, q2, q3, q4, q5

We can encode arithmetic expressions with Polish notation
    op arg1 arg2
where op is one of the operators  +,-,*

or with expression trees:
    (op, left_tree, right_tree)
    
Recursive definition of an Expression Tree:
 an expression tree is either a 
 - a scalar   or
 - a binary tree (op, left_tree, right_tree)
   where op is in  {+,-,*}  and  
   the two subtrees left_tree, right_tree are expressions trees.

When an expression tree is reduced to a scalar, we call it trivial.


Author: f.maire@qut.edu.au

Created on April 1 , 2021
    

This module contains functions to manipulate expression trees occuring in the
L&N game.

'''

import numpy as np
import random
import copy # for deepcopy
import collections
import time

SMALL_NUMBERS = tuple(range(1,11))
LARGE_NUMBERS = (25, 50, 75, 100)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    '''
    return [ (10404074, 'Joshua', 'Young'), (10240977, 'Jun', 'Chen') ]

# ----------------------------------------------------------------------------

def pick_numbers():
    '''
    Create a random list of numbers according to the L&N game rules.
    
    Returns
    -------
    Q : int list
        list of numbers drawn randomly for one round of the game
    '''
    LN = set(LARGE_NUMBERS)
    Q = []
    for i in range(6):
        x = random.choice(list(SMALL_NUMBERS)+list(LN))
        Q.append(x)
        if x in LN:
            LN.remove(x)
    return Q

# ----------------------------------------------------------------------------

def bottom_up_creator(Q):
    '''
    Create a random algebraic expression tree
    that respects the L&N rules.
    
    Warning: Q is shuffled during the process

    Parameters
    ----------
    Q : non empty list of available numbers
        

    Returns  T, U
    -------
    T : expression tree 
    U : values used in the tree
    '''
    n = random.randint(1,6) # number of values we are going to use
    
    random.shuffle(Q)
    # Q[:n]  # list of the numbers we should use
    U = Q[:n].copy()
    
    if n==1:
        # return [U[0], None, None], [U[0]] # T, U
        return U[0], [U[0]] # T, U
        
    F = [u for u in U]  # F is initially a forest of values
    # we start with at least two trees in the forest
    while len(F)>1:
        # pick two trees and connect then with an arithmetic operator
        random.shuffle(F)
        op = random.choice(['-','+','*'])
        T = [op,F[-2],F[-1]]  # combine the last two trees
        F[-2:] = [] # remove the last two trees from the forest
        # insert the new tree in the forest
        F.append(T)
    # assert len(F)==1
    return F[0], U

# ---------------------------------------------------------------------------- 

def display_tree(T, indent=0):
    '''
    Eval the algebraic expression represented by T
    
    Parameters
    ----------
    T : Expression Tree
    indent: indentation for the recursive call

    Returns None
    '''
    # if T is a scalar, then we return it directly
    if isinstance(T, int):
        print('|'*indent,T, sep='')
        return
    # T is non trivial
    root_item = T[0]
    print('|'*indent, root_item, sep='')
    display_tree(T[1], indent+1)
    print('|'*indent)
    display_tree(T[2], indent+1)

# ---------------------------------------------------------------------------- 

def eval_tree(T):
    '''
    Eval the algebraic expression represented by T
    
    Parameters
    ----------
    T : Expression Tree

    Returns
    -------
    value of the algebraic expression represented by the T
    '''
    # if T is a scalar, then we return it directly
    if isinstance(T, int):
        return T
    # T is non trivial
    root_item = T[0]
    # assert root_item in ('-','+','*')
    left_value = eval_tree(T[1])
    right_value = eval_tree(T[2])
    return eval( str(left_value) +root_item + str(right_value) )
    # return eval(root_item.join([str(left_value), str(right_value)]))

# ---------------------------------------------------------------------------- 

def expr_tree_2_polish_str(T):
    '''
    Convert the Expression Tree into Polish notation

    Parameters
    ----------
    T : expression tree

    Returns
    -------
    string in Polish notation represention the expression tree T
    '''
    if isinstance(T, int):
        return str(T)
    root_item = T[0]
    # assert root_item in ('-','+','*')
    left_str = expr_tree_2_polish_str(T[1])
    right_str = expr_tree_2_polish_str(T[2])
    return '[' + ','.join([root_item,left_str,right_str]) + ']'

# ----------------------------------------------------------------------------

def polish_str_2_expr_tree(pn_str):
    '''
    Convert a polish notation string of an expression tree
    into an expression tree T.

    Parameters
    ----------
    pn_str : string representing an L&N algebraic expression

    Returns
    -------
    T
    '''
    def find_match(i):
        '''
        Starting at position i where pn_str[i] == '['
        Return the index j of the matching ']'
        That is, pn_str[j] == ']' and the substring pn_str[i:j+1]
        is balanced
        '''
        stack = collections.deque()
        while i < (len(pn_str)):
            if pn_str[i] in '[]':
                if pn_str[i] == '[':
                    # Place opening bracket on top of stack
                    stack.append(pn_str[i])
                elif pn_str[i] == ']':
                    # If only one bracket left in the stack i has to be the index of its match
                    if len(stack) == 1:
                        return i
                    else:
                        # Remove the matching opening bracket
                        stack.pop()
            i += 1

     # .................................................................  

    # Insert the operator and place holders for the items and indices 1 and 2
    T = [pn_str[1], [], []]

    # If item at index 1 is a list
    if pn_str[4] == '[':
        left_p = 4
        right_p = find_match(left_p)
        left_al = polish_str_2_expr_tree(pn_str[left_p:right_p+1])
        T[1] = left_al

        # If item at index 2 is a list
        if pn_str[right_p+3] == '[':
            left_p = right_p+3
            right_p = find_match(left_p)
            right_al = polish_str_2_expr_tree(pn_str[left_p:right_p+1])
            T[2] = right_al
        # If item at index 2 is an integer
        else:
            right_al = ''
            for char in pn_str[right_p+3:]:
                if char != ',' and char != ']':
                    right_al += char
                else:
                    break
            T[2] = int(right_al)

    # If item at index 1 is an integer
    else:
        C = 0
        left_al = ''
        for char in pn_str[4:]:
            if char != ',':
                left_al += char
                C+=1
            else:
                break
        T[1] = int(left_al)

        # If item at index 2 is a list
        if pn_str[4+C+2] == '[':
            left_p = 4+C+2
            right_p = find_match(left_p)
            right_al = polish_str_2_expr_tree(pn_str[left_p:right_p+1])
            T[2] = right_al
        # If item at index 2 is an integer
        else:
            right_al = ''
            for char in pn_str[4+C+2:]:
                if char != ',' and char != ']':
                    right_al += char
                else:
                    break
            T[2] = int(right_al)
    
    return T

# ----------------------------------------------------------------------------

def op_address_list(T, prefix = None):
    '''
    Return the address list L of the internal nodes of the expresssion tree T
    
    If T is a scalar, then L = []

    Note that the function 'decompose' is more general.

    Parameters
    ----------
    T : expression tree
    prefix: prefix to prepend to the addresses returned in L

    Returns
    -------
    L
    '''
    if isinstance(T, int):
        return []
    
    if prefix is None:
        prefix = []
        
    L = [prefix.copy()+[0]] # first adddress is the op of the root of T
    left_al = op_address_list(T[1], prefix.copy()+[1])
    L.extend(left_al)
    right_al = op_address_list(T[2], prefix.copy()+[2])
    L.extend(right_al)
    
    return L

def num_address_list(T, prefix = None):
    '''
    Return the address list L of the internal nodes of the expresssion tree T
    
    If T is a scalar, then L = []

    Note that the function 'decompose' is more general.

    Parameters
    ----------
    T : expression tree
    prefix: prefix to prepend to the addresses returned in L

    Returns
    -------
    L
    '''
    if isinstance(T, int):
        return [prefix.copy()]
    
    if prefix is None:
        prefix = []
    
    L = []
    left_al = num_address_list(T[1], prefix.copy()+[1])
    L.extend(left_al)
    right_al = num_address_list(T[2], prefix.copy()+[2])
    L.extend(right_al)
    
    return L

# ----------------------------------------------------------------------------

def decompose(T, prefix = None):
    '''
    Compute
        Aop : address list of the operators
        Lop : list of the operators
        Anum : address of the numbers
        Lnum : list of the numbers
    
    For example, if 
    
    T =  ['-', ['+', ['-', 75, ['-', 10, 3]], ['-', 100, 50]], 3]
    
    then, 
    
     Aop is  [[0], [1, 0], [1, 1, 0], [1, 1, 2, 0], [1, 2, 0]] 
    
     Lop is ['-', '+', '-', '-', '-'] 
    
     Anum is [[1, 1, 1], [1, 1, 2, 1], [1, 1, 2, 2], [1, 2, 1], [1, 2, 2], [2]] 
    
     Lnum is [75, 10, 3, 100, 50, 3]    
        
    
    Parameters
    ----------
    T : expression tree 
    
    prefix : address to preprend 

    Returns
    -------
    Aop, Lop, Anum, Lnum
    '''
    if prefix is None:
        prefix = []

    if isinstance(T, int):
        Aop = []
        Lop = [] 
        Anum = [prefix]
        Lnum = []
        return Aop, Lop, Anum, Lnum
    
    assert isinstance(T, list)

    Aop = op_address_list(T)
    Lop = [get_item(T, x) for x in Aop]
    Anum = num_address_list(T)
    Lnum = [get_item(T, x) for x in Anum]

    return Aop, Lop, Anum, Lnum

# ----------------------------------------------------------------------------

def get_item(T, a):
    '''
    Get the item at address a in the expression tree T

    Parameters
    ----------
    T : expression tree
    a : valid address of an item in the tree

    Returns
    -------
    the item at address a
    '''
    if len(a)==0:
        return T
    # else
    return get_item(T[a[0]], a[1:])

# ----------------------------------------------------------------------------

def replace_subtree(T, a, S):
    '''
    Replace the subtree at address a
    with the subtree S in the expression tree T
    
    The address a is a sequence of integers in {0,1,2}.
    
    If a == [] , then we return S
    If a == [1], we replace the left subtree of T with S
    If a == [2], we replace the right subtree of T with S

    Returns
    ------- 
    The modified tree

    Warning: the original tree T is modified. 
             Use copy.deepcopy()  if you want to preserve the original tree.
    '''
    
    # base case, address empty
    if len(a)==0:
        return S
    
    # recursive case
    T[a[0]] = replace_subtree(T[a[0]], a[1:], S)
    return T

# ----------------------------------------------------------------------------

def mutate_num(T, Q):
    '''
    Mutate one of the numbers of the expression tree T
    
    Parameters
    ----------
    T : expression tree
    Q : list of numbers initially available in the game

    Returns
    -------
    A mutated copy of T
    '''
    Aop, Lop, Anum, Lnum = decompose(T)    
    mutant_T = copy.deepcopy(T)
    random_address_num = random.choice(Anum) # Pick a random address in a tree
    counter_Q = collections.Counter(Q) # Some small numbers can be repeated

    for number in Lnum:
        if number in Q:
            counter_Q.subtract([number])
            # Some small numbers can be repeated
            if (counter_Q[number] <= 0):
                counter_Q[number] = 0
    # When all Q values don't exist in T
    if (sum(counter_Q.values()) != 0):
        mutant_num = random.choice(list(counter_Q.keys()))
        while(counter_Q[mutant_num] == 0):
            mutant_num = random.choice(list(counter_Q.keys()))

        mutant_T = replace_subtree(mutant_T, random_address_num, mutant_num)
        return mutant_T
    else:            
        return mutant_T
    

# ----------------------------------------------------------------------------

def mutate_op(T):
    '''
    Mutate an operator of the expression tree T
    If T is a scalar, return T

    Parameters
    ----------
    T : non trivial expression tree

    Returns
    -------
    A mutated copy of T
    '''
    if isinstance(T, int):
        return T
    
    La = op_address_list(T)
    a = random.choice(La) # random address of an op in T
    op_c = get_item(T, a) # the char of the op

    op_list = ["+", "-", "*"] # List of possible operators
    op_list.remove(op_c) #remove existing operators

    # mutant_c : a different op
    mutant_c = random.choice(op_list)

    #initialize a mutated copy of T
    mutant_T = copy.deepcopy(T)
    mutant_T = replace_subtree(mutant_T,a,mutant_c)

    return mutant_T

# ----------------------------------------------------------------------------

def cross_over(P1, P2, Q):    
    '''
    Perform crossover on two non trivial parents
    
    Parameters
    ----------
    P1 : parent 1, non trivial expression tree  (root is an op)
    P2 : parent 2, non trivial expression tree  (root is an op)
        DESCRIPTION
        
    Q : list of the available numbers
        Q may contain repeated small numbers    
        

    Returns
    -------
    C1, C2 : two children obtained by crossover
    '''
    def get_num_ind(aop, Anum):
        '''
        Return the indices [a,b) of the range of numbers
        in Anum and Lum that are in the sub-tree 
        rooted at address aop

        Parameters
        ----------
        aop : address of an operator (considered as the root of a subtree).
              The address aop is an element of Aop
        Anum : the list of addresses of the numbers

        Returns
        -------
        a, b : endpoints of the semi-open interval
        '''
        d = len(aop)-1  # depth of the operator. 
                        # Root of the expression tree is a depth 0
        # K: list of the indices of the numbers in the subtrees
        # These numbers must have the same address prefix as aop
        p = aop[:d] # prefix common to the elements of the subtrees
        K = [k for k in range(len(Anum)) if Anum[k][:d]==p ]
        return K[0], K[-1]+1
        # .........................................................
        
    Aop_1, Lop_1, Anum_1, Lnum_1 = decompose(P1)
    Aop_2, Lop_2, Anum_2, Lnum_2 = decompose(P2)

    C1 = copy.deepcopy(P1)
    C2 = copy.deepcopy(P2)
    
    i1 = np.random.randint(0,len(Lop_1)) # pick a subtree in C1 by selecting the index
                                         # of an op
    i2 = np.random.randint(0,len(Lop_2)) # Select a subtree in C2 in a similar way
 
    # i1, i2 = 4, 0 # DEBUG    
 
    # Try to swap in C1 and C2 the sub-trees S1 and S2 
    # at addresses Lop_1[i1] and Lop_2[i2].
    # That's our crossover operation!
    
    # Compute some auxiliary number lists
    
    # Endpoints of the intervals of the subtrees
    a1, b1 = get_num_ind(Aop_1[i1], Anum_1)     # indices of the numbers in S1 
                                                # wrt C1 number list Lnum_1
    a2, b2 = get_num_ind(Aop_2[i2], Anum_2)   # same for S2 wrt C2
    
    # Lnum_1[a1:b1] is the list of numbers in S1
    # Lnum_2[a2:b2] is the list of numbers in S2
    
    # numbers is C1 not used in S1
    nums_C1mS1 = Lnum_1[:a1]+Lnum_1[b1:]
    # numbers is C2-S2
    nums_C2mS2 = Lnum_2[:a2]+Lnum_2[b2:]
    
    # S2 is a fine replacement of S1 in C1
    # if nums_S2 + nums_C1mS1 is contained in Q
    # if not we can bottom up a subtree with  Q-nums_C1mS1

    counter_Q = collections.Counter(Q) # some small numbers can be repeated
    
    d1 = len(Aop_1[i1])-1
    aS1 = Aop_1[i1][:d1] # address of the subtree S1 
    S1 = get_item(C1, aS1)

    d2 = len(Aop_2[i2])-1
    aS2 = Aop_2[i2][:d2] # address of the subtree S2
    S2 = get_item(C2, aS2)

    # print(' DEBUG -------- S1 and S2 ----------') # DEBUG
    # print(S1)
    # print(S2)


    # count the numbers (their occurences) in the candidate child C1
    counter_1 = collections.Counter(Lnum_2[a2:b2]+nums_C1mS1)
    
    # Test whether child C1 is ok
    if all(counter_Q[v]>=counter_1[v] for v in counter_Q):
        # candidate is fine!  :-)
        C1 = replace_subtree(C1, aS1, S2)
    else:
        available_nums = counter_Q.copy()
        available_nums.subtract(
            collections.Counter(nums_C1mS1)
            )
        R1, _ = bottom_up_creator(list(available_nums.elements()))
        C1 = replace_subtree(C1, aS1, R1)
        
    # count the numbers (their occurences) in the candidate child C2
    counter_2 = collections.Counter(Lnum_1[a1:b1]+nums_C2mS2)

    # Test whether child C2 is ok
    if all(counter_Q[v]>=counter_2[v] for v in counter_Q):
        # candidate is fine!  :-)
        C2 = replace_subtree(C2, aS2, S1)
    else:
        available_nums = counter_Q.copy()
        available_nums.subtract(
            collections.Counter(nums_C2mS2)
            )
        R2, _ = bottom_up_creator(list(available_nums.elements()))
        C2 = replace_subtree(C2, aS2, R2)

    return C1, C2

default_GA_params = {
    'max_num_iteration': 50,
    'population_size':100,
    'mutation_probability':0.1,
    'elit_ratio': 0.05,
    'parents_portion': 0.3}


def evolve_pop(Q, target, **ga_params):
    '''
    Evolve a population of expression trees for the game
    Letters and Numbers given a target value and a set of numbers.
    

    Parameters
    ----------
    Q : list of integers
        Integers that were drawn by the game host
    
    target: integer
           target value of the game
        
    params : dictionary, optional
        The default is GA_params.
        Dictionary of parameters for the genetic algorithm

    Returns
    -------
    v, T: the best expression tree found and its value

    '''
    params = default_GA_params.copy()
    params.update(ga_params)
    
    # print('GA Parameters ', params)
    
    mutation_probability = params['mutation_probability']
    pop_size = params['population_size']
    
    # ------------- Initialize Population ------------------------
    
    pop = [] # list of pairs (cost, individuals)
    
    for _ in range(pop_size):
        T, _ = bottom_up_creator(Q)
        cost = abs(target-eval_tree(T))
        pop.append((cost,T))
    
    # Sort the initial population
    # print(pop) # debug
    pop.sort(key=lambda x:x[0])
    
    # Report
    # print('\n'+'-'*40+'\n')
    # print("The best individual of the initial population has a cost of {}".format(pop[0][0]))
    # print("The best individual is \n")
    # display_tree(pop[0][1])
    # print('\n')
    # ------------- Loop on generations ------------------------
    
    # Rank of last individual in the current population
    # allowed to breed.
    rank_parent = int(params['parents_portion'] * 
                                      params['population_size'])
    
    # Rank of the last elite individual. The elite is copied unchanged 
    # into the next generation.
    rank_elite = max(1, int(params['elit_ratio'] *
                                      params['population_size']))
    
    for g in range(params['max_num_iteration']):
        # Generate children
        children = []
        while len(children) < pop_size:
            # pick two parents
            (_, P1), (_, P2) = random.sample(pop[:rank_parent], 2)
            # skip cases where one of the parents is trivial (a number)
            if isinstance(P1, list) and isinstance(P2, list):
                C1, C2 = cross_over(P1, P2, Q)
            else:
                # if one of the parents is trivial, just compute mutants
                C1 = mutate_num(P1,Q)
                C2 = mutate_num(P2,Q)
            # Compute the costs of the children
            cost_1 =  abs(target-eval_tree(C1))
            cost_2 =  abs(target-eval_tree(C2))
            children.extend([ (cost_1,C1), (cost_2,C2) ])
             
        new_pop = pop[rank_elite:]+children 
        
        # Mutate some individuals (keep aside the elite for now)
        # Pick randomly the indices of the mutants
        mutant_indices = random.sample(range(len(new_pop)), 
                                       int(mutation_probability*pop_size))      
        # i: index of a mutant in new_pop
        for i in mutant_indices:
            # Choose a mutation by flipping a coin
            Ti = new_pop[i][1]  #  new_pop[i][0]  is the cost of Ti
            # Flip a coin to decide whether to mutate an op or a number
            # If Ti is trivial, we can only use mutate_num
            if isinstance(Ti, int) or random.choice((False, True)): 
                Mi = mutate_num(Ti, Q)
            else:
                Mi = mutate_op(Ti)
            # update the mutated entry
            new_pop[i] = (abs(target-eval_tree(Mi)), Mi)
                
        # add without any chance of mutation the elite
        new_pop.extend(pop[:rank_elite])
        
        # sort
        new_pop.sort(key=lambda x:x[0])
        
        # keep only pop_size individuals
        pop = new_pop[:pop_size]
        
        # Report some stats
        # print(f"\nAfter {g+1} generations, the best individual has a cost of {pop[0][0]}\n")
        
        if pop[0][0] == 0:
            # found a solution!
            break

    # return best found
    res = list(pop[0])
    res.append(g)
    return tuple(res)

def find_max_pop(Q, target):
    pops = []
    for pop_size in range(500, 25000, 500):
        print(f"Evaluating population size of {pop_size}")

        start_time = time.time()
        v, T, g = evolve_pop(Q, target, 
                              max_num_iteration = 200,
                              population_size = pop_size,
                              parents_portion = 0.3)
        end_time = time.time()

        # Break if the evaluation took more than 2 seconds regardless of the result
        if end_time - start_time > 2:
            break
        
        # Save evaluation if under 2 seconds to grab the last one later
        pops.append(pop_size)

    print(f"\nMaximum population evaluated in less than 2 seconds: {pops[len(pops)-1]}\n")
    return pops[len(pops)-1]

def find_max_gens(max_pop):
    pops_gens = {}
    for i in range(20):
        pop_test = random.choice(range(5, max_pop))
        print(f"Searching for maximum generation using population of {pop_test}")

        v, T, g = evolve_pop(Q, target, 
                           max_num_iteration = 9999,
                           population_size = pop_test,
                           parents_portion = 0.3)

        # Generation is one more than the index iterated
        g += 1

        print(f"    Maximum generation: {g}")

        pops_gens[pop_test] = g
    
    return pops_gens

def evaluate_pops_gens(pops_gens):
    targets_Qs = {}
    # Create 30 problems, target and numbers
    for i in range(30):
        Q = pick_numbers()
        Q.sort()
        target = np.random.randint(1,1000)
        targets_Qs[target] = Q

    results = []
    index = 0
    # Evaluate each combination of population and max generation on all 30 problems
    for pop, gen in pops_gens.items():
        print(f"{index+1}. Evaluating max population of {pop} and max generation of {gen}")
        results.append(0)
        for target, Q in targets_Qs.items():
            v, T, g = evolve_pop(Q, target, 
                       max_num_iteration = gen,
                       population_size = pop,
                       parents_portion = 0.3)
            if v==0:
                #target found, increment the success count
                results[index] += 1
        results[index] = (results[index]/30)*100        #calculate the success rate as a percentage
        print(f"    Pair success rate: {results[index]}%")
        index += 1

# Q = pick_numbers()
# target = np.random.randint(1,10000)

# Q = [100, 50, 3, 3, 10, 75]
# target = 322

# Q = [25,10,2,9,8,7]
# target = 449

# Q = [50,75,9,10,2,2]
# target = 533

# Q = [100,25,7,5,3,1]
# target = 728

Q = [100,25,7,5,3,1]
target = 728

Q.sort()

print(f"\nUsing target {target}, numbers {Q} and population step size 500 to find max_pop and max_gen\n")

print("-"*10 + "MAX_POP" + "-"*10)
max_pop = find_max_pop(Q, target)

print("-"*10 + "MAX_GEN" + "-"*10)
pops_gens = find_max_gens(max_pop)

print("\n" + "-"*10 + "EVALUATE" + "-"*10)
evaluate_pops_gens(pops_gens)