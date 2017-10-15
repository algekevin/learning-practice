import math
import datetime
from datetime import date
import time
import random
import fibo
from functools import reduce

class Solver:
    def demo(self):
        while True:
            a = int(eval(input("a ")))
            b = int(eval(input("b ")))
            c = int(eval(input("c ")))
            d = b ** 2 - 4 * a * c
            if d >= 0:
                disc = math.sqrt(d)
                root1 = (-b + disc) / (2 * a)
                root2 = (-b - disc) / (2 * a)
                print((root1, root2))
            else:
                print('no real roots')
                return

    #Note: no difference between using ' and " ?

    def arithmetic(self):
        a = int(eval(input('a ')))
        b = int(eval(input('b ')))
        if b % a == 0:
            print('b is divisible by a')
        elif b + 1 == 10:
            print('Increment in b produces 10')
        else:
            print('b is not divisible by a and b + 1 is not equal to 10')

    def splitting(self):
        line = "Geek1 \nGeek2 \nGeek3"
        print(line.split())
        print(line.split(' ', 1))

    def formatting(self):
        # Python program to demonstrate the use of formatting using %

        # Initialize variable as a string
        variable = '15'
        string = "Variable as string = %s" %(variable)
        print(string)

        # Convert the variable to integer
        # And perform check other formatting options

        variable = int(variable)  # Without this the below statement
                                  # will give error.
        string = "Variable as integer =  %d" %(variable)
        print(string)
        print("Variable as float  =  %f" %(variable))
        print("Variable as hexadecimal = %x" %(variable))
        print("Variable as octal  = %o" %(variable))
        print("Variable as raw data = %r" %(variable))

    def templates(self):
        # A Python program to demonstrate working of string template
        from string import Template

        # List Student stores the name and marks of three students
        Student = [('Ram', 90), ('Ankit', 78), ('Bob', 92)]

        # We are creating a basic structure to print the name and
        # marks of the students.
        t = Template('Hi $name, you have got $marks marks')

        for i in Student:
            print((t.substitute(name=i[0], marks=i[1])))
        return

    def listStuff(self):
        # Python program to demonstrate list comprehension in Python

        # below list contains square of all odd numbers from
        # range 1 to 10
        odd_square = [x ** 2 for x in range(1, 11) if x % 2 == 1]
        print(odd_square)

        # for understanding, above generation is same as,
        odd_square = []
        for x in range(1, 11):
            if x % 2 == 1:
                odd_square.append(x ** 2)
        print(odd_square)

        # below list contains power of 2 from 1 to 8
        power_of_2 = [2 ** x for x in range(1, 9)]
        print(power_of_2)

        # below list contains prime and non-prime in range 1 to 50
        noprimes = [j for i in range(2, 8) for j in range(i * 2, 50, i)]
        primes = [x for x in range(2, 50) if x not in noprimes]
        print(primes)

        # list for lowering the characters
        print([x.lower() for x in ["A", "B", "C"]])

        # list which extracts number
        string = "my phone number is : 11122 !!"

        print("\nExtracted digits")
        numbers = [x for x in string if x.isdigit()]
        print(numbers)

        # A list of list for multiplication table
        a = 5
        table = [[a, b, a * b] for b in range(1, 11)]

        print("\nMultiplication Table")
        for i in table:
            print(i)

        #Recall, [start:stop:steps]

        # Let us first create a list to demonstrate slicing
        # lst contains all number from 1 to 10
        lst = list(range(1, 11))
        print(lst)

        #  below list has numbers from 2 to 5
        lst1_5 = lst[1: 5]
        print(lst1_5)

        #  below list has numbers from 6 to 8
        lst5_8 = lst[5: 8]
        print(lst5_8)

        #  below list has numbers from 2 to 10
        lst1_ = lst[1:]
        print(lst1_)

        #  below list has numbers from 1 to 5
        lst_5 = lst[: 5]
        print(lst_5)

        #  below list has numbers from 2 to 8 in step 2
        lst1_8_2 = lst[1: 8: 2]
        print(lst1_8_2)

        #  below list has numbers from 10 to 1
        lst_rev = lst[:: -1]
        print(lst_rev)

        #  below list has numbers from 10 to 6 in step 2
        lst_rev_9_5_2 = lst[9: 4: -2]
        print(lst_rev_9_5_2)

        #  filtering odd numbers
        lst = [x for x in range(1, 20) if x % 2 == 1]
        print(lst)

        #  filtering odd square which are divisble by 5
        lst = [x for x in [x ** 2 for x in range(1, 11) if x % 2 == 1] if x % 5 == 0]
        print(lst)

        #   filtering negative numbers
        lst = list(filter((lambda x: x < 0), list(range(-5, 5))))
        print(lst)

        #  implementing max() function, using
        print(reduce(lambda a, b: a if (a > b) else b, [7, 12, 45, 100, 15]))

    def setIntro(self):
        # Avantage of using a set as opposed to a list is that it has a highly optimized method for checking
        # whether a specific element is contained in the set. This is based on a HASH TABLE.

        # Frozen sets are immutable objects that only support methods and operators that produce a
        # result without affecting the frozen set of sets to which they are applied.

        # Python program to demonstrate differences
        # between normal and frozen set

        # Same as {"a", "b","c"}
        normal_set = set(["a", "b", "c"])

        # Adding an element to normal set is fine
        normal_set.add("d")

        print("Normal Set")
        print(normal_set)

        # A frozen set
        frozen_set = frozenset(["e", "f", "g"])

        print("Frozen Set")
        print(frozen_set)

        # Uncommenting the below line would cause error as
        # we are trying to add element to a frozen set
        # frozen_set.add("h")

        ##### add(x) Method: Adds the item x to set if it is not already in the set
        people = {"Jay", "Idrish", "Archil"}
        print(("People before mod: ", people))

        people.add("Daxit")
        print(("People aftermod: ", people))

        ##### union(s) Method: Returns a union of two sets. Using the | operator between
        ##### two sets is the same as writing set1.union(set2)

        vampires = {"Karan", "Arjun"}
        population = people.union(vampires)     # Can also simply do population = people|vampires
        print(("Union of people and vampires: ", population))

        ##### intersect(s) Method: Returns an intersection of two sets. The & operator
        ##### can also be used in this case.

        victims = people.intersection(vampires)
        print(("Intersection between people and vampires", victims))

        ##### difference(s) Method: Returns a set containing all the elements of invoking
        ##### set but not of the second set. We can use - operator here.

        safe = people.difference(vampires)      # Can also simply do safe = people - vampires
        print(("Taking set people and removing vampires: ", safe))

        ##### clear() Method: Empties the whole set.
        people.clear()
        print(("New people set after clearing: ", people))

        # There are two major pitfalls in Python sets:
        # 1. The set doesn't maintain elements in any particular order.
        # 2. Only instances of immutable types can be added to a Python set.

        # Sets and frozen sets support the following operators:
        # key in s                containment check
        # key not in s            non-containment check
        # s1 == s2                s1 is equivalent to s2
        # s1 != s2                s1 is not equivalent to s2
        # s1 <= s2                s1 is a subset of s2
        # s1 <  s2                s1 is a PROPER subset of s2
        # s1 >= s2                s1 is superset of s2
        # s1 >  s2                s1 is proper superset of s2
        # s1 |  s2                union of s1 and s2
        # s1 &  s2                intersection of s1 and s2
        # s1 -  s2                the set of elements in s1 but not s2
        # s1 ^  s2                the set of elements in precisely one of s1 or s2

        # Python program to demonstrate working of
        # Sets in Python

        # Creating two sets
        set1 = set()
        set2 = set()

        # Adding elements to set1
        for i in range(1, 6):
            set1.add(i)

        # Adding elements to set2
        for i in range(3, 8):
            set2.add(i)

        print(("Set1 = ", set1))
        print(("Set2 = ", set2))
        print("\n")

        # Union of set1 and set2
        set3 = set1 | set2  # set1.union(set2)
        print(("Union of Set1 & Set2: Set3 = ", set3))

        # Intersection of set1 and set2
        set4 = set1 & set2  # set1.intersection(set2)
        print(("Intersection of Set1 & Set2: Set4 = ", set4))
        print("\n")

        # Checking relation between set3 and set4
        if set3 > set4:  # set3.issuperset(set4)
            print("Set3 is superset of Set4")
        elif set3 < set4:  # set3.issubset(set4)
            print("Set3 is subset of Set4")
        else:  # set3 == set4
            print("Set3 is same as Set4")

        # displaying relation between set4 and set3
        if set4 < set3:  # set4.issubset(set3)
            print("Set4 is subset of Set3")
            print("\n")

        # difference between set3 and set4
        set5 = set3 - set4
        print(("Elements in Set3 and not in Set4: Set5 = ", set5))
        print("\n")

        # checkv if set4 and set5 are disjoint sets
        if set4.isdisjoint(set5):
            print("Set4 and Set5 have nothing in common\n")

        # Removing all the values of set5
        set5.clear()

        print("After applying clear on sets Set5: ")
        print(("Set5 = ", set5))

    def operators(self):
        # A simple Python program to show loop, unlikes most language it doesn't use ++

        for i in range(0,5):
            print(i)

        # Note: / for integers in Python works as floor division. If one is a float, it returns a float.
        print("5/2: ", 5/2)
        print("-5/2: ", (-5/2))
        print("5.0/2: ", (5.0/2))
        print("-5.0/2: ", (-5.0/2))

        # The real floor division operator is //
        # It returns the floor value for both integer and floating point arguments.
        print("floor(5.0/2): ", 5//2)
        print('floor(-5.0/2): ', (-5//2))

    def dictionaryStuff(self):
        # In Python, dictionary is similar to hash or maps in other languages. It consists of key value pairs.
        # The value can be access by unique key in the dictionary.

        # Create a new dictionary
        d = dict()                  # Or you can do d = {}

        # Add a key - value pairs to dictionary
        d['xyz'] = 123
        d['abc'] = 345

        # Print the whole dictionary
        print(d)

        # Print only the keys
        print(list(d.keys()))

        # Print only the values
        print(list(d.values()))

        # Iterate over dictionary
        for i in d:
            print("%s %d" %(i,d[i]))

        # Another method of iteration
        for index, value in enumerate(d):
            print(index, value, d[value])

        # Check if key exist
        print('xyz' in d)

        # Delete the key-value pair
        del d['xyz']

        # Check again
        print('xyz' in d)

    def breaks(self):
        # Break - Takes you out of the current loop
        # Continue - Ends the current iteration in the loop and moves to the next iteration.
        # Pass - Does nothing. It can be used when a statement is required syntactically but
        #     the program requires no action. It is commonly used for creating minimal classes.

        # Driver program to test below functions:

        # Array to be used for the below functions
        arr = [1, 3, 4, 5, 6, 7]

        print("Break method output")
        self.breakTest(arr)

        print("Continue method output")
        self.continueTest(arr)

        self.passTest(arr)

    # Function to illustrate break in loop
    def breakTest(self,arr):
        for i in arr:
            if i == 5:
                break
            print(i, end=' ')
        #For new line
        print()

    # Function to illustrate continue in loop
    def continueTest(self, arr):
        for i in arr:
            if i == 5:
                continue
            print(i, end=' ')
        #For new line
        print()


    #Function to illustrate pass
    def passTest(self, arr):
        pass

    def mfl(self):
        # map - The map() function applies a function to every member of iterable and returns the result.
        # If there are multiple arguments, map() returns a list consisting of tuples containing the
        #     corresponding items from all iterables.

        # filter - It takes a function returning True or False and applies it to a sequence,
        #     returning a list of only those members of the sequence for which the function returned True.

        # lambda - Python provides the ability to create a simple(no statements allowed internally) anonymous
        #    inline function called lambda function. Using lambda and map you can have two for loops in one line.

        # Now, a Python program to test map, filter and lambda.

        #Driver to test cube

        #Program for working of map
        print("MAP EXAMPLES")
        squares = list(map(self.square, list(range(10))))
        print(squares)

        # Now for lambda
        print("LAMBDA EXAMPLES")

        # First parentheses contains a lambda form, that is a squaring function and
        #     second parentheses represents calling lambda.
        print((lambda x: x**2)(5))

        # Make function of two arguments that return their product
        print((lambda x, y: x*y)(3, 4))

        # Now for filter
        print("FILTER EXAMPLE")
        special_squares = [x for x in squares if x > 9 and x < 60]
        print(special_squares)

        # For more clarity about map, filter and lamda, look at the below example:

        # Code without using map, filter and lambda

        # Find the number which are odd in the list
        # and multiply them by 5 and create a new list

        # Declare a new list
        x = [2, 3, 4, 5, 6]

        # Empty list for answer
        y = []

        # Perform the operations and print the answer
        for v in x:
            if v % 2:
                y += [v * 5]
        print(y)

        # Now, the same operation can be performed in two lines using map, filter and lambda:

        # Above code with map, filter and lambda

        # Declare a list
        x = [2, 3, 4, 5, 6]

        # Perform the same operation as  in above post
        y = [v * 5 for v in [u for u in x if u % 2]]
        print(y)

    # Function to test map
    def square(self,x):
        return x**2


    def exceptionHandling(self):
        # Exception is the base class for all the exceptions in Python.
        # Let us try to access the array element whose index is out of bounds and handle the exception.

        # Python program to handle simple runtime error
        a = [1, 2, 3]
        try:
            print("Second element = %d" %(a[1]))

            # Throws error since there are only 3 elements in the array.
            print("Fourth element = %d" %(a[3]))

        except IndexError:
            print("An error occurred.")

        # A try statement can have more than one except clause, to specify handlers for different exceptions.
        # Note that at most one handler will be executed.

        # Program to handle multiple errors with one except statement

        try:
            a = 3
            if a < 4:
                # Throws ZeroDivisionError for a = 3
                b = a/(a-3)
            # Throws NameError if a >= 4
            print("Value of b = ", b)

        # Note that parentheses () are necessary here for multiple exceptions.
        except(ZeroDivisionError, NameError):
            print("\nError occured and handled")
        print()
        # Driver program to test AbyB
        self.AbyB(2.0,3.0)
        self.AbyB(3.0,3.0)

        # The raises statement allows the programmer to force a specific exception to occur. The sole argument
        #    in raise indicates the exception to be raised. This must be either an exception instance or an
        #    exception class(a class that derives from Exception)

        try:
            raise NameError("Hi There")
        except NameError:
            print("An exception")
        #    raise       # To determine whether the exception was raised or not.
        # ^ Uncommenting that will cause a runtime error, which is the purpose.

    # In Python, you can also use else clause on try-except block which must be present after all the
    #    except clauses. The code enters the else block only if the try clause does not raise an exception.

    # Program to depict else clause with try-except
    # Function which returns a/b
    def AbyB(self,a,b):
        try:
            c = ((a+b) / (a-b))
        except ZeroDivisionError:
            print("a/b result in 0")
        else:
            print(c)

    # The __init__ method is similar to constructors in C++ and Java. It is run as soon as an object of a class
    # is instantiated. The method is useful to do any initialization you want to do with your object.
    # def __init__(self, name):
    #     self.name = name                    # __init__ and say_hi working together to print name.
    # def say_hi(self):
    #     print "Hello, my name is", self.name

    # Important facts about functions in Python that are useful to understand decorator functions:
    # In Python, we can define a function inside another function.
    # In Python, a function can be passed as parameter to another function(a function can also return another)

    # Adds a welcome message to the string
    # def messageWithWelcome(str):
    #     # Nested function
    #     def addWelcome():
    #         return "Welcome to "
    #     # Return concatenation of addWelcome() and str.
    #     return addWelcome() + str
    #
    # # To get site name to which welcome is added
    # def site(site_name):
    #     return site_name
    #
    # print messageWithWelcome(site("GeeksForGeeks"))

    # A decorator is a function that takes a function as its only parameter and returns a function. This is helpful
    #    to "wrap" functionality with the same code over and over again. For example, above code can be
    #    re-written as the following.
    # We use @func_name to specify a decorator to be applied on another function.

    # Adds a welcome to the string returned by fun(). Takes fun() as parameter and returns welcome().
    def decorate_message(fun):
        def addWelcome(site_name):
            return "Welcome to " + fun(site_name)
        return addWelcome

    @decorate_message
    def site(site_name):
        return site_name

    # This call is equivalent to call to decorate_message() with function site("GeeksForGeeks") as parameter.
    print(site("GeeksForGeeks"))

    # Decorators can also be useful to attach data(or add attribute) to functions.
    # A Python example to demonstrate that decorators can be used to attach data

    # A decorator function to attach data to func
    def attach_data(func):
        func.data = 3
        return func

    @attach_data
    def add(x,y):
        return x + y

    # This call is equivalent to attach_data() with add() as parameter
    print((add(2,3)))
    print((add.data))

    # add() returns sum of x and y passed as arguments but it is wrapped by a decorator function, calling add(2,3)
    #    would simply give sum of two numbers but when we call add.data then 'add' function is passed into then
    #    decorator function 'attach_data' as argument and this function returns 'add' function with an attribute
    #    'data' that is set to 3 and hence prints it.
    # Python decorators are a powerful tool to remove redundancy.

    print("Everything below this is AFTER function decorators")

    # In Python, value of an integer is not restricted by the number of bits and can expand to the limit
    # of the available memory. Thus we never need any special arrangement for storing large numbers.
    # However, in Python 2.7 there are two separate types 'int' which is 32 bit, and 'long int' which
    # is the same as 'int' of Python 3.x, so it can store arbitrarily large numbers.
    def longIntTests(self):
        x = 10
        print(type(x))

        x = 100000000000000000000000000000000000000
        print(type(x))

        print(100**100)          # :)


    # A module is a file containing definitions and statements, it can define functions, classes, and variables.
    #    It can also include runnable code. Grouping related code into a module makes the code easier to understand.
    # We can use any Python source file as a module by executing an import statement in some other Python source file.
    # When the interpreter encounters an import statement, it imports the module if the module is present in the
    #    search path. A search path is a list of directories that the interpreter searches for importing a module.
    # For example, to import the module math.py, we need 'import math' at the top of the script.

    # Python's from statement lets you import specific attributes from a module. The from .. import .. has the following
    #    syntax: from match import sqrt, factorial
    # If we simply do 'import math' then math.sqrt(16) and match.factorial() are required, instead of just being able
    #    to write 'print sqrt(16)', or 'print factorial(6)'.

    # The dir() built-in function returns a sorted list of strings containing the names defined by a module.
    # The list contains the names of all the modules, variables and functions that are defined in a module.

    # imports at the top
    #print dir(math)

    def mathStuff(self):
        print(math.sqrt(25))
        print(math.pi)

        # 2 radians = 114.59 degrees
        print(math.degrees(2))

        # 60 degrees = 1.04 radians
        print(math.radians(60))

        # Sine of 2 radians, Cosine of 0.5 radians, Tangent of 0.23 radians
        print(math.sin(2))
        print(math.cos(.5))
        print(math.tan(.23))

        print(math.factorial(4))

        # random int between 0 and 5
        print(random.randint(0,5))

        # random floating point number between 0 and 1
        print(random.random())

        # random number between 0 and 100
        print(random.random() * 100)

        List = [1,4,True,800,"python",27,"hello"]

        # random element from a set such a list is chosen
        print(random.choice(List))

        # returns number of seconds since Unix Epoch, Jan 1st 1970
        print(time.time())

        # Converts a number of seconds to a date object
        print(date.fromtimestamp(454554))

        # Using modules to do fib sequence, first writing then returning a list.
        fibo.fib(100)
        print(fibo.fib2(100))

    def inputStuff(self):
        # Ways to get multiple values from user in one line:
        # This way requires hitting enter two separate times
        x,y = input(), input()
        print(x,y)

        # Or you can do this way, which requires separation by a whitespace.
        x,y = input().split()
        print(x,y)

        # To convert these strings to ints:
        x,y = [int(x),int(y)]
        print('x+y =', x, '+', y, '=', x+y)

        # Can also use list comprehension
        x,y = [int(x) for x in [x,y]]
        print('x+y =', x, '+', y, '=', x+y)

        # Below is a complete one line code to read two int variables from input using split and list comprehension.
        x,y = [int(x) for x in input().split()]
        print('x+y =', x, '+', y, '=', x + y)

        # What about taking variable number of inputs?
        print('enter any number of ints:', end=' ')
        l = list(map(int, input().split()))
        print('list:', l)
        print('list\'s sum:', sum(l))

        # What about grouping three elements in list together and summing them?
        # This ignores anything after the first three if there are not at least three elements remaining.
        newL = [sum(l[i:i+3]) for i in range(0, len(l) - len(l) % 3, 3)]
        print('Summing three elements in list, no overlap, only if there will be three elements added:', newL)

        # This way will just add up the remaining elements even if there are not at least three remaining.
        newL = [sum(l[i:i+3]) for i in range(0, len(l), 3)]
        print('Summing three elements in list, no overlap, including the last one or two elements always:', newL)

#Solver().demo()
#Solver().arithmetic()
#Solver().splitting()
#Solver().formatting()
#Solver().templates()
#Solver().listStuff()
#Solver().setIntro()
#Solver().operators()
#Solver().dictionaryStuff()
#Solver().breaks()
#Solver().mfl()
#Solver().exceptionHandling()
#Solver("Kevin").say_hi()        # Go to line 522 for this
#Solver().longIntTests()
Solver().mathStuff()             # modules/math stuff around line 607
#Solver().inputStuff()           # Good list stuff with for loops here as well.

# Function decorator stuff around like 534 - 587