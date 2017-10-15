import numpy as np
import matplotlib.pyplot as plt         # First used at line 189ish
from numpy.linalg import inv, qr        # First used at line 414ish
from random import normalvariate        # First used at line 451ish
#import timeit

class NumPyDataLearning:
    data1 = [6, 7.5, 8, 0, 1]
    arr1 = np.array(data1)
    print(arr1)
    print()

    data2 = [data1, [5, 6, 7, 8]]
    arr2 = np.array(data2)
    print(arr2)
    print()

    print(np.zeros((3,6)))
    print()

    print(np.ones((3,6)))
    print()

    print(np.arange(2,15))
    print()

    int_arr1 = arr1.astype(np.int64)
    print(int_arr1)
    print()

    #operations
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    print(arr * arr)
    arr = arr.astype(np.float64)  #So we don't get 0 when we have fractions after this point
    print()

    print(arr - arr)
    print()

    print(1 / arr)
    print()

    print(arr ** 0.5)
    print()

    #splicing/indexing
    arr = np.arange(10)
    print(arr)
    print()

    arr[5:8] = 12
    print(arr)
    print()

    arr_slice = arr[5:8]
    arr_slice[1] = 12345

    print(arr)
    print()

    arr_slice[:] = 64   # Notice how the actual arr is being modified. If you want a copy, do .copy()
    print(arr)
    print()

    arr2d = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
    print(arr2d[2])
    print()
    print(arr2d[0][2])  # --------------------
    print()              # |                  |
    print(arr2d[0,2])   # These are equivalent

    arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    print(arr3d)
    print()
    print(arr3d[0])
    print()
    print(arr3d[1,0])
    print()

    # Splicing works similarly as you would think
    print(arr2d[:2, 1:])
    print()

    # Next is Boolean Indexing(p89)

    names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
    data2 = np.random.randn(7,4)
    print(names)
    print(data2)
    print(names == 'Bob')
    print(data2[names == 'Bob'])  # Can slice just like above. Is this vectorizing?
    print(data2[names == 'Bob', 2:])

    mask = (names == 'Bob') | (names == 'Will')     # Literal keywords 'and'/'or' do not work with boolean arrays
    print(mask)
    print(data2[mask])
    print()

    data2[data2 < 0] = 0
    print(data2)
    print()

    data2[names != 'Joe'] = 7
    print(data2)

    # Fancy Indexing (p92)
    arr = np.empty((8,4))       # Empty is slightly faster than .zeros, but requires manual change of each entry.
    for i in range(len(arr)):   # Or for i in range(8):
        arr[i] = i
    print(arr)
    print()
    print(arr[[4,3,0,6]])
    print('Using negative indices selects from the end: -3, -5, -7: ')
    print(arr[[-3,-5,-7]])

    arr = np.arange(32).reshape((8,4))
    print(arr)
    print()

    print(arr[[1, 5, 7, 2], [0, 3, 1, 2]]) # Getting elements (1,0), (5,3), ...
    print()
    print(arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])]) # Getting elements (1,0), (1, 3), (1, 1), (1, 2), (5, 0), ...
    print()

    # Transposing Arrays and Swapping Axes (p93)

    arr = np.arange(15).reshape((3,5))
    print(arr)
    print()
    print(arr.T)     # Transposing!
    print()

    # Very useful for matrix computations. ie Inner matrix product X^T X using np.dot:
    arr = np.random.randn(6,3)
    # print arr
    # print arr.T
    print(np.dot(arr.T, arr))
    print()

    # There is also some other crazy stuff doing things like
    # arr = np.arange(16).reshape((2,2,4))
    # print arr
    # print
    # Which gives some crazy stuff you can then transpose with axis numbers, etc.

    #####
    # Universal Functions: Fast Element-wise Array Functions (p95)
    #####

    # A ufunc is a function that performs elementwise operations on data in ndarrays.

    arr = np.arange(10)
    print(np.sqrt(arr)) # Many ufuncs are simple elementwise transformations, like sqrt or exp
    print()
    print(np.exp(arr))
    print()

    # Those two above are unary ufuncs. Others like add or maximum take 2 arrays, so binary ufuncs, and return a single
    # array.

    x = np.random.randn(8)
    y = np.random.randn(8)
    print('x: ', x)
    print()
    print('y: ', y)
    print()
    print('element-wise maximum: ', np.maximum(x,y))
    print()

    #####
    # Data Processing Using Arrays (p97) [Hella important visualizations, just the basic stuff]
    #####

    # Using NumPy arrays enables you to express many kinds of data processing tasks as concise array expressions
    # that might otherwise require writing loops. This practice of replacing explicit loops with array expressions
    # is commonly referred to as vectorization. In general, vectorized array operations will often be one or
    # two (or more) orders of magnitude faster than their pure Python equivalents, with the biggest impact in any kind
    # of numerical computations. Later, in Chapter 12, broadcasting will be explained, which is a powerful method
    # for vectorizing computations.

    # For a simple example, supposed we wished to evaluate the function sqrt(x^2 + y^2) across a regular grid of values.
    # The np.meshgrid function takes two 1D arrays and produces two 2D matrices corresponding to all pairs of (x, y)
    # in the two arrays:

    points = np.arange(-5, 5, 0.01) # 1000 equally spaced points
    xs, ys = np.meshgrid(points, points)
    print('xs: ', xs)
    print()
    print('ys: ', ys)
    print()

    # Now, evaluating the function is a simple matter of writing the same expression you would write with two points:
    # Import is at the top, obviously

    z = np.sqrt(xs ** 2 + ys ** 2)
    print('all z = xs^2 + ys^2', z)
    print()

    plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()
    plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
    #plt.show()

    # So what is this plot showing? We know that z ranges between 0 and ~7, this plot shows sort of a density of the
    # distribution of z values. There are 1000 values of xs and ys, so the 0th value of x is -5, and the 1000th value
    # of x is 5. A little hard to understand at first, but just try to trace through it.

    #####
    # Expressing Conditional Logic as Array Operations (p98)
    #####

    # The numpy.where function is a vectorized version of the ternary expression: x if condition else y.
    # Supposed we had a boolean array and two arrays of values:

    xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
    yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
    cond = np.array([True, False, True, True, False])

    # Now, supposed we wanted to take a value from xarr whenever the corresponding value in cond is True, otherwise
    # take the value from yarr. A list comprehension doing this might look like:

    result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
    print(result)
    print()

    # This has multiple problems. First, it will not be very fast for large arrays as all the work is being done in
    # pure Python. Second, it will not work with multidimensional arrays. With np.where you can write this
    # very concisely:

    result = np.where(cond, xarr, yarr)
    print(result)
    print()

    # The second and third arguments to np.where don't need to be arrays; one or both can be scalars.
    # A typical use of where in data analysis is to produce a new array of values based on another array.
    # Supposed you had a matrix of randomly generated data and you wanted to replace all positive values with 2
    # and all negative values with -2. This is very easy to do with np.where:

    arr = np.random.randn(4,4)
    print(arr)
    print()

    print(np.where(arr > 0, 2, -2))
    print()

    print(np.where(arr >0, 2, arr)) # Set only positive values to 2
    print()

    # The arrays passed to where can be more than just equal sized array or scalars.
    # With some cleverness you can use where to express more complicated logic; consider this example where I have
    # two booleans arrays, cond1 and cond2, and wish to assign a different value for each of the 4 possible pairs
    # of boolean values: (This code won't run btw, need to make the variables)

    # result = []
    # for i in range(n):
    #     if cond1[i] and cond2[i]:
    #         result.append(0)
    #     elif cond1[i]:
    #         result.append(1)
    #     elif cond2[i]:
    #         result.append(2)
    #     else:
    #         result.append(3)

    # While not immediately obvious, this for loop can be converted into a nested where expression:

    # np.where(cond1 and cond2, 0,
    #          np.where(cond1, 1,
    #                   np.where(cond2, 2, 3)))

    # In this particular example, we can also take advantage of the fact that boolean values are treated as 0 or 1
    # in calculations, so this could alternatively be expressed as an arithmetic operation, though a bit more cryptic:

    # result = 1 * cond1 + 2* cond2 + 3* -(cond1 or cond2)

    #####
    # Mathematical and Statistical Methods (p100)
    #####

    arr = np.random.randn(5, 4) # Normally distributed data
    print('arr: ', arr)
    print('arr mean: ', arr.mean())
    # print np.mean(arr) # Can use either mean()
    print('arr sum: ', arr.sum())

    # Functions like mean and sum take an optional axis argument which computes the statistic over the given axis,
    # resulting in an array with one fewer dimension:

    print('arr mean across axis 1: ', arr.mean(axis=1)) # axis=1 => horizontal, axis=0 => vertical I think?
    print('arr sum across axis 0(vertical): ', arr.sum(0)) # axis=0 not needed here I guess?

    # Other methods like cumsum and cumprod do not aggregate, instead producing an array of the intermediate results:

    arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    print('arr: ')
    print(arr)
    print('arr cumsum across axis 0(vertical): ')
    print(arr.cumsum(0))
    print('arr cumprod across axis 1(horizontal): ')
    print(arr.cumprod(1))

    # std and var are also functions. As well as min, max, argmin, argmax, etc etc.

    #####
    # Methods for Boolean Arrays (p101)
    #####

    # Sum is often used to count the number of True values in an array

    bools = np.random.randn(100)
    print((bools > 0).sum())

    # any() checks if at least one value is true, all() checks if all values are true.
    bools = np.array([False, False, True, False])
    print(bools.any())
    print(bools.all())
    print()
    # These methods also work for non-boolean arrays, where non-zero elements are evaluated as True.

    #####
    # Sorting
    #####

    arr = np.random.randn(8)
    arr.sort()
    print(arr)
    print()

    arr = np.random.randn(5,3)
    arr.sort(1) # Sorts each row
    print(arr)

    arr = np.random.randn(5,3)
    arr.sort(0) # Sorts each column
    print(arr)

    # The top level method np.sort returns a sorted copy of an array instead of modifying
    # the array in place. A quick-and-dirty way to compute the quantiles of an array is to sort
    # it and select the value at a particular rank:

    large_arr = np.random.randn(1000)
    large_arr.sort()
    print(large_arr[int(0.05 * len(large_arr))]) # 5% Quantile

    #####
    # Unique and other Set Logic
    #####

    # Unique returns an array without the duplicates

    names = np.array(['Bob', 'BoB', 'Rita', 'Kevin', 'Rita', 'Bob'])
    print(np.unique(names))
    ints = np.array([12, 12, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 10, 10, 11, 12, 12, 12, 12])
    print(np.unique(ints))

    # Array Set Operations:
    # intersect1d(x, y)       Compute sorted common elements of x and y
    # union1d(x, y)           Compute sorted union of elements
    # in1d(x, y)              Compute a boolean array indicating whether each element of x is contained in y
    # setdiff1d(x, y)         Set difference, elements contained in x that are not in y
    # setxor1d(x, y)          Set symmetric differences; Elements that are in either array, but not both.

    x = np.array([1, 2, 3, 10, 20, 30])
    y = np.array([2, 4, 6, 8, 10, 30])

    print('              union: ', np.union1d(x, y))
    print('          intersect: ', np.intersect1d(x, y))
    print('            x in y?: ', np.in1d(x, y))
    print('        in x, not y: ', np.setdiff1d(x, y))
    print('in x or y, not both: ', np.setxor1d(x, y))
    print()

    #####
    # File Intput and Output with Arrays (p103)
    #####

    #####
    # Storing Arrays on Disk in Binary Format
    #####

    arr = np.arange(10)
    np.save('some_array', arr)
    #print np.load('some_array.npy')
    array = np.load('some_array.npy')
    print(array)
    print()

    np.savez('array_archive.npz', a=arr, b=arr) # Saves multiple arrays in a zip archive
    arch = np.load('array_archive.npz')
    print(arch['b'])
    print()

    #####
    # Saving and Loading Text Files
    #####

    np.savetxt('array_ex', [-1.5, 1.2, -3, 324, 1618, -314, 450, 0, 250, 623, 125, 63])

    arr = np.loadtxt('array_ex', delimiter=',')
    print(arr)

    # Chapter 12 has more file stuff, definitely check it out

    #####
    # Linear Algebra
    #####

    x = np.array([[1,2,3], [4,5,6]])
    y = np.array([[6,23], [-1,7], [8,9]])
    print('X: ')
    print(x)
    print('Y: ')
    print(y)
    print('XY: ')
    print(x.dot(y))

    print('X dot np.ones(3): ')
    print(np.dot(x, np.ones(3)))
    print()

    X = np.random.randn(5, 5)
    mat = X.T.dot(X)
    print(inv(mat))
    print()
    print(mat.dot(inv(mat)))

    q, r = qr(mat)
    print(r)
    print()

    # Commonly used numpy.linalg functions
    # diag        Return the diagonal(or off-diagonal) elements of a square matrix as a 1D array, or convert a
    #             1D array into a square matrix with zeros on the off-diagonal
    # dot         Matrix Multiplication
    # trace       Compute the sum of the diagonal elements
    # det         Compute the matrix determinant
    # eig         Compute the eigenvalues and eigenvectors of a square matrix
    # inv         Compute the inverse of the matrix
    # pinv        Compute the Moore-Penrose pseudo-inverse inverse of a square matrix
    # qr          Compute the QR decomposition
    # svd         Compute the Singular Value Decomposition
    # solve       Solve the linear system Ax = b for x, where A is a square matrix
    # lstsq       Compute the least-squares solution to y = Xb

    #####
    # Random Number Generation (p106)
    #####

    # numpy's rng is much more efficient than pure python implementation

    # 4x4 array of samples from the standard normal distribution:
    samples = np.random.normal(size=(4,4))
    print(samples)
    print()

    # See the efficiency in action here:
    #N = 1000000
    #
    # Can't get this working, timeit is weird in an IDE. :/
    #
    # print 'Pure Python speed: '
    #print timeit.timeit(stmt="samples = [normalvariate(0,1) for _ in range(N)]", setup='pass',
    #                    timer=timeit.default_timer, number = N)
    #
    # print 'numpy speed: '
    #print timeit.timeit(stmt='np.random.normal(size=N)')

    # numpy.random functions
    # seed                Seed the random number generation
    # permutation         Return a random permutationof a sequence, or return a permutated range
    # shuffle             Randomly permute a sequence in place
    # rand                Draw samples from a uniform distribution
    # randint             Draw random integers from a given low-to-high range
    # randn               Draw samples from a normal distribution with mean 0 and std 1(MATLAB-like interface)
    # binomial            Draw samples from a binomial distribution
    # normal              Draw samples from a normal(Gaussian) distribution
    # beta                Draw samples from a beta distribution
    # chisquare           Draw samples from a chisquare distribution
    # gamma               Draw samples from a gamma distribution
    # uniform             Draw samples from a uniform [0,1) distribution

    #####
    # Random Walks example!!!! (p108)
    #####

    # Pure Python implementation first...
    # position = 0
    # walk = [position]
    # steps = 1000
    # for i in xrange(steps):
    #     step = 1 if random.randint(0,1) else -1
    #     position += step
    #     walk.append(position)

    # Now, just do 1000 coin flips at once and do the cumsum for more efficiency! Woohoo!
    nsteps = 1000
    draws = np.random.randint(0, 2, size = nsteps)
    steps = np.where(draws > 0, 1, -1)
    walk = steps.cumsum()

    # How about some stats along this walk?!
    print('Min during the random walk: ', walk.min())
    print('Max during the random walk: ', walk.max())

    # What about first crossing time? Like when we reach a particular value, say 10 steps away from the origin? How long
    # did that take with our random walk?
    print((np.abs(walk) >= 10).argmax()) # argmax() returns the first index of the max value in boolean array, True.
    print()

    #####
    # Simulating many random walks at once (p109)
    #####

    # Let's do 5000 random walks. Just modify the code above and donezo.

    nwalks = 5000
    nsteps = 1000
    draws = np.random.randint(0, 2, size=(nwalks, nsteps)) # 0 or 1
    print('draws: ', draws)
    steps = np.where(draws > 0, 1, -1)
    print('steps: ', steps)
    walks = steps.cumsum(1)
    print(walks)

    print('Min over all walks: ')
    print(walks.min())
    print('Max over all walks: ')
    print(walks.max())

    # Out of these walks, let's compute the minimum crossing time to 30 or -30. This is slightly tricky because
    # not all 5000 of them reach 30, we can check this using the any method:
    hits30 = (np.abs(walks) >= 30).any(1)
    print('Min crossing time of 30 or -30: ')
    print(hits30)

    # We can use thisi boolean array to select out the rows of walks that actually cross the absolute 30 level
    # and call argmax across axis 1 to get the crossing times:
    crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)
    print('crossing_times: ')
    print(crossing_times)
    print('mean crossing times: ')
    print(crossing_times.mean())

    # Can experiment with other  distributions for the steps other than equal sized coin flips. Just use a different
    # RNG function, like normal to generate normally distributed steps with some mean and std:
    # steps = np.random.normal(loc = 0, scale = 0.25, size = (nwalks, nsteps))

    # That's it for into numpy stuff! Officially finishes on page 110, which is the end of chapter 4.
    # Chapter 5(Pandas starts!) beings on page 111, which will be in a different python file.