import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import csv
from numpy import nan as NA

##### This all starts on page 177 of the PDF!

##### Chapter 7: Data Wrangling: Clean, Transform, Merge, Reshape

class PandasDataCh7:
    # Combining and Merging Data Sets

    #####
    # Database-style DataFrame Merges
    #####

    # Merge or join operations combine data sets by linking rows using one or more keys.
    # These operations are central to relational databases. The merge function in pandas is the main entry point
    # for using these algorithms on your data.

    df1 = DataFrame({'key': ['b','b','a','c','a','a','b'],
                     'data1': list(range(7))})
    df2 = DataFrame({'key': ['a','b','d'],
                     'data2': list(range(3))})
    print(df1, '\n\n', df2, '\n')

    print(pd.merge(df1, df2), '\n')
    print(pd.merge(df1, df2, on='key'), '\n') # Good practice

    df3 = DataFrame({'lkey': ['b','b','a','c','a','a','b'],
                     'data1': list(range(7))})
    df4 = DataFrame({'rkey': ['a','b','d'],
                     'data2': list(range(3))})

    print(pd.merge(df3, df4, left_on='lkey', right_on='rkey'), '\n')

    # Notice that c and d values and associated data are missing from the result. By default, merge does an 'inner'
    # join; the keys in the result are the intersection. Other possible options are 'left', 'right', and 'outer'.
    # The outer join takes the union of the keys, combining the effect of applying both left and right joins:

    print(pd.merge(df1, df2, how='outer'), '\n')

    left  = DataFrame({'key1': ['foo', 'foo', 'bar'],
                       'key2': ['one', 'two', 'one'],
                       'lval': [1, 2, 3]})
    right = DataFrame({'key1': ['foo','foo','bar','bar'],
                       'key2': ['one','one','one','two'],
                       'rval': [4, 5, 6, 7]})

    print(pd.merge(left, right, on='key1'), '\n')
    print(pd.merge(left, right, on='key1', suffixes=('_left', '_right')), '\n')

    # Table 7-1 has merge function arguments, page 181.

    #####
    # Merging on Index
    #####

    #####
    # Concatenating Along an Axis
    #####

    arr = np.arange(12).reshape((3,4))
    print(arr, '\n')
    print(np.concatenate([arr, arr], axis=1), '\n')

    s1 = Series([0,1], index=['a','b'])
    s2 = Series([2,3,4], index=['c','d','e'])
    s3 = Series([5,6], index=['f','g'])

    print(s1, '\n\n', s2, '\n\n', s3, '\n')

    print(pd.concat([s1,s2,s3]), '\n')
    print(pd.concat([s1,s2,s3], axis=1), '\n')

    s4 = pd.concat([s1 * 5, s3])
    print(s4, '\n')

    print(pd.concat([s1, s4], axis=1), '\n')
    print(pd.concat([s1, s4], axis=1, join='inner'), '\n')
    print(pd.concat([s1, s4], axis=1, join_axes=[['a','c','b','e']]), '\n')

    result = pd.concat([s1, s1, s3], keys=['one','two','three'])
    print(result, '\n')
    print(result.unstack(), '\n')

    # Concat function arguments in Table 7-2 on page 188

    #####
    # Combining Data with Overlap
    #####

    a = Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan],
               index=['f','e','d','c','b','a'])
    b = Series(np.arange(len(a), dtype=np.float64),
               index=['f','e','d','c','b','a'])
    b[-1]=np.nan

    print(a, '\n')
    print(b, '\n')
    print(np.where(pd.isnull(a), b, a), '\n')

    # Series has a combine_first method, which performs the equivalent of this operation plus data alignment:
    print(b[:-2].combine_first(a[2:]), '\n')

    # With DFs, combine_first naturally does the same thing column by column, so you can think of it as 'patching'
    # missing data in the calling object with data from the object you pass.

    # ie df1.combine_first(df2)

    #####
    # Reshaping and Pivoting
    #####

    #####
    # Data Transformation
    #####

    # Removing Duplicates
    data = DataFrame({'k1': ['one'] * 3 + ['two'] * 4,
                      'k2': [1,1,2,3,3,4,4]})
    print(data, '\n')
    print(data.duplicated(), '\n')
    print(data.drop_duplicates(), '\n')

    data['v1'] = list(range(7))
    print(data, '\n')
    print(data.drop_duplicates(['k1']), '\n')
    print(data.drop_duplicates(['k1', 'k2'], keep='last'), '\n')

    #####
    # Transforming Data Using a Function of Mapping
    #####

    #####
    # Discretization and binning
    #####
    ages = [20, 22, 25, 26, 28, 29, 21, 30, 29, 27, 45, 58, 63, 69, 36, 42]
    bins = [18, 25, 35, 60, 100]
    cats = pd.cut(ages, bins)
    print(cats, '\n')
    print(cats.codes, '\n')
    print(pd.value_counts(cats), '\n')

    # Can change which side has the closed bracket by passing right=False
    print(pd.value_counts(pd.cut(ages, bins, right=False)), '\n')

    # Passing your own label names:
    group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Sernior']
    print(pd.cut(ages, bins, labels=group_names), '\n')

    # Passing an integer to cut gives an equal number of bins
    data = np.random.rand(20)
    print(pd.value_counts(pd.cut(data, 4, precision=2)), '\n')