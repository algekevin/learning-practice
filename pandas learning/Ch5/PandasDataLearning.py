import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import pandas as pd
from numpy import nan as NA                     # Line 532
from pandas_datareader import data as web       # First used at line 465
import fix_yahoo_finance
import datetime

# This all beings on page 111

# Chapter 5
# Getting Started with pandas

class PandasDataLearning:
    #####
    # 5.1 Introduction to pandas Data Structures
    #####

    # To get comfortable with pandas, need to get comfy with its two workhorse data structures: Series and Data Frame.
    # While not a universal solution for every problem, they provide a solid, easy-to-use basis for most applications.

    #####
    # Series
    #####

    # A Series is a one-dimensional array-like object containing an array of data(of any NumPy data type) and an
    # associated array of data labels, called its index. The simplest Series is formed from only an array of data:
    obj = Series([4, 7, -5, 3])
    print(obj)
    print()

    # Often it will be desirable to create a Series with an index identifying each data point:
    obj2 = Series([4, 7, -5, 3], index=['d','b','a','c'])
    print(obj2)
    print(obj2.index)
    print()

    print('obj2[\'a\']: ', obj2['a'])
    print()

    obj2['d'] = 6
    print(obj2[['c','a','d']])
    print()

    # NumPy array operations, such as filtering with a boolean array, scalar multiplication, or applying math
    # functions, will preserve the index-value link:

    print('obj2: ')
    print(obj2)
    print()

    print('obj2[obj2 > 0]: ')
    print(obj2[obj2 > 0])

    print('obj2 * 2: ')
    print(obj2 * 2)

    print('np.exp(obj2): ')
    print(np.exp(obj2))

    print()

    print('b in obj2: ', 'b' in obj2)
    print('e in obj2: ', 'e' in obj2)
    print()

    # Should you have data contained in a Python dict, you can create a Series from it by passing the dict:
    sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
    obj3 = Series(sdata)
    print('obj3: ')
    print(obj3)
    print()

    # As you can see, when passing only the dict it returns the indeces in sorted order.

    states = ['California', 'Ohio', 'Texas', 'Oregon']
    obj4 = Series(sdata, index=states)
    print('obj4: ')
    print(obj4)      # Notice how not sorted by index

    # No value found for California, isnull and notnull functions in pandas should be used to detect missing data:
    print('isnull: ')
    print(pd.isnull(obj4))
    print()
    print('notnull: ')
    print(pd.notnull(obj4))
    print()

    # Series also has these as instance methods:
    # print obj4.isnull()

    # Working with missing data will be discussed in more detail later in this chapter.

    # A critical Series feature for many applications is that it automatically aligns differently-indexed data in
    # arithmetic operations:

    print('obj3 + obj4: ')
    print(obj3 + obj4)
    print()

    # Data alignment features are addressed as a separate topic.

    # Both the Series object itself and its index have a name attribute, which integrates with other key areas of
    # pandas functionality:
    obj4.name = 'population'
    obj4.index.name = 'state'
    print(obj4)
    print()

    # A Series' index can be altered in place by assignment:
    obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
    print(obj)
    print()

    #####
    # DataFrame
    #####

    # A DataFrame can sort of be thought of as a disc of Series(one for all sharing the same index).

    # There are numerous ways to construct a DataFrame, though one of the most common is from a dict of
    # equal-length lists or NumPy arrays
    data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
            'year' : [2000, 2001, 2002, 2001, 2002],
            'pop'  : [1.5, 1.7, 3.6, 2.4, 2.9]}
    frame = DataFrame(data)

    # The resulting DataFrame will have its index assigned automatically as with Series, and the columns are placed
    # in sorted order:
    print(frame)
    print()

    # If you specify a sequence of columns, the DataFrame's columns will be exactly what you pass:
    print(DataFrame(data, columns=['year', 'state', 'pop']))
    print()

    frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                       index=['one','two','three','four','five'])
    print(frame2)
    print()

    # A column in a DataFrame can be retrieved as a Series either by dict-like notation or by attribute:
    print(frame2['state'])
    print()
    print(frame2.year)
    print()

    # Rows can also be retrieved by position or name by a couple of methods, such as the ix indexing field(more later)
    print(frame2.ix['three'])
    print()

    # Columns can be modified by assignment. For example, the empty 'debt' column could be assigned a scalar value
    # or an array of values:

    frame2['debt'] = 16.5
    print(frame2)
    print()

    frame2['debt'] = np.arange(5)
    print(frame2)
    print()

    # When assigning lists or arrays to a column, the value's length must match the length of the DataFrame. If you
    # assign a Series, it will be instead conformed exactly to the DF's index, inserting missing values in any holes:

    val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
    frame2['debt'] = val
    print(frame2)
    print()

    # Assigning a column that doesn't exist will create a new column. The del keyword will delete columns as with a dict
    frame2['eastern'] = frame2.state == 'Ohio'
    print(frame2)
    print()
    del frame2['eastern']
    print(frame2.columns)
    print()

    # Another common form of data is a nested dict of dicts format:

    pop = {'Nevada': {2001: 2.4, 2002: 2.9},
           'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
    # If passed to DF, it will interpret the outer dict keys as the columns and the inner keys as the row indeces:
    frame3 = DataFrame(pop)
    print(frame3)
    print()

    # Of course you can always transpose the result:
    print(frame3.T)
    print()

    # The keys in the inner dicts are unioned and sorted to form the index in the result. This isn't true if
    # an explicit index is specified:

    print(DataFrame(pop, index=[2001, 2002, 2003]))
    print()

    # Dicts of Series are treated much in the same way:
    pdata = {'Ohio': frame3['Ohio'][:-1],
             'Nevada': frame3['Nevada'][:2]}
    print(DataFrame(pdata))
    print()

    frame3.index.name = 'year'; frame3.columns.name = 'state'
    print(frame3)
    print()

    print(frame3.values)
    print()

    print(frame2.values)
    print()

    # p120, Table 5-1 for possible data inputs to DataFrame constructor.

    #####
    # Index Objects
    #####

    # pandas' Index objects are responsible for holding the axis labels and other metadata (like the axis name or names)
    # Any array or other sequence of labels used when constructing a Series or DataFrame is internally converted to an
    # Index:

    obj = Series(list(range(3)), index=['b', 'a', 'c'])
    index = obj.index
    print(index)
    print(index[1:])
    print()

    # Index objects are immutable and thus can't be modified by the user:
    # index[1] = 'd' # Results in an error

    # Immutability is important so that Index objects can be safely shared among data structures:
    index = pd.Index(np.arange(3))
    obj2 = Series([1.5, -2.5, 0], index=index)
    print(obj2.index is index)
    print()

    #####
    # Essential Functionality (p122)
    #####

    # reindex creates a new object with the data conformed to a new index.
    print('obj: \n', obj)
    obj2 = obj.reindex(['a','b','c','d','e'])
    print('obj2: \n', obj2)
    print('obj with reindex: \n', obj.reindex(['a','b','c','d','e'], fill_value = 0))
    print('obj still unchanged: \n', obj)

    # For ordered data like time series, it may be desirable to do some interpolation or filling of values when
    # reindexing. The method option allows us to do this, using a method such as ffil which forward fills values:
    obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 4, 6])
    print('obj3: \n', obj3)
    print('obj3 reindexed with ffill: \n', obj3.reindex(list(range(10)), method='ffill'))
    # bfill fills or varries values backward, reverse of ffill.

    # With DF, reindex can alter either the (row) index, columns, or both. When passed just a sequence, rows are indexed
    # in the result:
    frame = DataFrame(np.arange(9).reshape((3,3)), index=['a','c','d'],
                      columns=['Ohio','Texas','California'])
    print('frame: \n', frame)
    frame2 = frame.reindex(['a','b','c','d'])
    print('frame2(frame reindexed): \n', frame2)
    # Columns can be reindexed using the columns keyword:
    states = ['Texas', 'Utah', 'California']
    print('frame reindexed with states, replacing ohio with utah: \n', frame.reindex(columns=states))

    # Both can be reindexed in one shot, though interpolation will only apply row-wise (axis 0):
    print('frame reindexed in one shot(columns and rows): ')
    print(frame.reindex(index=['a', 'b', 'c', 'd'], fill_value=69, columns=states, method='ffill'))

    # Reindexing can be done more succinctly by label-indexing with ix:
    print(frame.ix[['a','b','c','d',], states])

    #####
    # Dropping entries from an axis (p125)
    #####

    obj = Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
    new_obj = obj.drop('c')
    print('new_obj: \n', new_obj)
    print('obj without a and d: \n', obj.drop(['d', 'a']))

    # With DF, index values can be deleted from either axis:
    data = DataFrame(np.arange(16).reshape(4,4),
                     index=['Ohio', 'Colorado', 'Utah', 'New York'],
                     columns=['one', 'two', 'three', 'four'])
    print('data: \n', data)
    print('data without Colorado and Ohio: \n', data.drop(['Colorado', 'Ohio']))
    print('data without column \'two\': \n', data.drop('two', axis=1))
    print('more data drops: \n', data.drop(['two', 'four'], axis=1), '\n')

    #####
    # Indexing, selection, and filtering (p125)
    #####

    obj = Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
    print('obj[\'b\']: \n', obj['b'], '\n')
    print('obj[1]: \n', obj[1], '\n')
    print('obj[2:4]: \n', obj[2:4], '\n')
    print('obj[[b, a, d]]: \n', obj[['b', 'a', 'd']], '\n')
    print('obj[[1,3]]: \n', obj[[1,3]], '\n')
    print('obj[obj < 2]: \n', obj[obj < 2], '\n')

    # Slicing with labels behaves differently than normal Python slicing because the endpoint is inclusive:
    print('obj[\'b\':\'c\']: \n', obj['b':'c'], '\n')

    # Setting using these methods works just as you would expect:
    obj['b':'c'] = 5
    print('obj after changing b and c: \n', obj, '\n')

    data = DataFrame(np.arange(16).reshape((4,4)),
                     index = ['Ohio', 'Texas', 'California', 'Colorado'],
                     columns = ['one', 'two', 'three', 'four'])
    print(data, '\n')
    print(data[data['three'] > 5], '\n')
    data[data['three'] < 5] = 0
    print(data, '\n')
    print(data < 5, '\n')

    print(data.ix['Colorado', ['two', 'three']], '\n')
    print(data.ix['Colorado', [1, 2]], '\n')
    print(data.ix[['Colorado', 'Texas'], [1, 2]], '\n')
    print(data.ix[:'California', 'two'], '\n')
    print(data.ix[data.three > 5, :3])

    # Rich label indexing is all pushed into .ix
    # Table 5-6 on page 128 has lots of indexing options with DataFrame

    #####
    # Arithmetic and data alignment (p128)
    #####

    s1 = Series([7.3, -2.5, 3.4, 1.5], index=['a','c','d','e'])
    s2 = Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])
    print('s1: \n', s1, '\n')
    print('s2: \n', s2, '\n')
    print('s1+s2: \n', s1 + s2, '\n')

    # NA values are introduced to the indices that don't overlap. Missing values propogate in arithmetic computations.

    df1 = DataFrame(np.arange(9.).reshape((3,3)),  columns=list('bcd'), index=['Ohio', 'Texas', 'Colorado'])
    df2 = DataFrame(np.arange(12.).reshape((4,3)), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])

    print('df1: \n', df1, '\n')
    print('df2: \n', df2, '\n')
    print('df1+df2: \n', df1+df2, '\n')

    # Arithmetic methods with fill values

    df1 = DataFrame(np.arange(12.).reshape((3,4)), columns=list('abcd'))
    df2 = DataFrame(np.arange(20.).reshape((4,5)), columns=list('abcde'))
    print('df1: \n', df1, '\n')
    print('df2: \n', df2, '\n')
    print('df1+df2: \n', df1+df2, '\n')

    print('df1+df2, NaN being filled: \n', df1.add(df2, fill_value=0), '\n')
    # Can also reindex to specify a different fill value
    print(df1.reindex(columns=df2.columns, fill_value=0.))

    # Operations between DF and Series on Page 130, skipping it because it's sorta intuitive. Broadcasting is done there

    #####
    # Function application and mapping (p132)
    #####

    frame = DataFrame(np.random.randn(4, 3), columns=list('bde'),
                      index=['Utah', 'Ohio', 'Texas', 'Oregon'])
    print(frame, '\n')
    print(np.abs(frame), '\n')

    f = lambda x: x.max() - x.min() # This takes each row and subtracts the min from the max of frame, not abs(frame).
    print('frame.apply(f), where f is defined as x.max() - x.min(): \n', frame.apply(f), '\n')
    print('Same as above but over axis 1: \n', frame.apply(f, axis=1), '\n')

    # Not sure how the below is working, but it does haha. Even when the above is

    def f(x):
        return Series([x.min(), x.max()], index=['min', 'max'])
    print(frame.apply(f), '\n')

    format = lambda x: '%.2f' % x
    print(frame.applymap(format), '\n')
    print(frame['e'].map(format), '\n')

    #####
    # Sorting and ranking
    #####

    obj = Series(list(range(4)), index=['d','a','b','c'])
    print(obj.sort_index(), '\n')

    frame = DataFrame(np.arange(8).reshape((2,4)),
                      index = ['three', 'one'],
                      columns = ['d', 'b', 'c', 'a'])
    print(frame, '\n')
    print(frame.sort_index(), '\n')
    print(frame.sort_index(axis=1), '\n')
    print(frame.sort_index(axis=1, ascending=False), '\n')

    # To sort by values, use sort_values method for Series, the older 'order()' method is deprecated(gone).
    obj = Series([4, 7, -3, 2])
    print(obj, '\n')
    print(obj.sort_values(), '\n')
    # NaN is sorted to the bottom by default.

    frame = DataFrame({'b': [4,7,-3,2], 'a': [0,1,0,1]})
    print(frame, '\n')
    print('sorted by values of column b: \n', frame.sort_values(by='b'), '\n')
    print(frame.sort_values(by=['b', 'a']), '\n')
    print(frame.sort_values(by=['a', 'b']), '\n')

    # rank() doesn't make much sense to me on page 135. Really should go back and look through that again.

    #####
    # Axis indexes with duplicate values (p136)
    #####

    # Everything up to this point had unique labels. Many pandas functions like reindex require that, but it is not
    # mandatory.

    obj = Series(list(range(5)), index=['a','a','b','b','c'])
    print(obj, '\n')
    # is_unique can tell you whether its values are unique or not.
    print('Are the indices of obj unique?', obj.index.is_unique, '\n')

    # Same logic extends to DataFrame
    df = DataFrame(np.random.randn(4,3), index=['a','a','b','b'])
    print(df, '\n')
    print(df.ix['b'], '\n')

    #####
    # Summarizing and Computing Descriptive Statistics (p137)
    #####

    df = DataFrame([[1.4, np.nan], [7.1, -4.5],
                    [np.nan, np.nan], [0.75, -1.3]],
                    index=['a','b','c','d'],
                    columns=['one','two'])
    print(df, '\n')
    print('df sum across columns: \n', df.sum(), '\n')
    print('df sum across rows(axis=1): \n', df.sum(axis=1), '\n')
    # NaN values are excluded unless the entire slice(row or column in this case) is NaN. This can be disabled
    # using the skipna option:
    print('df mean across rows, not skipping NaN: \n', df.mean(axis=1, skipna=False), '\n')

    # Some methods, like idxmin and idxmax, return indirect statistics like the index value where the minimum
    # or maximum values are attained:
    print('index where max located: \n', df.idxmax(), '\n')
    print('cumsum: \n', df.cumsum(), '\n')

    # The magical describe()
    print(df.describe(), '\n')

    # On non-numeric data, describe() produces alternate summary statistics:
    obj = Series(['a','a','b','c'] * 4)
    print(obj, '\n')
    print(obj.describe(), '\n')

    #####
    # Correlation and Covariance (p139)
    #####

    # Some summary statistics, like correlation and covariance, are computed from pairs of arguments.
    # Consider some DFs of stock prices and volumes obtained from Yahoo! Finance:

    # start = datetime.datetime(2010, 1, 1)
    # end = datetime.datetime(2017, 6, 1)
    #
    # all_data = {}
    # for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOGL']:
    #     all_data[ticker] = web.get_data_yahoo(ticker, start, end)
    #
    # price  = DataFrame({tic: data['Adj Close']
    #                     for tic, data in all_data.iteritems()})
    # volume = DataFrame({tic: data['Volume']
    #                     for tic, data in all_data.iteritems()})
    #
    # returns = price.pct_change()
    # print 'Percent changes: \n', returns.tail(), '\n'
    #
    # # The corr method of Series computes the correclation of the overlapping, non-NA, aligned-by-index values in
    # # two Series. Relatedly, cov computes the covariance:
    #
    # print 'corr MSFT and IBM: \n', returns.MSFT.corr(returns.IBM), '\n'
    # print 'cov MSFT and IBM: \n', returns.MSFT.cov(returns.IBM), '\n'
    #
    # # DataFrame's corr and cov methods, on the other hand, return a full correlation or covariance matrix as a DF:
    # print 'returns.corr(): \n', returns.corr(), '\n'
    # print 'returns.cov(): \n', returns.cov(), '\n'
    #
    # print 'returns.corrwith(returns.IBM): \n', returns.corrwith(returns.IBM), '\n'
    # print 'returns.corrwith(volume), which is correlations of percent changes with volume: '
    # print returns.corrwith(volume), '\n'

    #####
    # Unique Values, Value Counts, and Membership (p141)
    #####

    obj = Series(['c','a','d','a','a','b','b','c','c'])
    uniques = obj.unique()
    print('uniques(values without their duplicates): \n', uniques, '\n')
    # uniques.sort()
    # print 'uniques sorted: \n', uniques, '\n'

    print('value counts of obj: \n', obj.value_counts(), '\n')
    print('value counts sorted by value name: \n', pd.value_counts(obj.values, sort=False), '\n')

    # isin computes a boolean array indicating whether each Series value is contained in the passed sequence of values.
    mask = obj.isin(['b','c'])
    print(mask, '\n')
    print(obj[mask], '\n')

    # In some cases, you may want to compute a histogram on multiple related columns in a DF. Example:
    data = DataFrame({'Qu1': [1,3,4,3,4],
                      'Qu2': [2,3,1,2,3],
                      'Qu3': [1,5,2,4,4]})
    print(data, '\n')
    result = data.apply(pd.value_counts).fillna(0)
    print(result, '\n')

    #####
    # Handling Missing Data (p142)
    #####

    # mostly basic stuff in first section...

    #####
    # Filtering Out Missing Data
    #####

    # dropna() returns a Series with only the non-null data and index values
    data = Series([1, NA, 3.5, NA, 7])
    print(data, '\n')
    print(data.dropna(), '\n')
    # Can also just do boolean indexing
    print(data[data.notnull()], '\n')

    # With DFs, things are more complex. dropna() by default drops any row containing a missing value:
    data = DataFrame([[1, 6.5, 3], [1, NA, NA],
                     [NA, NA, NA], [NA, 6.5, 3]])
    print(data, '\n')
    print(data.dropna(), '\n')
    # Passing how='all' will only drop rows that are all NA:
    print(data.dropna(how='all'), '\n')

    # Dropping columns in the same way is only a matter of passing axis=1:
    data[4] = NA
    print(data, '\n')
    print(data.dropna(axis=1, how='all'), '\n')
    print(data.dropna(axis=1), '\n')

    # A related way to filter out DF rows tends to concern time series data. Suppose you want to keep only rows
    # containing a certain number of observations. you can indicate this with the thresh argument:
    df = DataFrame(np.random.randn(7,3))
    print(df, '\n')
    df.ix[:4, 1] = NA
    df.ix[:2, 2] = NA
    print(df, '\n')
    print(df.dropna(thresh=3), '\n')

    #####
    # Filling in Missing Data
    #####

    print(df.fillna(0), '\n')
    print(df.fillna({1: 0.5, 2: 69}), '\n')
    _ = df.fillna(0, inplace=True)
    print(df, '\n')

    df = DataFrame(np.random.randn(6,3))
    df.ix[2:, 1] = NA
    df.ix[4:, 2] = NA
    print(df, '\n')

    print(df.fillna(method='ffill'), '\n')
    print(df.fillna(method='ffill', limit=2), '\n')

    data = Series([1, NA, 3.5, NA, 7])
    print('data where na values are filled with the mean of the data: \n', data.fillna(data.mean()), '\n')

    #####
    # Hierarchical Indexing (p146)
    #####

    # Hierarchical Indexing lets you have multiple index levels on an axis. Simple example:

    data = Series(np.random.randn(10),
                  index=[['a','a','a','b','b','b','c','c','d','d'],
                         [1,2,3,1,2,3,1,2,2,3]])
    print(data, '\n')
    print(data.index, '\n')

    print(data['b'], '\n')
    print(data['b':'c'], '\n')
    print(data.ix[['b', 'd']], '\n')
    print('selection from an inner level: \n', data[:, 2], '\n')
    print('data[\'a\',2]: ', data['a',2], '\n')

    # Hierarchical indexing plays a critical role in reshaping data and group-based operations like forming a
    # pivot table. For example, this data could be rearranged into a DF using its unstack method:

    print(data.unstack(), '\n')
    # The opposite is stack()
    print(data.unstack().stack(), '\n')

    # Stack and unstack will be explored more in Chapter 7

    # With a DF, either axis can have a hierarchical index:
    frame = DataFrame(np.arange(12).reshape((4,3)),
                      index=[['a','a','b','b'], [1,2,1,2]],
                      columns=[['Ohio', 'Ohio', 'Colorado'],
                               ['Green', 'Red', 'Green']])
    print(frame, '\n')

    # You can name the hierarchical levels(don't confuse index names with axis labels):
    frame.index.names = ['key1', 'key2']
    frame.columns.names = ['state', 'color']
    print(frame, '\n')

    # A MultiIndex can be created by itself and then reused; the columns in the above DF with level names
    # could be created like this:
    #MultiIndex.from_arrays([['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']],
    #                       names=['state', 'color'])

    #####
    # Reordering and Sorting levels (p149)
    #####

    print(frame.swaplevel('key1', 'key2'), '\n')
    print(frame.sortlevel(1), '\n')
    print(frame.swaplevel(0, 1).sortlevel(0), '\n')

    # Data selection performance is much better on hierarchically indexed objects if the index is
    # lexicographically sorted starting with the outer-most level, that is, the result of calling sortlevel(0)
    # or sort_index().

    #####
    # Summary Statistics by Level
    #####

    print(frame.sum(level='key2'), '\n')
    print(frame.sum(level='color', axis=1), '\n')

    #####
    # Using a DataFrame's Columns
    #####

    frame = DataFrame({'a': list(range(7)), 'b': list(range(7,0,-1)),
                       'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'],
                       'd': [0,1,2,0,1,2,3]})
    print(frame, '\n')

    # DF's set_index function will create a new DF using one or more of its columns as the index:
    frame2 = frame.set_index(['c','d'])
    print(frame2, '\n')

    # by default the columns are removed from the DF, though you can leave them in:
    print(frame.set_index(['c','d'], drop=False), '\n')

    # reset_index, on the other hand, does the opposite of set_index; the hierarchical index levels are
    # moved into the columns:

    print(frame2, '\n')
    print(frame2.reset_index(), '\n')

    # End of chapter 5 and whatever else there was! PandasDataCh6 contains Data Loading, Storage, and File Formats.