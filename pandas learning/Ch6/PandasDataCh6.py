import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import csv
from lxml.html import parse                 # Line 181
from urllib.request import urlopen          # Line 181
from pandas.io.parsers import TextParser    # Line 222
import requests                             # Line 236
import pandas.io.sql as sql                 # Line 288
import json
from numpy import nan as NA

# This is Chapter 6: Data Loading, Storage, and File Formats
# It all beings on page 155 of the PDF!

class PandasDataCh6:
    #####
    # Reading and Writing Data in Text Format
    #####

    # Table 6-1, Parsing functions in pandas:
    # read_csv            Load delimited data from a file, URL, or file-like object. Use comma as default delimiter.
    # read_table          Load delimited data from a file, URL, or file-like object. Use tab ('\t') as default delimiter.
    # read_fwf            Read data in fixed-width column format(that is, no delimiters).
    # read_clipboard      Version of read_table that reads data from the clipboard.
    #                     Useful for converting tables from web pages.

    # read_csv and read_table are likely the ones we will use the most. These are all meant to convert text into a DF.

    np.savetxt('ch06/ex1.csv', ['a,b,c,d,message\n1,2,3,4,hello\n5,6,7,8,world\n9,10,11,12,foo'], fmt='%s')

    df = pd.read_csv('ch06/ex1.csv')
    print(df, '\n')

    # print pd.read_table('ch06/ex1.csv', sep=','), '\n'     # This also works, sep being the delimiter.

    np.savetxt('ch06/ex2.csv', ['1,2,3,4,hello\n5,6,7,8,world\n9,10,11,12,foo'], fmt='%s')
    newdf = pd.read_csv('ch06/ex2.csv')
    print(newdf, '\n')

    newdf = pd.read_csv('ch06/ex2.csv', names=['a','b','c','d','message'])
    print(newdf, '\n')

    newdf = pd.read_csv('ch06/ex2.csv', names=['a', 'b', 'c', 'd', 'message'], index_col='message')
    print(newdf, '\n')

    np.savetxt('ch06/csv_mindex.csv', ['key1,key2,value1,value2\none,a,1,2\none,b,3,4\none,c,5,6\ntwo,a,9,10\ntwo,b,11,12'], fmt='%s')
    parsed = pd.read_csv('ch06/csv_mindex.csv', index_col=['key1', 'key2'])
    print(parsed, '\n')

    np.savetxt('ch06/ex5.csv', ['something,a,b,c,d,message\none,1,2,3,4,NA\ntwo,5,6,,8,world\nthree,9,10,11,12,foo'], fmt='%s')
    result = pd.read_csv('ch06/ex5.csv')
    print(result, '\n')

    # na_values option can take eithe ra list or set of strings to consider missing values:
    result = pd.read_csv('ch06/ex5.csv', na_values=['NULL'])
    print(result, '\n')

    # Different NA sentinels can be specified for each column in a dict:
    sentinels = {'message': ['foo', 'NA'], 'something': ['two']}
    result = pd.read_csv('ch06/ex5.csv', na_values=sentinels)
    print(result, '\n')

    #####
    # Table 6-2 on page 160 has read_csv/read_table function arguments.
    #####

    #####
    # Reading Text Files in Pieces p160
    #####

    print(pd.read_csv('ch06/ex5.csv', nrows=2), '\n')

    #####
    # Writing Data Out to a Text Format
    #####

    print(result, '\n')
    result.to_csv('ch06/out.csv')

    # Other delimiters can be used, of course:
    result.to_csv('ch06/outSepVertBar.csv', sep='|')

    # Missing values appear as empty strings in the output. You might want to denote them by some other sentinel val:
    result.to_csv('ch06/outWithNULL.csv', na_rep='NULL')

    # With no other options specified, both the row and column labels are written. Both of these can be disabled:
    result.to_csv('ch06/outNoRowColLabel', index=False, header=False)

    # Can also write only a subset of the columns, and in an order of your choosing:
    result.to_csv('ch06/outWithSomeCols', index=False, columns=['a','b','c'])

    # Series also has a to_csv method:
    dates = pd.date_range('1/1/2000', periods=7)
    ts = Series(np.arange(7), index=dates)
    ts.to_csv('ch06/tseries.csv')

    print(Series.from_csv('ch06/tseries.csv', parse_dates=True), '\n')

    #####
    # Manually Working with Delimited Formats (p163)
    #####

    # Most forms of tabular data can be loaded from disk using functions like pd.read_table.
    # However in some cases, manual processing may be necessary.
    # It's not uncommon to receive a file with one or more malformed lines that trip up read_table.
    # To illustrate basic tools...:

    # For any files with a single-character delimiter, you can use Python's built-in csv module.
    # To use it, pass any open file or file-like object to csv.reader:

    f = open('ch06/ex7.csv')
    reader = csv.reader(f)

    # Iterating through the reader like a file yields tuples of values in each like with any quote chars removed:

    for line in reader:
        print(line)
    print()

    # From there, it's up to us to do the wrangling necessary to put the data in the form that you need it. ie:

    lines = list(csv.reader(open('ch06/ex7.csv')))
    header, values = lines[0], lines[1:]
    data_dict = {h: v for h, v in zip(header, list(zip(*values)))}
    print(data_dict, '\n')

    # Defining a new format with a different delimiter, string quoting convention, or line terminator is done by
    # defining a simple subclass of csv.Dialect:

    # class my_dialect(csv.Dialect):
    #     lineterminator = '\n'
    #     delimiter = ';'
    #     quotechar = '\"'
    # reader = csv.reader(f, dialect=my_dialect)

    # Individual CSV dialect parameters can also be given as keywords to csv.reader without having to define a subclass:
    #reader = csv.reader(f, delimiter='|')

    # Possible options (attributes of csv.Dialect) and what they do can be found in Table 6-3 on page 164.

    # To write delimited files manually, you can use csv.writer. It accepts an open, writable file object and the same
    # dialect and format options as csv.reader:

    # with open('mydata.csv', 'w') as f:
    #     writer = csv.writer(f, dialect=my_dialect)
    #     writer.writerow(('one','two','three'))
    #     writer.writerow(('1','2','3'))
    #     writer.writerow(('4','5','6'))
    #     writer.writerow(('7','8','9'))

    #####
    # JSON Data
    #####

    # JSON (short for JavaScript Object Notation) has become one of the standard formats for sending data by HTTP
    # request between web browsers and other applications. It is a much more flexible data format than a tabular
    # text form like CSV.

    # JSON is very nearly valid Python code with the exception of its null value null and some other nuances, such as
    # disallowing trailing commas at the end of lists.

    # obj = """
    # {"name": "West",
    #  "places_lived": ["United States", "Spain", "Germany"],
    #  "pet": null,
    #  "siblings": [{"name": "Scott", "age": 25, "pet": "Zuko"},
    #               {"name": "Katie", "age": 33, "pet": "Cisco"}]
    # }
    # """

    #result = json.loads(obj)
    #print result, '\n'

    #asjson = json.dumps(result)

    #siblings = DataFrame(result['siblings'], columns=['name', 'age'])

    #####
    # XML and HTML: Web Scraping p166
    #####

    # To get started, find the URL you want to extract data from, open it with urllib2 and parse the stream
    # with lxml like so:

    parsed = parse(urlopen('https://finance.yahoo.com/quote/AAPL/options'))
    doc = parsed.getroot()

    links = doc.findall('.//a')
    print(links[15:20], '\n')

    lnk = links[10]
    print(lnk, '\n')
    print(lnk.get('href'), '\n')
    print(lnk.text_content(), '\n')

    # Thus, getting a l ist of all URLs in the document is a matter of writing this list comprehension:
    urls = [lnk.get('href') for lnk in doc.findall('.//a')]
    print(urls[-10:], '\n')

    # Finding the right tables in the doc can be a matter of trial and error; some websites make it easier
    # by giving a table of interest an 'id' attribute. I determined that these were the two tables containing
    # the call data and put data, respectively:

    tables = doc.findall('.//table')
    calls = tables[1]
    puts = tables[2]

    # Each table has a header row followed by each of the data rows:
    rows = calls.findall('.//tr')
    # For the header as well as the data rows, we want to extract the text from each cell; in the case of the header
    # these are 'th' cells and 'td' cells are for the data:
    def _unpack(row, kind='td'):
        elts = row.findall('.//%s' % kind)
        return [val.text_content() for val in elts]

    print(_unpack(rows[0], kind='th'), '\n')
    print(_unpack(rows[1], kind='td'), '\n')

    # Now it's a matter of combining all of these steps together to convert this data into a DF.
    # Since the numerical data is still in string format, we want to convert some, but perhaps not all, of the columns
    # to floating point format. Could do it manually, but pandas has a class TextParser that is used internally
    # in the read_csv and other parsing functions to do the appropriate automatic type conversion:

    # def parse_options_data(table):
    #     rows = table.findall('.//tr')
    #     header = _unpack(rows[0], kind='th')
    #     data = [_unpack(r) for r in rows[1:]]
    #     return TextParser(data, names=header).get_chunk()
    #
    # # Finally, we invoke this parsing function on the lxml table objects and get DF results:
    # call_data = parse_options_data(calls)
    # put_data  = parse_options_data(puts)
    # print call_data[:10]

    # Line 226 just doesn't wanna work for whatever reason.

    #####
    # Interacting with HTML and Web APIs
    #####

    # url = 'https://twitter.com/search?q=e3&src=typd'
    # resp = requests.get(url)
    # print resp, '\n'
    # data = json.loads(resp.text)
    # print data.keys(), '\n'
    #
    # tweet_fields = ['created_at', 'from_user', 'id', 'text']
    # tweets = DataFrame(data['results'], columns=tweet_fields)
    # print tweets, '\n'
    # print tweets.ix[7], '\n'

    # All of the above with twitter fails because twitter changed the way their API works. Much harder to get now.

    #####
    # Interacting with Databases
    #####

    import sqlite3
    query = """
    CREATE TABLE test
    (a VARCHAR(20), b VARCHAR(20),
     c REAL,        d INTEGER
     );"""
    con = sqlite3.connect(':memory:')
    con.execute(query)
    con.commit()

    data = [('Atlanta', 'Georgia', 1.25, 6),
            ('Tallahassee', 'Florida', 2.6, 3),
            ('Sacramento', 'California', 1.7, 5)]
    stmt = "INSERT INTO test VALUES(?, ?, ?, ?)"

    con.executemany(stmt, data)
    con.commit()

    cursor = con.execute('select * from test')
    rows = cursor.fetchall()
    print(rows, '\n')

    print(cursor.description, '\n')

    print(DataFrame(rows, columns=zip(*cursor.description)[0]), '\n')

    # This is all stuff we don't want to repeat every time we query the database.
    # pandas has a read_frame function in its pandas.io.sql module that simplifies the process.
    # Just pass the select statement and the connection object:

    #print sql.read_frame('select * from test', con), '\n' # Not working for some reason. :thinking:

    # End of Chapter 6!