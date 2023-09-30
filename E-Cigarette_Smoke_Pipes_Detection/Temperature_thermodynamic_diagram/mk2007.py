import pandas as pd
import sqlite3

conn = sqlite3.connect("./MK2007.db")
cursor = conn.cursor()

get_attribute_sql = "select name from pragma_table_info('check_data')"

cursor.execute(get_attribute_sql)
result = cursor.fetchall()

name_list = list([i[0] for i in result])
a = ', '.join(name_list)

select_sql = 'select '+a+' from check_data'

cursor.execute(select_sql)
data = cursor.fetchall()

test = pd.DataFrame(columns=name_list, data=data)

test.to_csv('./MK2007.csv')