import mysql.connector

def select_table():
  mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="sample_db",
    auth_plugin="mysql_native_password"
  )

  mycursor = mydb.cursor()

  
  mycursor.execute("SELECT * FROM customer")
  myresult = mycursor.fetchall()
  print(myresult)
  return myresult


select_list = select_table()
print('-------------------------')
for record in select_list:
  print(record)
print('-------------------------')

print(select_list[-1][2])
num = int(select_list[-1][2]) + 1
print(num)
