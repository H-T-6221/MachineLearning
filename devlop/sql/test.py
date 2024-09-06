import mysql.connector

print('Hello, World')

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="root",
  auth_plugin="mysql_native_password"
)

mycursor = mydb.cursor()

mycursor.execute("CREATE DATABASE test_db")


mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="root",
  database="test_db",
  auth_plugin="mysql_native_password"
)

mycursor = mydb.cursor()

mycursor.execute("CREATE TABLE customers (name VARCHAR(255), address VARCHAR(255))")

print('Bye')
