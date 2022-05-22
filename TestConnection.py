import mysql.connector


mydb = mysql.connector.connect(
    host="nhl-db.cvykkhznbvzb.us-east-2.rds.amazonaws.com",
    user="admin",
    password="3shh5vZR6nT4t9YmdjhU",
    database="nhl",
    port=3306
)

mycursor = mydb.cursor()
sql = "select * from new_shot_table limit 1"
mycursor.execute(sql)
results = mycursor.fetchall()
print(results)