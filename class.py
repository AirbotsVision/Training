#Operations - string,integer,conversion
class Operations:
 def conversion(self,str):
   print ("OCT:",oct(str))
   print("HEX:",hex(str))

 def integerOperation(self,a,b):
# arithmatic operator
   print(" ARITHMATIC OPERATION")
   print("A+B:",a+b)
   print("A-B:",a-b)
   print("A*B:",a*b)
   print("A%B:",a%b)
   print("A/B:",a/b)
   d=input("Number of times exponential")
   d=int(d)
   print("Exponential of A:",a**d)
   print("Exponential of B:",b**d)
   print("A//B:",a//b)
#comparision operators
   print("\n COMPARISION OPERATIONS")
   if a==b:
     print ("A & B equal")
   if a>b:
      print ("A greater")
   if a<b:
      print ("B greater")
   if a!=b:
      print("A&B not equal")
#bitwise operator
   print("\n BITWISE OPERATIONS")
   print ("AND:",a&b)
   print("OR:",a|b)
   print("EX-OR:",a^b)
   print("Complement of A:" ,~a)
   print("Shift operatons:",a<<2,b>>3)
#identity operators
   print(" IDENTITY OPERATIONS")
   if (a is b):
      print ("Both are same number")
   else:
      print ("A is not present in B")
 def stringOperation(self,s):
    print("checking lower case:",s.islower()) 
    print("checking upper case:",s.isupper())
    print ("Length of the string:",len(s))
    print("Convert string to lower:",s.lower())
    print("Convert string to upper:",s.upper())
op= Operations()
print ("Enter the number")
print ("\n1 for conversion\n 2 for integer operation\n 3 for string operation ")
num=input()
num=int(num)
if num==1:
   str=input("Enter the Number for conversion:")
   str=int(str)
   op.conversion(str)
if num==2:
   a=input("Enter the number A:")
   a=int(a)
   b=input("Enter the number B:")
   b=int(b)
   op.integerOperation(a,b)
if num==3:
   s=input("Enter the string for operation:")
   op.stringOperation(s)
  
 

  
    
