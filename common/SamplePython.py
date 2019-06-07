#package com.scripts

#object SamplePython {
  
import argparse

class sumeet_class(object):
    def __init__(self, name, balance, bank_name):
        self.name = name
        self.balance = balance
        self.bank_name = bank_name

    def withdraw(self, amount):
        if amount > self.balance:
            raise RuntimeError("Amount greater than available balance")
        self.balance -= amount
        return self.balance

    def deposit(self, amount):
        self.balance += amount
        return self.balance

    @property

    def name_return(self):
        return self.name

    def balance_return(self):
        return self.balance

    def bank_return(self):
        return self.bank_name

def main(argv=None):

        parser = argparse.ArgumentParser()
        parser.add_argument("-name", "--name", choices=["sumeet"],
                            help="name of the person")

        parser.add_argument("-balance", "--balance", choices=[10, 20, 30],
                            type=int, help = "Initial balance")

        parser.add_argument("-bank1", choices=["JPM", "HDFC", "AXIS"], dest="bank_name",
                            help="name of the person", default=False)

        args = parser.parse_args() if not argv else parser.parse_args(argv)
        print(args)
        sumeet_obj = sumeet_class(**vars(args))
        print(sumeet_obj.withdraw(10))
        #print("Bank_name==============="+ args.bank_name)
        if not sumeet_obj.bank_return():
            print("noooo")
        else:
            print("Yessssss")

if __name__ == "__main__":
     main()
     s = "Hello {name} how are you? which {class1} you study in?"
     print(s)
     m = s.format(name = "Sumeet" , class1 = "10th")
     print(m)
  
#}